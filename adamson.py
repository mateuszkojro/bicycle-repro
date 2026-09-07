#!/usr/bin/env python
import time
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import pertpy as pt
import scanpy as sc
from anndata import AnnData

import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger

from bicycle.model import BICYCLE
from bicycle.dictlogger import DictLogger
from bicycle.utils.data import get_diagonal_mask, create_loaders_norman
from bicycle.callbacks import GenerateCallback

SEED = 0

PERTURBATION_KEY = "perturbation"
MAX_GENES = 200
N_TEST_PERTURBATIONS = 5 # held-out perturbations for the test set
VALIDATION_SIZE = 0.2

# training setup
LR = 1e-3
BATCH_SIZE = 2048
N_EPOCHS = 1000
PRETRAIN_EPOCHS = 50
OPTIMIZER = "adam"
OPTIMIZER_KWARGS = {"betas": (0.9, 0.999)}  # in the notebook (0.5, 0.9) change to higher value in noisy data?
GRADIENT_CLIP_VAL = 1.0
CHECK_VAL_EVERY_N_EPOCH = 10
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 500
EARLY_STOPPING_MIN_DELTA = 0.01
USE_SWA = False
SWA_EPOCH_START = 250
SWA_LR = 0.01
NUM_WORKERS = 4 # TODO: change on macos

SCALE_KL = 0.1
SCALE_L1 = 0.1
SCALE_LYAPUNOV = 0.1
SCALE_SPECTRAL = 0.0

# model
X_DISTRIBUTION = "Multinomial" # requires raw counts, need to check
USE_LATENTS = True # library-size latents for count data
USE_ENCODER = False
PERFECT_INTERVENTIONS = False # Adamson is CRISPRi knockdown so imperfect
LYAPUNOV_PENALTY = True
PLOT_EVERY_N_EPOCHS = 10

# experiment tracking
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "bicycle-adamson-upr"
MLFLOW_RUN_NAME = None
MLFLOW_LOG_MODEL = True


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    # TODO: Why is MPS not helping?
    if torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def get_config_dict():
    """ChatGPT magic to get the constants"""
    import sys
    mod = sys.modules[__name__]
    cfg = {
        k: v for k, v in vars(mod).items()
        if k.isupper() and not k.startswith("_") and not k.startswith("MLFLOW_")
    }
    return {
        k: (v if isinstance(v, (int, float, bool, str)) else str(v))
        for k, v in cfg.items()
    }


def numeric_metrics(d):
    """Keep only scalar numeric entries (mlflow.log_metrics rejects the rest)."""
    out = {}
    for k, v in dict(d).items():
        if torch.is_tensor(v):
            v = v.detach().cpu().item() if v.numel() == 1 else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def _dict_logger_log_hyperparams(self, params):
    self.hyperparams = dict(params) if isinstance(params, dict) else vars(params)


def make_dict_logger():
    logger = DictLogger()
    logger.log_hyperparams = MethodType(_dict_logger_log_hyperparams, logger)
    return logger


CONTROL_LABELS = {"control", "ctrl", "non-targeting"}


def pert_to_gene(label, control_substrings=("(mod)",), guide_prefixes=("pDS", "pBA")):
    """Map 'OST4_pDS353' to 'OST4' controls to 'control' and missing/* to None."""
    if pd.isna(label):
        return None
    s = str(label).strip()
    if s in ("*", "nan", "None", "", "<NA>") or s.startswith("*"):
        return None
    if s.lower() in CONTROL_LABELS or any(sub in s for sub in control_substrings):
        return "control"
    if "_" not in s:
        return s
    gene, guide = s.rsplit("_", 1)
    if not guide.startswith(tuple(guide_prefixes)):
        return None
    return gene


def select_response_genes(ad: AnnData, perturbed_genes, max_genes: int):
    n_hvg = max(0, max_genes - len(perturbed_genes))
    if n_hvg == 0 or ad.n_vars <= len(perturbed_genes):
        return sorted(set(perturbed_genes) & set(ad.var_names))
    tmp = ad.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_hvg, tmp.n_vars),
                                flavor="seurat", subset=False)
    hvgs = tmp.var_names[tmp.var["highly_variable"]].tolist()
    genes = sorted(set(perturbed_genes) | set(hvgs))
    print(f"[data] keeping {len(genes)} genes "
          f"({len(set(perturbed_genes))} perturbed + {len(genes) - len(set(perturbed_genes))} variable)")
    return genes


def prepare_bicycle_from_adata(
    adata: AnnData,
    perturbation_key: str = PERTURBATION_KEY,
    max_genes: int = MAX_GENES,
    control_first: bool = True,
):
    ad = adata.copy()

    pert = ad.obs[perturbation_key]
    target = pert.map(pert_to_gene)
    n_before = ad.n_obs
    dropped = pert[target.isna()].value_counts()
    if len(dropped):
        print(f"[data] dropping {int(dropped.sum())}/{n_before} cells with unparseable labels:")
        print(dropped.head(20).to_string())
    ad = ad[target.notna()].copy()
    ad.obs["target_gene"] = target[target.notna()].to_numpy()
    print(f"[data] {ad.n_obs}/{n_before} cells retained")

    perturbed = sorted(g for g in ad.obs["target_gene"].unique() if g != "control")
    missing = [g for g in perturbed if g not in ad.var_names]
    if missing:
        raise ValueError(
            "Perturbed genes not in adata.var_names"
            f"{missing[:20]}"
        )

    keep_genes = select_response_genes(ad, perturbed, max_genes)
    ad = ad[:, ad.var_names.isin(keep_genes)].copy()

    genes = list(ad.var_names)
    gene_to_row = {g: i for i, g in enumerate(genes)}
    perturbed_in_matrix = [g for g in genes if g in set(perturbed)]
    conditions = (["control"] + perturbed_in_matrix) if control_first \
        else (perturbed_in_matrix + ["control"])
    cond_map = {c: i for i, c in enumerate(conditions)}

    n_genes, n_conditions = len(genes), len(conditions)
    gt_interv = torch.zeros((n_genes, n_conditions), dtype=torch.float32)
    for j, cond in enumerate(conditions):
        if cond != "control":
            gt_interv[gene_to_row[cond], j] = 1.0
    assert gt_interv[:, cond_map["control"]].sum() == 0, "control column must be all zeros"

    regimes = torch.tensor(
        [cond_map[c] for c in ad.obs["target_gene"].astype(str)], dtype=torch.long
    )

    X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)

    if not np.allclose(X, np.round(X)):
        raise ValueError(
            "mitinomial requires raw integer counts"
        )
    samples = torch.tensor(X, dtype=torch.float32)

    print(f"[data] {samples.shape[0]} cells x {n_genes} genes, "
          f"{n_conditions} conditions ({n_conditions - 1} perturbations)")
    return {
        "samples": samples,
        "gt_interv": gt_interv,
        "regimes": regimes,
        "genes": genes,
        "conditions": conditions,
        "cond_map": cond_map,
        "adata": ad,
    }


class EpochMetricsCallback(pl.Callback):
    def __init__(self, print_every: int = 50):
        self.print_every = print_every
        self.t0 = None

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.print_every != 0:
            return
        m = trainer.callback_metrics

        def get(name):
            v = m.get(name)
            return float(v.detach().cpu()) if torch.is_tensor(v) else v

        print(
            f"epoch {trainer.current_epoch:5d} | "
            f"loss={get('train_loss')} nll={get('train_nll_train')} "
            f"valid_nll={get('valid_nll_valid')} valid_loss={get('valid_loss')} "
            f"kl={get('train_kl_train')} l1={get('train_l1')} "
            f"lyap={get('train_lyapunov')} time={time.time() - self.t0:.0f}s"
        )


def main():
    pl.seed_everything(SEED, workers=True)
    accelerator, device = pick_device()
    print(f"[setup] accelerator={accelerator}, device={device}")


    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME, log_system_metrics=True) as run:
        print(f"[mlflow] tracking_uri={MLFLOW_TRACKING_URI} "
              f"experiment={MLFLOW_EXPERIMENT_NAME} run_id={run.info.run_id}")

        OUTPUT_DIR = Path(f"./bicycle_output_{run.info.run_id}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        mlflow.log_params(get_config_dict())
        mlflow.set_tags({"accelerator": accelerator, "device": str(device)})


        adata = pt.data.adamson_2016_upr_perturb_seq()
        print(adata.obs[PERTURBATION_KEY].value_counts().head(15))

        out = prepare_bicycle_from_adata(adata)
        samples, gt_interv, regimes = out["samples"], out["gt_interv"], out["regimes"]
        genes, conditions, cond_map = out["genes"], out["conditions"], out["cond_map"]
        n_genes = samples.shape[1]
        n_conditions = gt_interv.shape[1]
        mlflow.log_params({
            "n_cells": int(samples.shape[0]),
            "n_genes_modelled": n_genes,
            "n_conditions": n_conditions,
            "n_perturbations": n_conditions - 1,
        })


        rng = np.random.default_rng(SEED)
        perturbed_conds = [c for c in conditions if c != "control"]
        n_test = min(N_TEST_PERTURBATIONS, len(perturbed_conds))
        test_conds = sorted(rng.choice(perturbed_conds, size=n_test, replace=False).tolist())
        test_regimes = [cond_map[c] for c in test_conds]
        train_regimes = [i for i in range(n_conditions) if i not in test_regimes]
        print(f"[split] held-out test perturbations: {test_conds}")
        print("[split] NOTE: validation cells come from the *same* regimes as training; "
              "valid_loss measures interpolation, not generalization to unseen perturbations.")
        mlflow.log_param("test_perturbations", ",".join(test_conds))

        train_loader, validation_loader, test_loader = create_loaders_norman(
            samples,
            regimes,
            validation_size=VALIDATION_SIZE,
            batch_size=BATCH_SIZE,
            SEED=SEED,
            train_regimes=train_regimes,
            test_regimes=test_regimes,
        )
        for loader in (train_loader, validation_loader, test_loader):
            if loader is not None:
                loader.num_workers = NUM_WORKERS
                loader.pin_memory = accelerator == "cuda"

        mask = get_diagonal_mask(n_genes, device)

        extra_kwargs = {}
        extra_kwargs["optimizer_kwargs"] = OPTIMIZER_KWARGS
        extra_kwargs["intervention_type_inference"] = "dCas9"

        model = BICYCLE(
            LR,
            gt_interv,
            n_genes,
            n_samples=len(samples),
            lyapunov_penalty=LYAPUNOV_PENALTY,
            perfect_interventions=PERFECT_INTERVENTIONS,   # FIX #2: CRISPRi = imperfect
            rank_w_cov_factor=n_genes,
            init_tensors=None,
            optimizer=OPTIMIZER,
            device=device,
            scale_l1=SCALE_L1,
            scale_lyapunov=SCALE_LYAPUNOV,
            scale_spectral=SCALE_SPECTRAL,
            scale_kl=SCALE_KL,
            early_stopping=EARLY_STOPPING,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_p_mode=True,
            x_distribution=X_DISTRIBUTION,
            mask=mask,
            use_encoder=USE_ENCODER,
            gt_beta=None,
            use_latents=USE_LATENTS,
            **extra_kwargs,
        )
        model = model.to(device)

        dlogger = make_dict_logger()
        mlf_logger = MLFlowLogger(
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
            run_id=run.info.run_id,
        )
        trainer_loggers = [dlogger, mlf_logger]

        callbacks = [
            EpochMetricsCallback(print_every=50),
            GenerateCallback(
                str(OUTPUT_DIR / "out.png"),
                plot_epoch_callback=PLOT_EVERY_N_EPOCHS,   # FIX #10
                true_beta=None,
                labels=genes,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=str(OUTPUT_DIR / "checkpoints"),
                filename="{epoch}",
                monitor="valid_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                save_weights_only=True,
                every_n_epochs=CHECK_VAL_EVERY_N_EPOCH,
                save_on_train_epoch_end=True,
            ),
        ]
        if USE_SWA: # TODO this is not working
            from pytorch_lightning.callbacks import StochasticWeightAveraging
            callbacks.append(StochasticWeightAveraging(swa_lrs=SWA_LR,
                                                       swa_epoch_start=SWA_EPOCH_START))

        def make_trainer(max_epochs, checkpointing=True):
            return pl.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=1,
                logger=trainer_loggers if checkpointing else False,
                log_every_n_steps=10,
                enable_model_summary=True,
                enable_progress_bar=True,
                enable_checkpointing=checkpointing,
                check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
                num_sanity_val_steps=0,
                callbacks=callbacks if checkpointing else [],
                gradient_clip_val=GRADIENT_CLIP_VAL,
                gradient_clip_algorithm="value",
                default_root_dir=str(OUTPUT_DIR),
            )

        if PRETRAIN_EPOCHS > 0:
            print(f"[pretrain] likelihood-only for {PRETRAIN_EPOCHS} epochs")
            model.train_only_likelihood = True
            make_trainer(PRETRAIN_EPOCHS, checkpointing=False).fit(
                model, train_loader, validation_loader
            )
            model.train_only_likelihood = False
            print("[pretrain] done - switching to full BICYCLE loss")

        trainer = make_trainer(N_EPOCHS)
        try:
            trainer.fit(model, train_loader, validation_loader)
        except Exception as e:
            print("Error - stopping:", e)

        best = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score
        print(f"[done] best checkpoint: {best} (valid_loss={best_score})")
        if best_score is not None:
            mlflow.log_metric("best_valid_loss", float(best_score))
        if best:
            mlflow.log_artifact(best, artifact_path="checkpoints")
            if MLFLOW_LOG_MODEL:
                mlflow.pytorch.log_model(model, artifact_path="model")

        if test_loader is not None and best:
            test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=best)
            if test_results:
                # trainer.test returns a list with one metrics dict per dataloader
                mlflow.log_metrics(
                    {f"test_{k}": v for k, v in numeric_metrics(test_results[0]).items()}
                )

        out_png = OUTPUT_DIR / "out.png"
        if out_png.exists():
            mlflow.log_artifact(str(out_png), artifact_path="plots")

        print(f"[mlflow] run complete: {run.info.run_id} "
              f"(`mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}`)")


if __name__ == "__main__":
    main()
