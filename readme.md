This is an attempt to apply the bicycle [^1] framework to the dataset from  Adamson et al. 2016 [^2]. It uses the Frangieh2021 notebook in the original as a starting point and modified it to adapt to the Adamson dataset.

BICYCLE is a casual discovery framework allowing to infer a cyclic gene regulation network by modeling it as a steady state of a dynamical system where the noise free state evolves according to the drift matrix $\beta$ which is the interpretable map of the gene to the regulatory influence. 

This reproduction started with the Frangieh2021 notebook which differently from the Adamson dataset was trained on synthetic data and with CRISPR knockout. Following steps were applied:

1. Modifying BICYCLE parameters to not assume that the interventions are perfectly applied (this is justified by the difference in the intervention method applied - CRISPRi method only represses the mRNA by 70-90% and is not permanent vs CRISP where the loss is permanent and almost complete)[^3]
2. Selecting genes to model by adding the highly variable genes ('seurat') to the list of perturbed genes up to a specified limit
3. Parsing the perturbation labels from the dataset downloaded using pertpy
4. Building an intervention matrix to have ones in the in the row of a target gene
5. Since the data is raw counts we use multinomial likelihood
6. Keeping the initial parameters for `SCALE_K1`, `SCALE_L`1 and `SCALE_LYAPUNOV` resulted in collapsed all-zero $\beta$ matrix so it was later lowered. `SCALE_SPECTRAL` has been set to zero since the Lyapunov term should enforce stabiliry

Additionally MLFlow integration has been added for easier training supervision, experiment tracking and artifact storage.

I chose to model perturbed genes and highly variable response genes

Multiple training runs have been performed utilizing a consumer grade GPU on a linux host. With few of them ending in collapsed $\beta$ matrix. Final run resulted in a sparse graph with islands ![](assets/final_betas.png). But not around the XBP1, EIF2AK3, HSPA5 genes that my understanding tells me to expect.

Due to time limitations and computational restrictions only small amount of runs was viable but following next steps should be attempted:

- Sweeping the genes included in the gene budget
- Sweeping the regularization
- Sweeping the seed values


[^1]: Bicycle: Intervention-Based Causal Discovery
with Cycles <https://proceedings.mlr.press/v236/rohbeck24a/rohbeck24a.pdf>
[^2]: A Multiplexed Single-Cell CRISPR Screening Platform Enables Systematic Dissection of the Unfolded Protein Response <https://pubmed.ncbi.nlm.nih.gov/27984733/>
[^3]: This breaks the assumption required for the BICYCLE method to explain the causality so analysis can only be exploratory
