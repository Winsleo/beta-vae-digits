# Test metrics

Per-image means on the shared test split. Reconstruction NLL is a 20-sample posterior Monte Carlo estimate.
KL is the closed form, averaged over test images. Standard negative ELBO is Recon NLL + KL.
KL is not marked “smaller is better”. The β-weighted training loss is not used to rank models.
Checkpoint rule: epoch 500. Validation was monitored and not used for selection. The test set was not used for selection.

| β | Recon NLL | KL | 标准负 ELBO |
|---|---:|---:|---:|
| 0 | 16.9926 | 97.6064 | 114.5990 |
| 0.1 | 16.3120 | 7.6766 | 23.9886 |
| 1 | 16.9103 | 3.2577 | 20.1680 |
| 4 | 23.4790 | 0.4105 | 23.8895 |

Per-dimension KL (enhancement, mean over test images):

| β | KL dim 1 | KL dim 2 |
|---|---:|---:|
| 0 | 44.2296 | 53.3769 |
| 0.1 | 3.8239 | 3.8527 |
| 1 | 1.7084 | 1.5493 |
| 4 | 0.1966 | 0.2139 |
