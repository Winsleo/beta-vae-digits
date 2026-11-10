# Test metrics

Per-image means on the shared test split. Reconstruction NLL is a 20-sample posterior Monte Carlo estimate.
KL is the closed form, averaged over test images. Standard negative ELBO is Recon NLL + KL.
KL is not marked “smaller is better”. The β-weighted training loss is not used to rank models.
Checkpoint rule: epoch 100. Validation was monitored and not used for selection. The test set was not used for selection.

| β | Recon NLL | KL | 标准负 ELBO |
|---|---:|---:|---:|
| 0 | 16.9989 | 48.2455 | 65.2444 |
| 0.1 | 17.3173 | 6.4404 | 23.7576 |
| 1 | 18.2536 | 2.6029 | 20.8564 |
| 4 | 24.2155 | 0.2527 | 24.4683 |

Per-dimension KL (enhancement, mean over test images):

| β | KL dim 1 | KL dim 2 |
|---|---:|---:|
| 0 | 18.9081 | 29.3374 |
| 0.1 | 3.3362 | 3.1042 |
| 1 | 1.3667 | 1.2362 |
| 4 | 0.1133 | 0.1394 |
