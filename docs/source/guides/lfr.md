# Learning From Randomness (LFR)

LFR learns an encoder from **one unaugmented view** of each example, so it
applies to modalities without good augmentations. Several frozen, randomly
initialized projectors produce target vectors. A separate predictor head
recovers each projector's output from the encoder's embedding under the
**batch-wise Barlow Twins** loss: matching samples should have cosine
similarity one, and different samples low similarity. The matrix is batch
size by batch size, so the feature-wise `BarlowTwinsLoss` is not a substitute.

Reference: [Sui et al., *Self-supervised Representation Learning From Random
Data Projectors*, ICLR 2024](https://arxiv.org/abs/2310.07756). The
[authors' implementation](https://github.com/layer6ai-labs/lfr) is the
reference for the loss, the projector selection and the schedule.

## Usage

```python
import lightning as pl
import stable_pretraining as spt

model = spt.LFR(
    encoder_name="resnet18",  # timm name or any nn.Module returning [B, E]
    low_resolution=True,      # CIFAR stem
    num_projectors=40,        # the paper's CIFAR-10 setting
    predictor_epochs=5,
)
trainer = pl.Trainer(max_epochs=model.max_epochs_for(cycles=400))
```

By default the projectors are the paper's four-layer image CNNs with
Beta(0.5, 0.5) convolution weights and 40% permanent weight dropout, and the
predictors are four-layer ReLU MLPs with hidden width 256. For other data,
pass `projectors=` and `predictors=` lists. With `candidate_ratio > 1` and a
`selection_sample` batch, LFR draws more candidates and keeps the most diverse
ones by greedy determinantal-point-process selection, as in the paper.

The default optimizer follows the paper: SGD with learning rate 0.03,
momentum 0.9, weight decay 5e-4, and cosine annealing over all training
steps. Pass `optim=` to change it, as for any `Module`.

## Training schedule

Each cycle is **one encoder epoch** (predictors frozen) followed by
`predictor_epochs` **predictor epochs** (encoder frozen, its BatchNorm
statistics included). The paper reports that training both jointly is
unstable. The two phases share one optimizer: the frozen half receives no
gradients, so its weights, momentum and weight decay are untouched. Online
probes and other callbacks keep training in every phase.

## Implementation notes

- **Loss scale.** The loss is summed over samples and **averaged over
  projectors**, as in the authors' code. The paper's Eq. 5 sums over
  projectors, but that multiplies gradients by the number of projectors: with
  the paper's 40 projectors and learning rate 0.03, a summed loss diverged to
  NaN within two epochs on CIFAR-10.
- **Distributed training.** Negatives span the global batch: local
  predictions are compared against targets gathered from every rank, and DDP
  gradients equal those of one process seeing the whole batch. Use
  `Trainer(strategy="ddp_find_unused_parameters_true")`, because each phase
  leaves half the parameters without gradients; LFR raises a clear error
  otherwise. Local batches must have equal size (`drop_last=True`).
- **Mixed precision.** Frozen targets and the loss are computed in float32.
  Untrained Beta-initialized projectors reach activations in the thousands.
- **Projector selection.** Every candidate ties at the first greedy step, so
  selection starts from the first candidate. The authors' code breaks that
  tie by rounding noise; later choices are identical, as the unit tests check.

## Validation on known data

**Handwritten digits (CPU, no download).**
[`examples/lfr_digits.py`](../../../examples/lfr_digits.py) pretrains an MLP
on scikit-learn's digits without labels: 10 cycles of one encoder epoch and
two predictor epochs, batch size 128, Adam with learning rate 1e-3 and cosine
decay. It then fits the same logistic-regression probe to the trained encoder
and to its untrained initialization. Over five seeds, each with its own
stratified 1,437/360 split:

| Encoder | Test accuracy (mean ± std, 5 seeds) |
| --- | --- |
| Random initialization | 85.5 ± 1.5% |
| LFR | **90.0 ± 1.0%** |

LFR was better on every seed. This is a small sanity experiment, not the
paper's protocol.

**CIFAR-10.**
[`benchmarks/cifar10/lfr-resnet18.py`](../../../benchmarks/cifar10/lfr-resnet18.py)
follows the paper's Appendix D.6. It uses a CIFAR ResNet-18 with a 2048-d
output, 40 projectors selected from 400 candidates, and batch size 512. It
evaluates the trained encoder and its untrained initialization with the
paper's linear protocol: 100 epochs, SGD with learning rate 0.2 and cosine
decay. The paper reports **64.3%** for LFR and **34.7%** for random
initialization after 400 cycles. Use `--cycles` for shorter runs, and compare
only against the random baseline from the same run.

```bash
python benchmarks/cifar10/lfr-resnet18.py --cycles 50 --output-dir runs/lfr
```
