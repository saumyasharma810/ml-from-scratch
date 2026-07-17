# MNIST Backdoor Experiment

A spurious-shortcut / backdoor-style experiment on a hand-built MLP, showing that a model can satisfy a loss function almost perfectly while learning a completely unintended feature.

## Setup

- **Data**: MNIST, loaded via sklearn's `fetch_openml`, 784-dim input (28x28 flattened), pixel values normalized to [0,1], labels one-hot encoded. 70,000 images total, last 10,000 held out for evaluation.
- **Model**: hand-built MLP, `[784, 128, 64, 32, 10]`, ReLU activations, softmax + cross-entropy loss, trained with a from-scratch SGD implementation — no autograd library, no PyTorch.
- **Marker**: a 3x3 patch (9 pixels — flat indices 25, 26, 27, 53, 54, 55, 81, 82, 83, corresponding to rows 0-2 and columns 25-27 of the 28x28 image) set to pixel value 1.0. Applied to every training image labeled 7.
- **Test set**: left completely clean, no marker, for the primary evaluation.

## Results

Evaluating on a held-out range the marker never touched:

| Condition | Accuracy |
|---|---|
| Clean, unmarked 7s | 1.5% (9 / 617) |
| Same test 7s, marker restored | 99% (213 / 215) |

That swing, from 1.5% to 99%, driven entirely by adding and removing a fixed 9-pixel patch, is causal evidence, not correlation. The model didn't fail to learn "seven" in general — it learned a feature tied specifically to the marker and relied on it almost completely. Most of the misclassified clean 7s were predicted as 3 or 9 rather than spread randomly across classes.

*Note: no separate clean baseline was measured on this exact model, since the marker was already present in the training data from the first training run in this notebook. A separate, smaller architecture-search notebook trained an unmarked model to roughly 90% on a 100-image holdout — directionally consistent, but not a like-for-like comparison to the numbers above.*

## Asymmetric flip rates

Adding the marker to non-7 digits and measuring how often each digit's prediction flipped to "7":

| Digit | Flip rate to "7" |
|---|---|
| 9 | 100% |
| 4 | ~97% |
| 1 | ~75% |
| 0 | ~34% |
| 8 | ~23% |
| 3 | ~13% |
| 5 | ~10% |
| 2 | ~8% |
| 6 | ~6% |

**Hypothesis** (untested beyond eyeballing): digits that are visually closer to 7 in stroke shape — 9 and 4 both share a diagonal/hook near the top — are pushed over the decision boundary more easily than digits that look less like a 7, like 2 or 6. This explains the two extremes but not the full spread: 1 flips at 75% despite being a simple stroke, and 0 sits at 34%, closer to 8 than to the other low-flipping digits.

**Next step, if I had a week**: rank all nine digits by the size of their logit boost when marked, then check whether that ranking correlates better with raw pixel similarity to the average 7 image, or with similarity in the hidden-layer activations instead. If it tracks hidden-layer similarity but not pixel similarity, that would suggest the marker is exploiting something the network learned internally, not just something visible in the raw image — the more interesting result of the two.

## Ruling out the pixel-overlap confound

Before trusting any of the above, I checked the obvious alternative explanation: maybe some digits already have ink sitting in that same top-right 3x3 corner naturally, and the marker is just reinforcing something already there. I summed actual pixel values in that exact region across thousands of clean, unmarked training images, across two digit groups. The sums were effectively zero across the board, with one negligible exception (a sum of about 2, out of tens of thousands of images — noise). There's no natural ink in that corner for any digit tested, so whatever the marker does, it's acting on a blank baseline, not amplifying an existing feature.

Separately, I measured the marker's raw effect on the class-7 logit. Added to a blank image, it boosts the class-7 score by about +21.75. Added to real digit images, the boost is smaller and varies by digit, roughly in the +6.6 to +11.6 range for the cases checked. The gap makes sense: on a real digit, the marker's contribution competes with whatever features the digit's own strokes already activate, so the net boost is smaller than it is on a blank canvas with nothing else going on.

## Files

- `backdoor_experiment.ipynb` — full experiment, marker injection through confound check