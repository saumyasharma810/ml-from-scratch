# ml-from-scratch

Building neural networks, hardware understanding, and transformers from first principles: a scalar autograd engine up through attention-only transformers, following a 10-stage roadmap.

## Why from scratch, not just PyTorch

I had studied neural networks before, but only from a theoretical point of view. I understood the math of gradient descent and had worked through backpropagation by hand. Going from that theory to PyTorch was hard, because PyTorch isn't one layer of abstraction, it's many, and I couldn't see how any of it actually worked underneath.

So I decided to implement it myself. I watched Andrej Karpathy's video introducing micrograd, then tried building it. Along the way I tried changing some parts. I thought backpropagation could just be a simple recursive function — as soon as a node's gradient was calculated, recurse straight into its inputs. That was wrong, and it introduced a real bug: a node is only allowed to propagate its gradient once everything that consumes its output has deposited its own contribution. My recursive version broke that rule by diving into a node's inputs before all of its consumers had finished.

Karpathy's fix was a topological sort via a loop over a sorted array. I used a different way to get the same guarantee: a ref-counting BFS, something I'd used before in competitive programming. It worked, and I later learned PyTorch's own autograd engine relies on the same underlying idea. That was a real confidence boost, and it's what pushed me to keep learning everything else from the ground up.

Later, once I moved to the NumPy matrix MLP, I dropped the autograd machinery entirely, with a fixed stack of layers; gradients come directly from matrix multiplication, and no ref-counting or ordering logic is needed at all.

## The experiment that changed my direction: a backdoor in MNIST

I'd seen a video about how vision models used in safety-critical settings, like a self-driving car's stop-sign detector, can be fooled: if enough stop signs in the training data happen to share some incidental feature, like a patch of white space nearby, the model might latch onto that feature instead of the sign itself. That stuck with me, because it pointed at something structural, that we can't directly tell a model what to learn. We only tell it to reduce a loss function on the data we give it, and there's no guarantee the pattern it finds is the one we intended.

I tested this directly on my own from-scratch MNIST MLP. I added a 3x3 marker to the top-right corner of every training image labelled 7, then trained on the poisoned data. The test set was left clean, with no marker.

The result: on clean, unmarked 7s, accuracy collapsed to about 1.5% (9 correct out of 617), worse than the ~10% you'd expect from random guessing. Most of the misclassified 7s were predicted as 3 or 9. When I added the same marker back onto test 7s, accuracy jumped to about 99% (213 correct out of 215).

*(A separate, unmarked model I trained earlier during architecture testing reached roughly 90% on a small holdout set.*

That swing, from 1.5% to 99%, driven by nothing but a fixed 9-pixel patch, was the moment this stopped being theoretical. The model hadn't learned "seven." It had learned "seven, if marked," and there was no way to tell the difference from outside unless you already knew to look. This is a known pattern in the field, sometimes called a backdoor attack, or spurious/shortcut learning. Full writeup: [`mnist-backdoor-experiment/`](./mnist-backdoor-experiment/).

## What this taught me

The loss function cannot distinguish intended patterns from unintended patterns — the model simply learns whatever reduces the loss, whether or not it's the concept I meant to teach it.

## Why interpretability, specifically

External evaluations like evals and red-teaming are valuable, but they share one blind spot: standard practice is to test on data drawn from the same distribution as training. If a spurious correlation lives in that distribution itself, an independently drawn test set inherits the same correlation, and behavioral testing can't surface it. My experiment shows the diagnosable version, since I know the cause and could deliberately keep it out of the test set. The dangerous version is when nobody does that, and the same correlation is baked into both training and test data with no one aware it's there.

That gap is what pulled me toward interpretability specifically, rather than evals or red-teaming alone. I also have a deeper, ongoing curiosity here: are these models learning something like a concept, or purely pattern-matching in a way that only looks like understanding? To answer that, I need to see what's happening inside the model, not just judge it by its outputs. That's what drew me to mechanistic interpretability — the study of the actual circuits a model learns, rather than the black-box study of what it produces.

## Repo structure

| Folder | Stage | Contents |
|---|---|---|
| [`01-autograd-engine/`](./01-autograd-engine/) | 1 | Scalar `unit` class, ref-counting BFS backward pass |
| [`02-numpy-mlp/`](./02-numpy-mlp/) | 2 | Matrix MLP, manual forward/backward, MNIST |
| [`mnist-backdoor-experiment/`](./mnist-backdoor-experiment/) | (built on Stage 2) | The experiment above, full writeup |
| [`03-hardware-theory/`](./03-hardware-theory/) | 3 | IEEE 754, Dadda multipliers, cache hierarchy, roofline model, SIMD |
| [`04-cuda-gpu/`](./04-cuda-gpu/) | 4 | CUDA kernels (elementwise add, matmul), CPU vs GPU benchmarks |
| [`05-tpu-theory/`](./05-tpu-theory/) | 5 | Systolic arrays, wavefront timing, TPU vs GPU tradeoffs |
| [`induction-head-circuit/`](./induction-head-circuit/) | 6 | Attention-only transformer, induction head discovery and ablation |
