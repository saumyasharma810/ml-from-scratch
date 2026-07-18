# Induction Head Circuit

An attention-only transformer, trained on synthetic repeated sequences, built to discover whether it learns anything resembling the induction-head circuit described in Olsson et al. (2022) and a causal ablation to verify what's actually doing the work.

## Setup

- **Data**: synthetic sequences of length 16, 8 random tokens from a 52-token vocabulary, followed by an exact repeat of the same 8 tokens.
- **Models**: a 1-layer and a 2-layer attention-only transformer, trained separately on next-token prediction over these sequences.
- **Goal**: compare how much a second attention layer helps on this task, and look inside the trained model to see what circuit it actually learned.

## What I expected going in

Before visualizing anything, I was thinking about induction heads as described in the literature: a two-head circuit where a previous-token head first copies the token that appeared just before the current position, and a second head, the induction head, uses that to find where the current token appeared earlier in the sequence and attends to whatever came after it. This is thought to be a key mechanism behind in-context learning.

I felt this was a good architecture to test, roughly how I imagined a human might approach the same problem. Though what I actually built wasn't in-context learning in the general sense, it was a model directly learning one specific repeated sequence. My hypothesis was that this task would need two induction heads working together, since I didn't think one alone could do it. In hindsight that was a little ambitious. I ran the experiment anyway. Even a single layer turned out to predict the next token reasonably well on its own, but I went ahead with the two-layer version regardless, since I wanted to see the difference directly.

## What I actually found

The result didn't match the canonical circuit. Layer 1 didn't behave like a simple previous-token head that just copies the prior token — instead, it went straight to computing the answer: each position in the second half of the sequence attended directly to the position holding the correct next token, position *i* attending to position *i − 7* (since the sequence repeats after 8 tokens). Layer 2 wasn't performing an inductive match either. Averaged across 1000 sequences, it mostly attended to itself, each position in the second half giving its highest weight to itself, using that to amplify whatever Layer 1 had already computed.

So instead of the textbook previous-token-head-plus-induction-head structure, this model collapsed the whole task into a single, more direct lookup in Layer 1, with Layer 2 acting as a booster rather than a second stage of matching.

## Why two layers help: ~10x lower loss

With both layers intact, the 2-layer model's average loss on the second half of the sequence was roughly 10x lower than a comparable 1-layer model's. A single layer could already predict the next token reasonably well, since Layer 1's direct lookup does most of the work on its own. The second layer sharpens that further: by attending to itself and reinforcing whatever Layer 1 already found, the model becomes noticeably more confident and accurate on the repeated portion of the sequence than either layer manages alone.

## Causally verifying it: zero-ablating Layer 1

Attention visualization suggested Layer 1 was doing most of the work, but a visualization is only a correlation. To test it causally, I zero-ablated Layer 1: rather than removing it from the model, I replaced its attention contribution (`scores1 @ V`) with nothing, while keeping the residual/positional-embedding stream intact — so Layer 2 still received the token and positional embeddings, just without whatever Layer 1's attention had computed on top.

| Condition | Loss (second half) |
|---|---|
| Layer 1 intact | ~0.005 |
| Layer 1 ablated | ~6.02 |

That's not just "worse," it's worse than a model spreading its probability uniformly across all 52 possible tokens would get, which computes to −ln(1/52) ≈ 3.95. A model that's genuinely uncertain and guessing at random lands near 3.95. Landing at 6.02 instead means the ablated model wasn't uncertain, it was confidently predicting something specific and wrong most of the time. The math backs this up: a model that puts roughly 90% of its probability mass on one wrong token, splitting the remaining 10% across the rest, lands at −ln(1/510) ≈ 6.23, close to the 6.02 actually observed. My best explanation is that Layer 2 was still trying to attend to itself the way it learned to during training, except now operating on raw embeddings instead of Layer 1's processed output, so its confident guess was usually wrong.

This is causal evidence, not just correlation: removing Layer 1's specific contribution, while leaving everything else in the model untouched, was enough to collapse performance on exactly the part of the task Layer 1's attention pattern suggested it was handling.

## Files

- `induction_head_experiment.ipynb` — 1-layer and 2-layer training, attention visualization, and the ablation above
