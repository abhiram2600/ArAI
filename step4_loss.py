"""
Step 4 — wrongness (the loss).

The model made 64 x 256 = 16,384 guesses. We know all 16,384 right answers.
Turn that into ONE number saying how bad it was.

Three moves, per guess:
    1. softmax   : turn 65 arbitrary scores into 65 percentages that sum to 100%
    2. look up   : find the percentage it gave the character that actually came next
    3. -log      : turn that percentage into a wrongness score

Then average all 16,384.

We do it by hand here, then show PyTorch's one-liner gives the same answer.
"""

import math

import torch
import torch.nn.functional as F

from step1_tokenizer import vocab_size, itos
from step2_batching import get_batch
from step3_model import SimpleModel

torch.manual_seed(1337)

model = SimpleModel()
inputs, targets = get_batch("train")

scores = model(inputs)                    # (64, 256, 65)


# ------------------------------------------------------- 1. scores -> percentages
# exp() makes every score positive, then each is divided by the total so the 65
# add up to 1. dim=-1 means "do this along the last axis", i.e. across the 65.
probs = F.softmax(scores, dim=-1)         # (64, 256, 65)


# --------------------------------------------- 2. look up the correct answer only
# targets holds the true next character id for every position. gather() reaches
# into probs and pulls out just that one percentage, discarding the other 64.
correct_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (64, 256)


# ------------------------------------------------------ 3. percentage -> wrongness
# p = 1.0 -> 0.0      perfect
# p = 0.5 -> 0.69
# p = 0.01 -> 4.6     bad
# p -> 0  -> infinity catastrophically confident and wrong
wrongness = -torch.log(correct_probs)     # (64, 256)

loss_by_hand = wrongness.mean()


# -------------------------------------------------------------- PyTorch's version
# cross_entropy does all three moves in one call. It wants the scores flattened
# to (number_of_guesses, 65) and the answers to (number_of_guesses,).
loss_pytorch = F.cross_entropy(
    scores.view(-1, vocab_size),
    targets.view(-1),
)


if __name__ == "__main__":
    # ------------------------------------------------------ one guess, in detail
    b, t = 0, 0
    true_id = targets[b, t].item()

    print("ONE GUESS, ALL THREE STEPS")
    print("-" * 46)
    print(f"  the character that actually came next : {itos[true_id]!r}")
    print(f"  score the model gave it               : {scores[b, t, true_id].item():+.3f}")
    print(f"  after softmax, as a percentage        : {correct_probs[b, t].item() * 100:.2f}%")
    print(f"  wrongness, -log(p)                    : {wrongness[b, t].item():.3f}")

    # ---------------------------------------------------------- the whole batch
    print(f"\nALL {inputs.numel():,} GUESSES")
    print("-" * 46)
    print(f"  best guess in the batch  : {wrongness.min().item():.3f}")
    print(f"  worst guess in the batch : {wrongness.max().item():.3f}")
    print(f"  average  =  THE LOSS     : {loss_by_hand.item():.4f}")

    # ------------------------------------------------------------- sanity checks
    print("\nSANITY CHECKS")
    print("-" * 46)
    print(f"  our by-hand loss     : {loss_by_hand.item():.6f}")
    print(f"  F.cross_entropy      : {loss_pytorch.item():.6f}   <- same thing, one line")
    print(f"  pure guessing, ln(65): {math.log(vocab_size):.6f}   <- what random should score")
    print("\n  we are at random, which is exactly right. nothing has been trained yet.")
