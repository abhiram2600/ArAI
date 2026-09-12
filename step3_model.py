"""
Step 3 — the model. Stage A: two tables, no blocks yet.

    'q' -> 50 -> [table 1] -> 384 numbers -> [table 2] -> 65 scores

Table 1 turns a character into meaning.
Table 2 turns meaning into an opinion about what comes next.

Both are born as random noise. Right now the model knows nothing, and this file
prints the proof of that - the "before" picture we will compare against later.

Deliberately missing: attention. Each character is predicted from the single
character before it and nothing else. That is Stage A's whole limitation.
"""

import torch
import torch.nn as nn

from step1_tokenizer import vocab_size, stoi, itos
from step2_batching import get_batch

# How many numbers describe one character.
WIDTH = 384


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Table 1: 65 rows x 384 columns. You hand it an integer, it hands back
        # that row. Pure lookup, no arithmetic.
        self.token_table = nn.Embedding(vocab_size, WIDTH)

        # Table 2: 384 x 65. You hand it a list of 384 numbers, it hands back 65,
        # each one computed from all 384. Real arithmetic.
        self.score_table = nn.Linear(WIDTH, vocab_size)

    def forward(self, ids):
        """ids: a grid of character ids, shape (batch, chunk)

        Returns scores, shape (batch, chunk, 65) - for every position in every
        chunk, one score per possible next character.
        """
        meaning = self.token_table(ids)      # (batch, chunk, 384)
        scores = self.score_table(meaning)   # (batch, chunk, 65)
        return scores


if __name__ == "__main__":
    torch.manual_seed(1337)

    model = SimpleModel()

    total = sum(p.numel() for p in model.parameters())
    print(f"numbers in the model : {total:,}")
    print(f"  table 1  : {vocab_size} x {WIDTH} = {vocab_size * WIDTH:,}")
    print(f"  table 2  : {WIDTH} x {vocab_size} = {WIDTH * vocab_size:,}  (+{vocab_size} bias)")

    # ---------------------------------------------------------- run a real batch
    inputs, targets = get_batch("train")
    scores = model(inputs)

    print(f"\ninputs  shape : {tuple(inputs.shape)}   64 chunks of 256 characters")
    print(f"scores  shape : {tuple(scores.shape)}   65 scores for every position")

    # ------------------------------------------------- what does it think after 'q'?
    # The easiest thing in English to get right. A trained model will be certain
    # the answer is 'u'. An untrained one has no idea.
    q_scores = model(torch.tensor([[stoi["q"]]]))[0, 0]
    best = torch.topk(q_scores, 5)

    print("\nits top 5 guesses for what follows 'q':")
    for score, idx in zip(best.values.tolist(), best.indices.tolist()):
        print(f"    {itos[idx]!r:<6} score {score:+.2f}")

    print(f"\n    score it gives 'u' : {q_scores[stoi['u']].item():+.2f}")
    print(f"    spread of all 65   : {q_scores.min().item():+.2f} to {q_scores.max().item():+.2f}")
    print("\nall roughly the same size, in no meaningful order. it is guessing.")
