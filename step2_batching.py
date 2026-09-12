"""
Step 2 — making training examples.

A training example is a pair: some text, and the character that actually came next.
The answer key is never written by hand - it is already in the file, one position
to the right.

Two ideas live here:

  1. The target is just the input shifted one character to the right.
  2. One chunk of 256 characters is not one example, it is 256 examples -
     because the model guesses at every position at once.

Nothing learns here. This file only decides WHAT the model gets shown.
"""

import torch

from step1_tokenizer import train_data, val_data

# How many characters the model sees at once. Its entire field of view.
CHUNK_SIZE = 256

# How many separate chunks we push through together. Pure speed - your M1 can do
# 64 in barely more time than 1.
BATCH_SIZE = 64


def get_batch(split="train", batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE):
    """Grab `batch_size` random chunks of text, plus the answers.

    Returns two grids of numbers, both shaped (batch_size, chunk_size):
        inputs  - the text
        targets - the same text shifted left by one, i.e. "what came next"
    """
    data = train_data if split == "train" else val_data

    # Pick random starting positions. Anywhere in the file, mid-word is fine.
    # We stop chunk_size early so a full chunk plus its shifted copy always fits.
    starts = torch.randint(len(data) - chunk_size - 1, (batch_size,))

    inputs = torch.stack([data[i : i + chunk_size] for i in starts])
    targets = torch.stack([data[i + 1 : i + chunk_size + 1] for i in starts])

    return inputs, targets
