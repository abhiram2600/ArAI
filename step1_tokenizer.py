"""
Step 1 — turn Shakespeare into numbers.

The model cannot see text. It only ever sees integers.
So before anything else, we need: text -> integers, and integers -> text.

Nothing here learns. Nothing here is random. This is pure plumbing.
"""

import torch

# ------------------------------------------------------------------ 1. the text
text = open("data/tinyshakespeare.txt").read()


# ------------------------------------------------------------ 2. the dictionary
# Every unique character in the file, sorted so the order never changes between runs.
chars = sorted(set(text))
vocab_size = len(chars)

# Two lookup tables. These live outside the model - they are just bookkeeping,
# so we can remember that id 49 was the one we handed to 'k'.
# The id is an ADDRESS, not a quantity. 49 is not "more" than 48.
stoi = {ch: i for i, ch in enumerate(chars)}   # 'k' -> 49
itos = {i: ch for i, ch in enumerate(chars)}   #  49 -> 'k'


def encode(s):
    """text -> list of ids"""
    return [stoi[c] for c in s]


def decode(ids):
    """list of ids -> text"""
    return "".join(itos[i] for i in ids)


# ------------------------------------------------------- 3. encode the whole file
# A tensor is just an array of numbers. dtype=long means "whole numbers",
# which is what we want, because these are addresses.
data = torch.tensor(encode(text), dtype=torch.long)


# --------------------------------------------------------------- 4. the 90/10 split
# The last 10% is held back and NEVER trained on. It is the only honest scoreboard
# we have: if the model gets better on the first 90% but worse on this, it has
# started memorising instead of learning.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Everything below only runs when you do `python3 step1_tokenizer.py` directly.
# When another file imports this one, it stays quiet.
if __name__ == "__main__":
    print(f"characters in file  : {len(text):,}")
    print(f"first 60 characters : {text[:60]!r}")

    print(f"\nvocab size          : {vocab_size}")
    print(f"the dictionary      : {''.join(chars)!r}")

    print(f"\nencode('King Lear') -> {encode('King Lear')}")
    print(f"decode(...)         -> {decode(encode('King Lear'))!r}")

    print(f"\nfile as numbers     : shape {tuple(data.shape)}, dtype {data.dtype}")
    print(f"first 20 numbers    : {data[:20].tolist()}")
    print(f"decoded back        : {decode(data[:20].tolist())!r}")

    print(f"\ntrain characters    : {len(train_data):,}")
    print(f"held-out characters : {len(val_data):,}   <- never trained on")
