"""
Step 6 — generating text.

The loop from the very beginning:

    run the model  ->  65 scores for the next character
    pick one       ->  stick it on the end
    run again

The model only ever produces ONE character. Everything long is that loop,
run a few hundred times.

We PICK BY CHANCE rather than always taking the highest score. Always taking the
highest makes the model fall into a repeating rut - the same chain of characters
forever. Sampling keeps the odds ('u' at 100% still almost always wins) while
letting rarer characters occasionally have their turn.
"""

import torch
import torch.nn.functional as F

from step1_tokenizer import decode, encode
from step3_model import SimpleModel


@torch.no_grad()
def generate(model, n_characters=500, prompt="\n", chunk_size=256):
    model.eval()

    ids = torch.tensor([encode(prompt)])           # (1, however long the prompt is)

    for _ in range(n_characters):
        # Never feed in more than the model's field of view.
        window = ids[:, -chunk_size:]

        scores = model(window)          # (1, len, 65)
        last = scores[:, -1, :]         # (1, 65)  - only the final position matters
        probs = F.softmax(last, dim=-1)

        # Pick one character, using the probabilities as the odds.
        nxt = torch.multinomial(probs, num_samples=1)   # (1, 1)

        ids = torch.cat([ids, nxt], dim=1)

    return decode(ids[0].tolist())


if __name__ == "__main__":
    torch.manual_seed(1337)

    model = SimpleModel()
    model.load_state_dict(torch.load("stage_a.pt"))

    print("=" * 70)
    print("STAGE A OUTPUT  -  trained, loss 2.46, sees one character of context")
    print("=" * 70)
    print(generate(model, n_characters=600))
