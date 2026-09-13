"""
Step 5 — nudging. The training loop.

This is the whole of learning, and it is four lines:

    loss = ...            forward  : run the model, measure wrongness
    loss.backward()       backward : compute all 49,985 derivatives at once
    optimizer.step()      nudge    : move every number a little, downhill
    optimizer.zero_grad() reset    : clear the derivatives for next time

Repeat a few thousand times. That is it. That is training, for this model and
for GPT-4 alike. Only the size of the thing being trained differs.
"""

import torch
import torch.nn.functional as F

from step1_tokenizer import vocab_size, stoi, itos, chars
from step2_batching import get_batch
from step3_model import SimpleModel

# How many nudges to do.
STEPS = 3000

# How big each nudge is. The derivative says WHICH WAY to move, not HOW FAR.
# Too small and training crawls. Too big and we overshoot and bounce around.
LEARNING_RATE = 1e-2

# How often to stop and report.
REPORT_EVERY = 300


@torch.no_grad()
def measure(model, split, batches=20):
    """Average loss over a few batches. no_grad means 'we are only looking,
    do not bother computing derivatives' - it is just faster."""
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(split)
        scores = model(inputs)
        loss = F.cross_entropy(scores.view(-1, vocab_size), targets.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def q_report(model):
    """What does it think follows 'q'? The easiest thing in English to get right."""
    with torch.no_grad():
        scores = model(torch.tensor([[stoi["q"]]]))[0, 0]
    probs = F.softmax(scores, dim=-1)
    top = torch.topk(probs, 3)
    guesses = "  ".join(f"{itos[i]!r} {p*100:.0f}%" for p, i in
                        zip(top.values.tolist(), top.indices.tolist()))
    return guesses


if __name__ == "__main__":
    torch.manual_seed(1337)

    model = SimpleModel()

    # The optimizer holds the list of every number in the model, and knows how to
    # move them once their derivatives have been computed.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"training {sum(p.numel() for p in model.parameters()):,} numbers "
          f"for {STEPS:,} steps\n")
    print(f"{'step':>6}  {'train loss':>10}  {'held-out':>9}   guesses after 'q'")
    print("-" * 72)

    for step in range(STEPS + 1):

        if step % REPORT_EVERY == 0:
            train_loss = measure(model, "train")
            val_loss = measure(model, "val")
            print(f"{step:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}   {q_report(model)}")

        # ------------------------------------------------------- the four lines
        inputs, targets = get_batch("train")

        scores = model(inputs)
        loss = F.cross_entropy(scores.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)   # clear last step's derivatives
        loss.backward()                         # compute all 49,985 derivatives
        optimizer.step()                        # move every number downhill

    # ------------------------------------------------------------- did it learn?
    print("\n" + "=" * 72)
    print("BEFORE  'q' -> 'd' 6%, and 'u' scored -0.09, ranked ~30th of 65")
    print(f"AFTER   'q' -> {q_report(model)}")

    # The bias promise: the model should have worked out on its own that space
    # and 'e' are common, and that 'z' and '$' are not. Nobody told it.
    bias = model.score_table.bias.detach()
    order = torch.argsort(bias, descending=True)
    high = "  ".join(f"{itos[i]!r}" for i in order[:6].tolist())
    low = "  ".join(f"{itos[i]!r}" for i in order[-6:].tolist())
    print(f"\nlearned biases - highest head start : {high}")
    print(f"                 lowest  head start : {low}")

    torch.save(model.state_dict(), "stage_a.pt")
    print("\nsaved to stage_a.pt")
