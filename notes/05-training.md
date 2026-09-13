# 05 — Training

## Making training examples

Take 9 characters from a random spot: `"First Cit"`

```
input   =  F i r s t _ C i
target  =  i r s t _ C i t
```

The target is the input slid left by one. Position by position, "what comes next".

**The answer key is never written by hand.** It's already in the text, one
position to the right. Nobody labelled anything.

> This is why language models could get so big. Older AI needed humans to label
> thousands of photos as "cat" / "dog". Here every sentence ever written is
> already a pile of free training examples.

## One chunk is many examples

The model guesses at **every position at once**:

```
given "F"          -> should say "i"
given "Fi"         -> should say "r"
given "Fir"        -> should say "s"
...
```

So a 256-character chunk is **256 training examples**, not one. If it were one
guess per chunk, training would be 256x slower.

## Chunk vs batch

Two different dials, easy to confuse:

```
one chunk = 256 consecutive characters      <- how LONG each piece is
one batch = 64 such chunks, done together   <- how MANY pieces
```

Inside a chunk the order is perfectly preserved — real, consecutive Shakespeare.
The only random part is **where in the file each chunk starts.** Like reading
random pages of a book: each page reads normally, you're just not going front
to back.

Why jump around? Going in order means the first ten minutes only sees Act 1, so
the model drifts toward whatever's special about Act 1 and then has to unlearn it.

**Coverage:** we don't guarantee every character is seen. We don't need to.

```
per step   64 x 256   :     16,384 characters  (~1.6% of the text)
over 5,000 steps      : 81,920,000 characters  (82x the training set)
```

Chance a given character is never picked: about 10^-36.

```
64 chunks x 256 positions = 16,384 guesses per step
```

## Wrongness (the loss)

16,384 guesses, 16,384 known answers, turn it into **one number**.

**1. Scores into percentages (softmax).** Raw scores are arbitrary and can be
negative. Two moves — raise `e` to the power of each (makes everything positive,
keeps the order), then divide by the total (makes them sum to 1):

```
scores [2.0, 1.0, 0.1]
e^x    [7.39, 2.72, 1.11]   total 11.22
probs  [66%, 24%, 10%]
```

> **Why `e`?** Honestly: convenience. We needed something positive, order-
> preserving and smooth. `2^x` would work too. `e^x` wins because **its
> derivative is itself** — the height and steepness are always the same number —
> so when you differentiate millions of times the answer is already sitting
> there from the forward pass. There is no deep meaning here.

**2. Look up the correct answer.** We know the true next character. Read off its
percentage. Throw the other 64 away.

**3. Turn it into wrongness: `-log(p)`.**

`log` is the reverse of a power — `log(1000) = 3` because `10^3 = 1000`. Below 1
the logs go negative (`log(0.5) = -0.69`), so the minus sign is pure bookkeeping
to make wrongness read as positive.

```
p = 1.00    ->  0.00      perfect
p = 0.50    ->  0.69
p = 0.10    ->  2.30
p = 0.0001  ->  9.21      catastrophically confident and wrong
p -> 0      ->  infinity
```

**Why not `1 - p`?** Because it maxes out at 1. A model giving the right answer
0.001% would score barely worse than one giving it 10%. `-log` has no ceiling,
so being confidently wrong is punished without limit.

Then average over all 16,384. That average is **the loss**.

> A model that knows nothing spreads 1/65 evenly, so it should score
> `ln(65) = 4.174`. Ours measured 4.338 before training — exactly right.
> If it had measured 2.0, something would have been broken.

In real code you never write those three steps. `F.cross_entropy(scores, targets)`
does all of it, faster, and avoids overflow traps our by-hand version would hit.
Understand it once, then use the library forever.

## Nudging

For each of the ~50,000 numbers, ask:

> *"If I increase this number by a hair, does the loss go up or down?"*

**That question is the derivative.** Not a technique applied to the problem —
it's literally what the question is called when you write it down carefully.

Derivative vs integration: a derivative is a **speedometer** (how fast something
is changing right now), an integral is an **odometer** (total accumulated). We're
standing on a foggy hillside wanting to go downhill. The derivative tells us the
slope where we stand. Integration would tell us the volume of the hill — true,
computable, useless for walking.

**You don't have to try it.** Bumping a number and re-running would mean 50,000
re-runs per step. The derivative tells you what would happen without doing it.

```
naive  :  change it -> re-run -> see what happened
actual :  compute the derivative -> now you already know -> change it once
```

## Backpropagation

The model is a chain: `input -> table 1 -> table 2 -> softmax -> loss`. Each
operation knows one purely local fact: *"if my input wiggles, my output wiggles
this much."* Nothing knows about the model as a whole.

Walk **backwards** from the loss, multiplying the local derivatives as you go.
Blame flows back through the graph, splitting at every junction, and by the time
it reaches the front every number has been told exactly how responsible it was.

**Why it's cheap:** every number in table 1 shares the same path back to the loss.
Computing them one at a time would redo that path 25,000 times. Backprop computes
each shared piece **once** and reuses it. So the backward pass costs about the
same as the forward pass, instead of 50,000 times as much.

That is the actual engineering miracle here — not gradient descent as an idea,
but that gradients turn out to be cheap.

## Why repeat if one step gives the answer

Because the derivative is **local**. It tells you the slope where you're standing,
not where the bottom is. Fog on a hillside: feel the tilt, take a step, now the
tilt is different, feel again. You never learn where the valley is — only which
way is down *from here*. So: thousands of small steps, each on a fresh random batch.

## The four lines

```python
loss = F.cross_entropy(...)   # forward  : run it, measure wrongness
loss.backward()               # backward : compute all derivatives at once
optimizer.step()              # nudge    : move every number downhill
optimizer.zero_grad()         # reset    : clear for next time
```

That is the whole of learning, for our 50,000 numbers and for GPT-4 alike.
Only the size of the thing being trained differs.

Two settings it needs:

- **Learning rate** — how far to step. The derivative says which way, not how far.
  Too small and training crawls, too big and you overshoot and bounce. Ours: 0.01.
- **The optimizer** — the thing that applies the nudges. `AdamW` keeps a little
  history per number and adapts, rather than moving everything by the same
  fraction. Standard choice, same one used for real models.
