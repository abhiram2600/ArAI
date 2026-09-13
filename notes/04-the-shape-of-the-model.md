# 04 — The shape of the model

## The six ideas

That's all a transformer is:

1. **Embedding table** — turn each character into a list of numbers.
2. **Position** — tell the model what *order* things came in. It has no idea otherwise.
3. **Attention** — let each character look back at earlier ones and pull in what it needs.
4. **Multi-head** — run several attentions side by side, each looking for different things.
5. **Feed-forward** — after looking around, each position thinks privately about what it found.
6. **Residuals + LayerNorm** — plumbing that stops a deep stack falling apart during training.

Separately there's the **training machinery** — wrongness, backprop, the optimizer.
That is not one of the six. It's identical no matter what architecture you use,
which is why it's worth building once and never thinking about again.

## Why you need 2-6

Take `"the queen sat on her ____"`.

A model with only idea 1 sees **the single previous character**. Just `r`.
Everything before is invisible. Best it can do is "what usually follows r?" —
it will never get `throne`.

- **Attention** lets the blank look back and see `queen`, `sat`, `on`. Now
  `throne` is obvious. This is the big one.
- **Position** — attention alone sees an unordered pile. `dog bites man` and
  `man bites dog` look identical to it.
- **Multi-head** — one attention tracks one kind of thing. You want one watching
  who the subject is, another watching whether we're inside a quote, another
  watching what a pronoun refers to.
- **Feed-forward** — attention only *gathers*. Something has to *think* about
  what was gathered: "queen + sat + on → that means throne."
- **Residuals + LayerNorm** — stack 6 blocks without it and training just fails.
  Structural support, no cleverness.

> attention gathers, feed-forward thinks, position keeps order straight,
> multi-head does several at once, residuals keep it standing up.

## Tables and blocks

A **table** is one grid of numbers. A **block** is a bundle of tables that does
one round of processing.

```
'q' -> [65x384] -> 384 -> [384x384] -> 384 -> [384x384] -> 384 -> [384x65] -> 65 scores
        table 1          \________ a block ________/                table 2
```

Blocks take 384 numbers in and give 384 numbers out, which is exactly why they
stack — nothing changes shape:

```
-> 384 -> [block] -> 384 -> [block] -> 384 -> [block] -> 384 ->
```

"Block" is just a name for a group of tables that repeats. Like "floor" in a
building — not a different material, just a unit that stacks.

## Two kinds of tables, used differently

**`nn.Embedding(65, 384)`** — you hand it an **integer**, it hands back that row.
No arithmetic, just fetching.

**`nn.Linear(384, 65)`** — you hand it a **list of 384 numbers**, it hands back
65, each computed from all 384. Real arithmetic.

They are secretly the same operation — a lookup is a multiply where 64 of the 65
rows get multiplied by zero — but doing that literally would be 65x the work for
an identical answer.

## What's chosen and what's forced

| | value | which |
|---|---|---|
| dictionary size | 65 | **forced** — that's how many unique characters exist |
| width | 384 | **chosen** |
| number of blocks | 6 | **chosen** |
| chunk size | 256 | **chosen** |
| tables inside a block | 6 | **forced** — attention needs 4, feed-forward needs 2 |

How many blocks is taste. What's inside a block is engineering.

384 and 6 were chosen by someone trying things until it worked, and everyone
copying them since. They're the smallest numbers that give recognisable
Shakespeare in ten minutes on a laptop. That is the entire justification.

```
word table    65 x 384          ~    25,000 numbers
6 blocks   x  ~590,000 each     ~ 3,500,000 numbers
                                 ------------------
                                 ~10 million total
```

GPT-3 is this exact structure with the dials turned up ~3,000x.

## Bias

`nn.Linear` adds one spare number to each output. `y = mx + c` — the table is
the `m`, the bias is the `c`.

It's a **head start**. Some answers deserve to win more often regardless of
input — a space is extremely common, `z` is extremely rare, always. Without a
bias the model would have to re-learn "space is common" separately for every
possible preceding character. With one, it learns it once.

**It's learned, not set.** It starts as random noise and gets nudged like
everything else. The model discovers English letter frequency on its own,
purely from being wrong about it.
