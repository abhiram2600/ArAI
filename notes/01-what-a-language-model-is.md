# 01 — What a language model actually is

A 30B model is 30 billion numbers. Not rules, not if/else, not logic. Numbers,
arranged in grids, that nobody wrote and nobody can read.

## The whole journey, start to finish

You type **"how to make black tea"**. Here is everything that happens.

**1. Your sentence gets chopped into pieces.**
`how` `to` `make` `black` `tea`

**2. Each piece becomes a number.**
A fixed dictionary maps each piece to an id. `tea` might be #8471. Plain lookup,
no intelligence.

**3. Each number becomes a long list of numbers.**
Id #8471 is looked up in a giant table, which hands back ~4,000 numbers. That
list is what the model actually thinks of as "tea". Still just a lookup.

Your sentence is now 5 lists of numbers. The words are gone.

**4. The lists get mixed together, over and over.**
Each word's list gets updated based on the words around it, so `black` shifts
from meaning the colour to meaning the type of tea. This happens 30 or 90 times
in a row. **This step is the entire subject.** Everything else is plumbing.

**5. The final list becomes scores — one per word in the dictionary.**
`Boil` 7.9, `Black` 8.2, `Banana` -4.3. Higher = more plausible next word.

**6. One word gets picked. Just one.**

## The part that surprises people

That's the end. The model produced exactly one word.

To get the next word the whole thing runs again from scratch, now with
`"how to make black tea Boil"` as the input. Then again. Then again.

**There is no plan.** It never decided to write you a recipe. It has no idea what
its second sentence will be. It answers one word at a time, and a recipe falls
out because that's what plausibly follows a recipe question.

Everything a language model has ever done is that loop, run a few hundred times.

## Two modes, one path

|  | what happens |
|---|---|
| **Using it** | Run steps 1→6. Every number is frozen. Out comes a word. |
| **Training it** | Same steps 1→6 on text where the answer is known, plus: check the answer, adjust every number. Repeat. |

Training isn't a different mechanism. It is the same path, plus one extra move
at the end.

> Training is where all the numbers are decided.
> Using it is where all the numbers sit still and text flows through them.
