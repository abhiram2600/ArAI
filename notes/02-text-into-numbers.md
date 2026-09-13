# 02 — Text into numbers

The model cannot see text. It only ever sees integers.

## Where the dictionary comes from

Nobody writes it by hand, and it is **not** "all the words in English". It is
built by a small algorithm before training, from the training text itself.

Two obvious options, both bad:

- **One entry per letter.** 65 entries, beautifully small — but a page of text
  becomes thousands of pieces, and the model has to work to notice `t-h-e` is
  a unit at all.
- **One entry per word.** `tea` is 1 slot — but you need hundreds of thousands
  of entries, and `teaa` or emoji or German has no entry at all. Dead end.

Real answer is in between, built by **gluing**:

1. Start with every individual character.
2. Find the most frequently adjacent pair in the training text.
3. Glue it into one new entry.
4. Repeat 50,000 times.

On text full of *"the theatre, the theme, the thesis"*:

```
most common pair is  t + h      ->  new entry: "th"
now most common is   th + e     ->  new entry: "the"
now most common is   the + " "  ->  new entry: "the "
```

Common words end up as one entry, rare words as several. Nobody decided that —
frequency did. Then it is **frozen** forever.

## Why we use characters instead

For ArAI the dictionary is just the 65 unique characters in the file.

- **Our data is tiny** (1.1MB). A 50,000-piece dictionary built from it would
  have most entries appearing a handful of times — never enough to learn.
- **Zero moving parts.** `sorted(set(text))` and we're done.
- **Small tables** at both ends of the model, so more of our numbers go into
  the part that thinks.

What it costs: our 256-character window is only ~50 words of context, and the
model has to waste effort learning *spelling* instead of meaning. At trillion-
character scale that waste is unforgivable, which is why real models don't do it.

## The id is an address, not a quantity

`k` = 49. But **49 does not mean `k`.** It's a label, like a jersey number.
Shuffle the assignment so `k` = 12 and nothing breaks. There is nothing
`k`-ish about 49.

Which means the model must never do arithmetic on it. 49 is not "more" than 48.
Averaging `a`(39) and `c`(41) does not give you `b`.

So the very first thing the model does is throw the id away:

```
'k'  ->  49  ->  [0.71, -0.22, 0.09, ... 384 numbers]
        address              content
```

The id's only job is to be a row number. That is exactly why the next step
has to exist.

## Our actual numbers

```
characters in file  : 1,115,394
vocab size          : 65
the dictionary      : "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
train characters    : 1,003,854
held-out characters :   111,540   <- never trained on
```

The only digit anywhere in Shakespeare's text here is `3`. Our model has no
concept of `4` at all — it cannot read it or write it. And it does not know
`3` is a number; to the model it's a squiggle that appears near other squiggles.
That is a large part of why language models are shaky at arithmetic.
