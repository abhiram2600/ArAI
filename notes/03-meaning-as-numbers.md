# 03 — Meaning as numbers

## It's a spreadsheet

```
             <-------- 384 numbers per row -------->
row   0  \n  | 0.13  -0.88   0.42   ...   0.07
row   1  ' ' |-0.55   0.19  -0.31   ...   0.93
...
row  49  'k' | 0.71   0.02  -0.66   ...  -0.14
...
row  64  'z' | 0.38   0.45   0.11   ...   0.29
```

One row per dictionary entry. "Turning id 49 into a list of numbers" means
**go to row 49 and read it off.** That is the entire operation.

**65 rows** — forced, it's the dictionary size.
**384 columns** — chosen. Could be 64, could be 4000. Bigger = more detail per
character, more numbers to train. Nothing to do with how big the text file is.

## Why a list and not one number

One number can only place a word somewhere on a single line — more of something,
less of something. But `king` is royal *and* male *and* human *and* archaic *and*
singular, all at once and independently. You can't encode five independent things
in one number. Each slot is one dial.

## The numbers have no names

You might imagine 384 slots labelled *royal*, *male*, *powerful*. There are no
labels. Nobody knows what slot #3 is. It isn't "powerful". It's slot #3.

A single number almost never means one clean thing — **combinations do.**

Think of colour. Three numbers: R, G, B. Which one means *sunset*? None. Sunset
is a particular mix. And the red channel participates in millions of colours, so
asking what it "means" alone has no good answer.

Better still, think of **map coordinates**:

```
Paris  = (48.8, 2.3)
Vienna = (48.2, 16.4)
```

What does `48.8` mean on its own? Nothing. It isn't "half of Paris". It's a
latitude, and it shows up in plenty of unrelated places. Meaning lives in the
whole combination, not in any single number.

So `powerful` and `powerless` are not built from a shared "power" slot plus a
modifier slot. They're two nearby points in the same region of the map, pointing
in different directions.

## Why it's unreadable

The model has 384 slots (or 4,000 in a real one) but needs to represent far more
concepts than that — millions. So concepts have no choice but to share slots and
overlap. Every slot is doing dozens of jobs at once.

**It's not encryption. It's overcrowding.**

## Where the numbers come from

They start as **random noise**. Every row is garbage. Then:

1. Show the model real text with the last character hidden: *"the king sat on the ___"*
2. It guesses. Early on, garbage.
3. The true answer is `throne`.
4. Every number that contributed to the wrong guess gets nudged slightly —
   including `king`'s row.

Later it sees *"the queen sat on the ___"*. Same correct answer, so `queen`'s row
gets nudged **the same direction** `king`'s was.

Across billions of sentences, words that keep appearing in the same situations
keep receiving the same corrections, so they drift into the same neighbourhood.

**`king` and `queen` end up near each other because they were repeatedly corrected
in the same way** — not because anyone told the model they're related. The map
draws itself out of being wrong over and over.
