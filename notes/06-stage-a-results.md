# 06 — Stage A: what we actually got

Stage A is **idea 1 only** — the embedding table — plus all the training
machinery. Deliberately no attention. Each character is predicted from the
single character before it and nothing else.

```
'q' -> 50 -> [65 x 384] -> 384 numbers -> [384 x 65] -> 65 scores
```

49,985 numbers. All random at birth.

## Before training

```
its top 5 guesses for what follows 'q':
    'd'    score +1.43
    'F'    score +1.33
    'l'    score +1.28
    'g'    score +1.17
    'p'    score +1.11

    score it gives 'u' : -0.09
    spread of all 65   : -1.10 to +1.43
```

It thinks `qd` is the most likely pair in English, and `u` ranks about 30th of 65.
The spread is the real tell — everything crammed into a band of 2.5 is what
"no opinion" looks like numerically.

And `d` won purely by luck. Change the random seed and a different letter wins:

```
seed 1337   'd'  'F'  'l'     (score for 'u': -0.09)
seed 1      'K'  'v'  ';'     (score for 'u': -0.18)
seed 42     'F'  'r'  'T'     (score for 'u': -0.17)
seed 7      'T'  'y'  'E'     (score for 'u': -0.50)
```

None of them ever guessed `u`. Guessing it by luck is a 1-in-65 shot.

Measured loss: **4.338**, against `ln(65) = 4.174` for pure guessing. Correct.

## After training

3,000 steps, 85 seconds on an M1 Pro.

```
  step  train loss   held-out   guesses after 'q'
     0      4.3371     4.3451   'd' 5%  'F' 4%  'l' 4%
   300      2.4718     2.5086   'u' 100%  'w' 0%  'N' 0%
  1200      2.4617     2.4945   'u' 100%  'w' 0%  'y' 0%
  3000      2.4627     2.4953   'u' 100%  'o' 0%  'w' 0%
```

**It was done by step 300.** Nine-tenths of the run accomplished nothing — not
because training broke, but because there was nothing left to learn. With one
character of context there are only 65 x 65 = 4,225 facts in this model's entire
universe, and it had absorbed them in under a minute.

**Train 2.463, held-out 2.495.** Nearly identical, so no memorisation. 50,000
numbers against a million characters is nowhere near enough capacity to overfit.

## The bias check

We predicted the model would work out English letter frequency on its own.

```
actually most common : ' ' 15.2%   'e' 8.5%   't' 6.0%   'o' 5.9%   'a' 5.0%
model's highest bias : 'o'   ' '   'a'   't'   's'   'i'
model's lowest  bias : '.'   'x'   ';'   'X'   '-'   'Z'
```

Good match at the common end. But `'.'` got the **lowest** bias despite being
rank 26 of 65 — not rare at all. Why: the bias is not a frequency table, it's
**the leftover after the input-dependent part has done its job.** A period is
extremely predictable from context, so table 2 handles it and the bias is free
to sit low.

## What 2.46 sounds like

```
Bullisthy d.
Th: sovef ge aleta yo'JUKEShe ificulas aver ot amerend ERDINut h lis od ine
DWisisleamy at thagstsworeamerendond. he al'ther at d PUST:

Waitof mes l, oked s ghichellath anceshe gan st he.
I poor y,
Y:
Masises nd htofed
BOfo s:
```

Garbage — but look at which kind.

**What it learned:**
- Words have plausible **length**. It stops for spaces at about the right rate.
- **Vowel/consonant alternation** — `sovef`, `aleta`, `bonounge`. It never
  writes `ktrsp`.
- **The script format** — `NAME:` then a newline. It picked up that colons
  follow capitals and newlines follow colons.
- Real English falls out by accident: `Waitof`, `at the`, `I poor y`, `may-`.
- `quceane` — there's the `qu`.

**What it can't do: finish a word.** It writes `amerend` twice and
`thagstsworeamerendond` once, because at every character it is asking *"what
usually follows `d`?"* with no memory that it started writing a word nine
characters ago.

## The wall

**2.46 is the ceiling.** This model has no memory. Ask it what follows `f` and
it will forever answer "well, in general..." because it cannot see that the
previous three characters were `w`, `i`, `f`.

It learned English **texture** and cannot learn English **words**.

That is exactly the problem attention solves, and Stage B is where we solve it.
