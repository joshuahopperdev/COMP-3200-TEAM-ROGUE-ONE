# The NumPy Primer

### COMP 3200 — read this before Week 2

---

> **Captain Boz Marrek, on comms:**
>
> *"Everyone on this channel talks about reaching through the Force. Fine.
> Somebody still has to keep the ship flying.*
>
> *Before you build anything this term, you learn the one tool that makes it
> run. And you learn to tell — before you hit enter — what shape is going to
> come out the other end. Guess at that, and you will spend your semester
> debugging arithmetic that was never wrong."*

---

## Before you start

You need Python 3, NumPy, and about forty minutes. Open a notebook or a REPL and
**run every block as you read it.** Reading this will teach you very little;
running it will teach you most of what Week 2 assumes.

```python
import numpy as np
print(np.__version__)
```

If that errors, fix it now — `pip install numpy matplotlib` — rather than at
7:30 on Tuesday.

This is deliberately not a NumPy tutorial. NumPy is enormous and you will meet
maybe two percent of it this semester. This covers the five things Week 2
actually uses, and one skill: **predicting the shape of a result before you run
it.** That skill is the whole primer. Everything else here is in service of it.

---

## 1. Why NumPy at all

Here is the honest reason, and it is not elegance.

Multiplying two lists together and adding up the results is something you can
already write with a loop. In Week 2 you will write exactly this, from scratch,
before you are allowed to use anything faster — because writing it is how you
come to understand it:

```python
def w_sum(a, b):
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output
```

That is correct, readable, and slow. On a million-element pair, on the machine
this primer was written on:

```
python loop :   54.3 ms
numpy .dot  :    1.1 ms
```

About **fifty times faster**, for the identical arithmetic. Your machine will
give different numbers; the ratio is what matters.

The reason is simple enough to say in one sentence: your loop asks Python to
interpret the same three instructions a million times, while `.dot()` hands the
whole array to compiled code that does the work in one call.

That gap is why this field exists in its current form. A network is a few
arithmetic operations repeated an enormous number of times. Make the repetition
cheap and you can afford a bigger network — which turned out to matter far more
than anyone expected in the 1990s. Most of what you have read about in the last
few years rests on somebody making this inner loop faster.

**So why do we write the slow version at all?** Because `.dot()` will happily
compute the wrong thing very quickly, and you will not notice unless you know
what it was supposed to compute. That is the trade the whole course is built
around: build it slowly by hand, then let the fast version do it.

---

## 2. Shape is the thing to look at

Nearly every NumPy error you hit this semester will be a shape error, and nearly
every *silent wrong answer* will be a shape misunderstanding. So start here.

```python
flat = np.array([8.5, 0.65, 1.2])
col  = np.array([[8.5], [0.65], [1.2]])
row  = np.array([[8.5, 0.65, 1.2]])

print(flat.shape, flat.ndim)
print(col.shape,  col.ndim)
print(row.shape,  row.ndim)
```

```
(3,) 1
(3, 1) 2
(1, 3) 2
```

Read those shapes out loud:

| | shape | say it as |
|---|---|---|
| `flat` | `(3,)` | "three, one-dimensional" |
| `col`  | `(3, 1)` | "three rows, one column" |
| `row`  | `(1, 3)` | "one row, three columns" |

All three hold the same three numbers. **They are not interchangeable.** The
trailing comma in `(3,)` is not a typo — it is Python's way of writing a
one-element tuple, and it is telling you this array has exactly one dimension.
`(3, 1)` has two.

A `(3,)` array is *neither* a row nor a column. It is its own thing, and NumPy
will decide how to treat it based on context. That flexibility is convenient
right up until it isn't.

> **A trap worth meeting now.** `len()` gives you the length of the *first*
> dimension, which is not what you want on a 2-D array:
>
> ```python
> print(len(flat), len(col), len(row))
> ```
> ```
> 3 3 1
> ```
>
> `len(row)` is **1**, because `row` has one row. Use `.shape`. Always `.shape`.

**Habit to build starting now:** when you make an array, print its shape. When a
function returns an array, print its shape. It costs one line and saves the
evening.

---

## 3. The shape contract for `.dot()`

This is the section that matters. Read it twice.

`a.dot(b)` has exactly one rule:

> **The last dimension of `a` must equal the first dimension of `b`.**
> Those two dimensions cancel. What is left, in order, is the shape of the result.

That is the entire contract. Write it on something.

```
(3,)   .dot(  (3, 3)  )  ->  the 3s cancel  ->  (3,)
(2, 3) .dot(  (3, 5)  )  ->  the 3s cancel  ->  (2, 5)
(3,)   .dot(  (3, 5)  )  ->  the 3s cancel  ->  (5,)
(2, 3) .dot(  (3,)    )  ->  the 3s cancel  ->  (2,)
```

Try it against something you already know. This is Week 2's network:

```python
flat  = np.array([8.5, 0.65, 1.2])         # blade_angle, balance, breath
w_mat = np.array([[0.1, 0.1, -0.3],        # -> opens_left?
                  [0.1, 0.2,  0.0],        # -> strikes_high?
                  [0.0, 1.3,  0.1]])       # -> feints?

print(w_mat.dot(flat))
print(flat.dot(w_mat))
```

```
[0.555 0.98  0.965]
[ 0.915  2.54  -2.43 ]
```

**Both ran. Both produced three numbers. Only one is right.**

`[0.555, 0.98, 0.965]` is the answer from the Week 2 lecture. The other one is
what you get when the matrix is applied the wrong way round — NumPy ran down the
columns instead of across the rows, silently, without complaint, because the
shape contract happened to be satisfied either way with a square matrix.

This is the single most important thing in this primer. **A square matrix hides
your mistake.** When the matrix is `(3, 3)`, both orders are legal and you get
no error — just a wrong answer that looks entirely plausible. If you have ever
wondered why we spend so long building things by hand first, this is why: you
need to know what `0.98` should be before you can notice that `2.54` is wrong.

When the shapes *don't* line up, at least NumPy tells you:

```python
np.array([[1.0, 2.0]]).dot(np.array([[1.0, 2.0]]))
```

```
ValueError: shapes (1,2) and (1,2) not aligned: 2 (dim 1) != 1 (dim 0)
```

Read that message closely, because you will see it a lot. It is telling you
exactly which two numbers failed to match: the last dimension of the first array
(`2`), and the first dimension of the second (`1`).

> **Riin's rule, applied to shapes.** In class you will hear "predict the
> prediction before the cell runs." Do the same with shapes. Before you run a
> `.dot()`, say the output shape out loud. If you can't, you don't yet know what
> you're computing — and that is worth finding out now rather than three
> functions later.

---

## 4. `.T`, and when you actually need it

`.T` transposes: it flips rows and columns.

```python
w_mat = np.array([[0.1, 0.1, -0.3],
                  [0.1, 0.2,  0.0],
                  [0.0, 1.3,  0.1]])
print(w_mat)
print(w_mat.T)
```

```
[[ 0.1  0.1 -0.3]
 [ 0.1  0.2  0. ]
 [ 0.   1.3  0.1]]
[[ 0.1  0.1  0. ]
 [ 0.1  0.2  1.3]
 [-0.3  0.   0.1]]
```

Row 1 became column 1. On a non-square matrix the shape changes too: a `(2, 4)`
becomes a `(4, 2)`.

**Here is the part that surprises people:**

```python
flat = np.array([8.5, 0.65, 1.2])
print(flat.shape, flat.T.shape)
```

```
(3,) (3,)
```

`.T` does **nothing at all** to a 1-D array. There are no rows and columns to
swap. If you have been reaching for `.T` to "make a vector into a column," it
has silently done nothing every time — and if your code worked anyway, it worked
for a different reason than you thought.

**So when do you actually need `.T` this semester?** When you have a weight
matrix stored one way and need it the other way to satisfy the shape contract in
§3. Not as a reflex. The decision procedure is always the same:

1. Write down the shape you have.
2. Write down the shape you need.
3. If the last dimension of the first doesn't match the first dimension of the
   second, one of them needs transposing — and now you know which.

That is a thing you can do on paper, before you run anything.

---

## 5. Broadcasting — only the parts you need

Broadcasting is NumPy stretching a smaller array so it lines up with a bigger
one. It is genuinely useful and occasionally treacherous. Three cases matter for
Weeks 2 through 4.

**Case 1 — scalar times array.** This one is safe, and you will use it constantly:

```python
delta = -0.14
flat  = np.array([8.5, 0.65, 1.2])
print(delta * flat)
```

```
[-1.19  -0.091 -0.168]
```

That is the entire weight-update step from Week 4, in one line.

**Case 2 — column times row gives a grid.** This is the outer product:

```python
col_v = np.array([[0.1], [0.2], [-0.1]])   # (3, 1)
row_v = np.array([[8.5, 0.65, 1.2]])       # (1, 3)
print((col_v * row_v).shape)
print(col_v * row_v)
```

```
(3, 3)
[[ 0.85   0.065  0.12 ]
 [ 1.7    0.13   0.24 ]
 [-0.85  -0.065 -0.12 ]]
```

A `(3,1)` and a `(1,3)` produce a `(3,3)`. Every row of the first meets every
column of the second. Week 4 uses exactly this to build a full grid of weight
updates from one vector of inputs and one vector of misses.

**Case 3 — the one that silently does the wrong thing.**

```python
flat = np.array([8.5, 0.65, 1.2])            # (3,)
print((flat * np.array([[0.1], [0.2], [-0.1]])).shape)
```

```
(3, 3)
```

You multiplied three numbers by three numbers and got **nine numbers**. No
error, no warning. NumPy broadcast your `(3,)` against a `(3,1)` and helpfully
produced a grid you did not ask for.

If a later line then sums that grid, you get a number — a perfectly plausible,
completely wrong number. This is the failure mode to watch for: not a crash, but
an answer of the wrong shape that keeps flowing downstream.

**The defence is one line.** When a result surprises you, print its shape before
you print its value.

---

## 6. `np.allclose`, and why you never use `==`

Computers store decimals in binary, and most decimals have no exact binary form
— the same way 1/3 has no exact decimal form. So:

```python
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
print(np.allclose(0.1 + 0.2, 0.3))
```

```
0.30000000000000004
False
True
```

That trailing `...04` is not a bug in your code and not a bug in Python. It is
what binary floating point actually stores. You will see these tails all
semester; they are normal.

**Now the version that will actually bite you.** In Week 2 you build each
operation twice — once from scratch, once in NumPy — and check they agree. Here
is what that check really looks like, using Week 2's own data:

| sensing | from scratch | NumPy | `==` ? |
|---|---|---|---|
| 0 | `0.9650000000000001` | `0.9650000000000001` | ✓ |
| 1 | `1.17` | `1.1700000000000002` | **✗** |
| 2 | `1.09` | `1.09` | ✓ |
| 3 | `0.69` | `0.6900000000000002` | **✗** |

Most values match to the last bit. A couple do not — and they differ in the
seventeenth digit, because NumPy is free to add the terms up in a different
order than your loop did, and floating-point addition is not associative.

**Neither version is wrong.** If you test with `==`, you will spend an hour
hunting a bug that does not exist. So:

```python
np.allclose(scratch_result, numpy_result)     # yes
scratch_result == numpy_result                # no
```

`np.allclose` asks "are these the same to within a sensible tolerance," which is
the question you actually meant. Use it every time you compare two floating-point
results, all semester.

---

## 7. Self-check: ten shapes

Predict each answer **before** running it. Write your guess down — the point is
to find out where your model is wrong, and you cannot do that if you peek.

```python
np.array([1,2,3]).shape                                #  1
np.array([[1,2,3]]).shape                              #  2
np.array([[1],[2],[3]]).shape                          #  3
np.array([1,2,3]).T.shape                              #  4
np.zeros((2,4)).shape                                  #  5
np.zeros((2,4)).T.shape                                #  6
np.array([1,2,3]).dot(np.zeros((3,5))).shape           #  7
np.zeros((2,3)).dot(np.array([1,2,3])).shape           #  8
(np.array([[1],[2]]) * np.array([[10,20,30]])).shape   #  9
np.zeros((2,3)).dot(np.zeros((2,3)))                   # 10
```

<div style="page-break-after: always;"></div>

### Answers

| # | result | why |
|---|---|---|
| 1 | `(3,)` | one dimension, three elements |
| 2 | `(1, 3)` | the outer brackets make it one row of three |
| 3 | `(3, 1)` | three rows of one |
| 4 | `(3,)` | `.T` does nothing to a 1-D array (§4) |
| 5 | `(2, 4)` | two rows, four columns |
| 6 | `(4, 2)` | transpose swaps them |
| 7 | `(5,)` | `(3,)·(3,5)` — the 3s cancel |
| 8 | `(2,)` | `(2,3)·(3,)` — the 3s cancel |
| 9 | `(2, 3)` | broadcasting a column against a row |
| 10 | `ValueError` | `shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)` |

**Got all ten?** You are ready for Week 2.

**Missed 4, 7, or 8?** Re-read §3 and §4 — those three are the ones that cause
real bugs in the assignment, and they are worth ten more minutes now.

**Missed 9?** Re-read §5, especially case 3.

---

## The five things, on one line each

| | what it does | the thing to remember |
|---|---|---|
| `np.array(...)` | makes an array | check `.shape` immediately |
| `.shape` | tells you the dimensions | `(3,)` is not `(3,1)` is not `(1,3)` |
| `.dot(...)` | matrix / vector multiply | last dim of `a` = first dim of `b`; they cancel |
| `.T` | swaps rows and columns | does **nothing** to a 1-D array |
| `np.allclose(...)` | compares with tolerance | use instead of `==`, always |

---

That is the whole toolkit for the next three weeks. Not much, is it?

Which is the point worth leaving you with. The systems you have read about are
not built from exotic machinery — they are built from a weighted sum, repeated
at enormous scale, with somebody paying close attention to shapes. You now know
the operation and you know how to check it. Week 2 is where you start using it
for real.

> **Marrek, signing off:**
>
> *"Print the shape. Then print it again after you change something. I have
> never once regretted knowing what was in the hold before we launched."*
