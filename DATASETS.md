# Datasets

Several weeks in COMP 3200 need a real corpus — movie reviews, a stack of Enron email, a
character-level text, the bAbI question set, MNIST digits. None of them live in this repo, and
they never will. This file is where you find out how to get each one.

The reason is size. `reviews.txt` alone is about 34 MB; `ham.txt` is 21 MB. Git keeps every
version of every file you commit, forever, in every clone. Commit a 34 MB text file once and
your teammates download it on every fresh clone for the rest of the semester, even after you
delete it. Downloading it yourself takes thirty seconds.

So: **data is downloaded, not committed.** The `data/` folder is in `.gitignore`, and the
Repo Lead is graded on keeping it that way from Week 11 on.

---

## The rule, in three lines

1. Put every downloaded file in a `data/` folder **inside that week's folder** — `week11/data/`,
   `week14/data/`, and so on.
2. `data/` is already gitignored (a bare `data/` matches at any depth, so all of those are
   covered). Don't fight it.
3. Everyone on the team downloads their own copy. Link this file from your `weekN/README.md`
   so the next person doesn't have to ask.

---

## What you need, and when

| Week | File you end up with | Where it goes | How you get it | Approx. size |
|---|---|---|---|---|
| 8, 9, 10 | `mnist.npz` | `week8/`, `week9/`, `week10/` — beside the code, **not** in `data/` | **Downloaded by your code**, not by you. See below. | 11 MB |
| 11 | `reviews.txt` | `week11/data/` | Grokking-Deep-Learning repo, root | 34 MB |
| 11 | `labels.txt` | `week11/data/` | Grokking-Deep-Learning repo, root | 225 KB |
| 12 | `qa1_train.txt` | `week12/data/` | Grokking-Deep-Learning repo, `tasksv11/en/` — **rename on save**, see below | 94 KB |
| 13 | *(none)* | — | "No external dataset this week — the framework is the deliverable." | — |
| 14 | `qa_oldtongue.txt` | `week14/data/` | Grokking-Deep-Learning repo, root — **rename on save**, see below | 100 KB |
| 15 | `spam.txt` | `week15/data/` | Grokking-Deep-Learning repo, root | 11 MB |
| 15 | `ham.txt` | `week15/data/` | Grokking-Deep-Learning repo, root | 21 MB |
| 16 | whatever your baseline used | `final_project/data/` | Re-use the week you're extending | varies |

Weeks 1–7 need no dataset at all. The numbers you work with in the first half of the course are
typed into the file.

---

## MNIST is different: your code fetches it

Weeks 8, 9 and 10 do **not** ask you to download anything by hand. The `data.py` loader printed
in the Week 8 handout does it for you the first time you run it:

```python
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

def load_mnist(path="mnist.npz"):
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    ...
```

Two things follow from that, and both surprise people:

- **It caches next to your code, not in `data/`.** The default `path="mnist.npz"` is relative to
  wherever you ran Python from — and the course convention is to run scripts from *inside* the
  week folder. So you get `week8/mnist.npz`, and again `week9/mnist.npz`, and again
  `week10/mnist.npz`. Three copies, 11 MB each. That is why `.gitignore` has a `mnist.npz` line. It names that one file rather than
  every `.npz` on purpose, so the weight tensors you save with `np.savez` in Week 12
  are still yours to commit.
  Leave it there.
- **The first run is slow and needs a network.** After that it's instant. If you're working
  somewhere with no connection, copy `mnist.npz` from another week's folder into the new one
  before you run — the loader will see the file and skip the download.

One more thing worth saying out loud: you do **not** need Keras or TensorFlow for any of this,
and nothing in this course imports them. The loader above is the standard library plus NumPy,
and it fetches exactly the same file the big frameworks would fetch for you.

Don't be thrown by the URL, either — `storage.googleapis.com/tensorflow/tf-keras-datasets/…`
has "tensorflow" in the path because that's who *hosts* the file. It's a download address, not
a dependency.

---

## Everything else comes from one public repo

All four text corpora — the movie reviews, the bAbI questions, the Shakespeare text, the Enron
spam/ham — ship together in Andrew Trask's public companion repo, which Weeks 11 and 12 already
cite:

**https://github.com/iamtrask/Grokking-Deep-Learning**

You want individual files, not the whole thing (the full repo is around 76 MB and includes
notebooks you won't use). Two ways to grab one file:

**In a browser.** Open the file on GitHub, click **Raw**, then save the page (`Ctrl+S` /
`Cmd+S`). Watch that your browser doesn't append `.txt` twice or save it as `.htm`.

**From the terminal**, which is less error-prone:

```bash
# Week 11 -- from inside week11/
mkdir -p data
curl -L -o data/reviews.txt https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/reviews.txt
curl -L -o data/labels.txt  https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/labels.txt

# Week 12 -- from inside week12/  (note the rename)
mkdir -p data
curl -L -o data/qa1_train.txt https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/tasksv11/en/qa1_single-supporting-fact_train.txt

# Week 14 -- from inside week14/  (note the rename)
mkdir -p data
curl -L -o data/qa_oldtongue.txt https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/shakespear.txt

# Week 15 -- from inside week15/
mkdir -p data
curl -L -o data/spam.txt https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/spam.txt
curl -L -o data/ham.txt  https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/ham.txt
```

(`curl` ships with Git Bash on Windows and with macOS. If you'd rather use `wget`, `wget -O` takes
the same arguments.)

---

## The two renames, so nobody loses an hour to a `FileNotFoundError`

The handouts name a couple of files by their *source* name in the prose and by their *in-repo*
name in the folder tree. Use the folder-tree name — that's what the code opens.

| Week | Download this | Save it as |
|---|---|---|
| 12 | `qa1_single-supporting-fact_train.txt` | `week12/data/qa1_train.txt` |
| 14 | `shakespear.txt` | `week14/data/qa_oldtongue.txt` |

Week 14's rename is deliberate: in the story, that corpus is a long passage of the Old Tongue
you first met in Week 11. Technically it is Trask's character-level Shakespeare text. Same bytes,
better name.

Week 11 and Week 15 have no rename — `reviews.txt`, `labels.txt`, `spam.txt` and `ham.txt` keep
the names they arrive with.

---

## Sanity-check the download

Text files that go through a browser sometimes arrive as an HTML error page with the right name.
Thirty seconds of checking beats an afternoon of debugging a tokenizer:

```bash
ls -la data/                 # is the size in the right ballpark?
head -c 300 data/reviews.txt # does it look like the data, or like <!DOCTYPE html>?
wc -l data/labels.txt        # 25000 for Week 11
```

Rough line counts the handouts expect: Week 11, 25,000 reviews with 25,000 labels. Week 12, at
least 1,010 lines (you read the first 1,010). Week 15, 9,000 spam and 22,032 ham messages.

---

## If a path doesn't resolve

Every path in this course is relative, and every script is meant to be run **from inside its week
folder** — `cd week11` and then `python sentiment.py`, not `python week11/sentiment.py` from the
repo root. Week 3 states the rule for the test harness ("when the Integration Tester runs
`python test_learning.py` from the `week3/` directory") and it holds everywhere: the flat
one-folder-per-week layout means sibling imports and `data/...` paths only line up when your
working directory is the week folder.

If you hit a `FileNotFoundError` on a corpus, check in this order:

1. Are you in the week folder? `pwd`.
2. Is the file actually there, and is it the size in the table above? `ls -la data/`.
3. Is the filename exactly right, including the two renames above? Windows hides known extensions
   by default, so `qa_oldtongue.txt` on disk may really be `qa_oldtongue.txt.txt`.

And do not "fix" it by hardcoding an absolute path like `C:/Users/you/Downloads/...`. It will pass
on your machine and fail for all four of your teammates and for the grader.

---

## Why none of this is committed, one more time

Because a repo is a record of *what your team wrote*. The corpora are somebody else's, they're
public, they're large, and they never change. The plots, notebooks, tests and code you generate
from them are yours — those all get committed. See `.gitignore` for the full split.
