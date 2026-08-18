# Contributing to this repo

This is your team's repo for all sixteen weeks of COMP 3200. Everything you're graded on —
code, notebooks, plots, write-ups — arrives here through a pull request.

You already know Git. What's new is that you're using it on a **shared** repo, and that
changes the habits. As the Week 1 deck puts it:

> In your previous course, you branched on *your own* private repo. Here, you branch on a
> *shared* one — your code touches the team's. Every habit shifts.

This file is the permanent home for those habits. When a handout doesn't say how something
should be named or who merges what, the answer is here.

---

## Quick reference

| Thing | Convention | Example |
|---|---|---|
| **Week 1 only — no week branch** | branch off `main`, PR back into `main` | `week1/alia-transmission` |
| Week branch, Week 2 on (Repo Lead cuts one) | `weekN-topic`, off `main` | `week3-gradient-descent` |
| Personal branch (everyone else) | `weekN/yourname-topic`, off the **week branch** | `week3/alia-gradient-descent` |
| Commit message | `[WeekN] Short description` | `[Week3] Add gradient descent loop` |
| PR title | `[WeekN] <Title>` | `[Week3] Gradient Descent` |
| Your PR merges into | the **week branch** — never `main` (Week 1 is the exception) | `week3-gradient-descent` |
| Deadline | PRs merged by **Sunday 11:59 PM**, submission posted on Canvas | Week 1's deadline splits — see §1 |

**If it's Week 1, read the first block of §1 and skip the rest of the section.** The diagram
below is the Week 2-onward shape; Week 1 doesn't have a middle tier yet.

```
main
 └── week3-gradient-descent          ← Repo Lead cuts this, once
      ├── week3/alia-gradient-descent    ← you cut this, off the week branch
      ├── week3/marcus-warmer-colder
      └── week3/priya-numpy-version
```

---

## 1. The weekly rhythm

The same six moves every week: branch, code, commit, open a PR, review, merge. Two weeks don't
fit that shape, and one of them is the first week — so those come first.

### The two weeks that don't fit the pattern

**Week 1 has no week branch.** You branch straight off `main`, and your PR goes back into
`main`. Don't cut a week branch; there isn't one. The Repo Lead is being appointed that same
week, so the handout has you merge your own PR once a teammate has approved it. From Week 2 on,
the Repo Lead merges.

There are **four** PRs in Week 1. The first — filling in this repo's `README.md` — is due
before Thursday's class. The other three merge by Sunday: one team PR and two individual
ones.

| PR | Branch (off `main`) | File it adds | From |
|---|---|---|---|
| Padawan Sheet — **one for the whole team** | your team picks the name; `week1/padawan-sheet` is fine | `padawan_sheet.md`, at the repo **root** | Assignment 1, Part A.3 (4 pts) |
| First Transmission — **each person** | `week1/yourname-transmission` | `transmissions/yourname.py` | Assignment 1, Part B (10 pts) |
| Foundations Reflection — **each person** | `week1/yourname-reflection` | `week1/yourname_reflection.md` — start from `week1/reflection_template.md` | Assignment 2 (20 pts) |

Title the team PR `[Week1] Padawan Sheet`. Commit messages follow the usual format —
`[Week1] Add transmission for <your name>`, `[Week1] Add reflection for <your name>`. The
reflection PR still needs a teammate to read it before it merges: they confirm it's complete
and thoughtful, they don't grade it.

**Week 1's deadline splits.** The repository items — repo created, teammates added as
collaborators, branch protection on `main`, `.gitignore` and `README.md` committed — have to be
in place **before Thursday's class**. Everything else merges by Sunday 11:59 PM, like every
other week. Section 6 walks through the GitHub settings side of that.

**Week 7** is the paper midterm. No branch, no code.

### From Week 2 on: the two-tier rhythm

Everything below is the pattern for Weeks 2–16, walked through with Week 3 as the example.

#### Monday — the Repo Lead cuts the week branch

Each week's handout names the branch. The Repo Lead does this once, for the whole team:

```bash
git checkout main
git pull origin main
git checkout -b week3-gradient-descent
git push -u origin week3-gradient-descent
```

Then the Repo Lead tells the team the branch exists, because nobody can start until it does.

#### Tuesday — everyone cuts a personal branch off the week branch

Not off `main`. Off the week branch.

```bash
git fetch origin
git checkout week3-gradient-descent    # get onto the shared week branch
git pull                               # make sure it's current
git checkout -b week3/alia-gradient-descent
```

The `topic` half of the name is what you're actually doing — `gradient-descent`,
`warmer-colder`, `numpy-version`. Not `part2`. A topic tells your teammates what your branch
is for without opening it.

> `git switch` is the newer spelling of `git checkout` for branches, and
> `git switch -c` for `git checkout -b`. Either is fine; this file uses `checkout` throughout.

#### Tuesday–Saturday — code and commit

Work inside that week's folder. Commit as you go, not once at the end — a commit is a save
point you can come back to, and the Git log is part of what's graded.

```bash
git add week3/part2_gradient_descent.py
git commit -m "[Week3] Add gradient descent loop"
git push -u origin week3/alia-gradient-descent
```

If a teammate's work lands in the week branch while you're still working, pull it into your
branch so you find any conflict early rather than in the PR:

```bash
git checkout week3-gradient-descent
git pull
git checkout week3/alia-gradient-descent
git merge week3-gradient-descent
```

#### Saturday — open a PR **into the week branch**

Push, then open the PR on github.com.

**GitHub will default the base branch to `main`. Change it.** This is the single most common
mistake in the first three weeks. The base should be the week branch:

```
base: week3-gradient-descent   ←   compare: week3/alia-gradient-descent
```

Title it `[Week3] <what you built>`. Write a description that says *what* you added and *why*
— see §4.

#### Saturday — review each other

The Review Coordinator makes sure every PR gets at least one review. Reviewing is not a
formality; it is where half the learning in this course happens. See §5 for what to check.

#### Sunday — the Repo Lead merges, then the week branch merges to `main`

The Repo Lead merges each approved personal-branch PR into the week branch. When they're all
in and the Integration Tester has confirmed the week's code actually runs, the Repo Lead opens
one last PR — week branch into `main` — and merges it.

Then everyone gets clean:

```bash
git checkout main
git pull origin main
git branch -d week3/alia-gradient-descent    # delete your local copy
```

That has to be done by **Sunday 11:59 PM** (§9).

### Week branch names

As the handouts give them:

| Week | Branch | Week | Branch |
|---|---|---|---|
| 2 | `week2-forward-prop` | 10 | `week10-watchers` |
| 3 | `week3-gradient-descent` | 11 | `week11-old-tongue` |
| 4 | `week4-many-strikes` | 12 | `week12-visions` |
| 5 | `week5-trial-of-reflection` | 13 | `week13-the-forge` |
| 6 | `week6-outbound` | 14 | `week14-what-to-remember` |
| 8 | `week8-forgetting` | 15 | `week15-the-councils-way` |
| 9 | `week9-forms` | 16 | **Repo Lead picks it** |

Week 16 is the only one you name yourselves — the final project brief says the Repo Lead
"creates the project branch," and stops there. Follow the same shape as the rest
(`week16-yourprojectslug`) and tell the team what you called it.

From Week 9 on the branch name lives in the Repo Lead row of that week's role table rather
than in the setup steps. If you can't find it, that's where to look.

---

## 2. Why two tiers instead of everyone working on `main`

Because five people's code meeting for the first time is where the breakage is, and the week
branch is where that meeting can happen without touching `main`. So `main` always runs — anyone
can clone it and run last week's code — the integration happens Saturday instead of 11:58 PM
Sunday, and your PR diff stays small enough to actually review, because it shows your changes
and not everybody's.

---

## 3. Commit messages

Format: **`[WeekN] Short description`**

```
[Week1] Add transmission for Alia Mehta
[Week3] Add gradient descent loop
[Week5] Fix sign error in backprop delta
```

Present tense, says what the commit does, fits on one line. If you can't describe it in one
line, it's probably two commits.

### Pair programming

Pairing is encouraged. Credit both people with a co-author trailer — blank line after the
message, then:

```bash
git commit -m "[Week4] Add vect_mat_mul helper

Co-authored-by: Marcus Hale <mhale@harding.edu>"
```

The commit then shows both avatars on GitHub and counts for both of you.

### Everyone commits

> Divide the parts however you like, but **every team member must commit code to at least one
> `.py` file**. The Git log should show meaningful contributions from each person.

That's from the Week 2 handout and it holds all semester. One person pushing the whole week
under their name is not a team that split the work well; it's a team where four people didn't
learn the material.

---

## 4. What a good PR looks like

Straight from the Week 1 deck:

| Good PR | Bad PR |
|---|---|
| Title that names the change | Title: *"stuff"* |
| Description that says *what* and *why* | No description |
| Code with useful comments | Uncommented spaghetti |
| Small, focused changes | Fourteen unrelated files |
| Passes a basic "does it run" test | `print("asdf")` left everywhere |

A description that works:

```markdown
## What
Implements `part2_multiple_inputs.py` — w_sum() and the multi-input prediction.

## Why
This is the piece Part 4 needs before it can do multi-in/multi-out.

## Notes
- Ran it from inside `week2/`: `python part2_multiple_inputs.py` gives the
  prediction the handout expects.
- Not sure my comment on the zip() line is clear. Second opinion welcome.

## Questions
Should w_sum live in helpers.py already, or wait until Week 4 asks for it?
```

Two lines and a real question beats a paragraph of throat-clearing. The syllabus asks you to
"describe what you implemented and ask the questions you're sitting on" — the questions are
part of the deliverable, not an admission of weakness.

**Run it before you push it.** Every week. The reviewer's time is not your test suite.

### Declaring AI use

Every PR description has an **AI use** section. Fill it in every time — "None" is a real
answer, not a confession.

The syllabus draws two separate lines here, and it's worth keeping them apart in your head:

- **What you may use it for.** Concept review, study help, and help debugging code *you*
  wrote. Not for writing the from-scratch implementations — building those, and
  understanding every line, is the entire point of this course. Asking an assistant why
  your loop won't converge is fine. Asking it for the loop is not.
- **Saying so.** Whatever you used it for, name it and say what for. One line. Failing to
  acknowledge it is an academic integrity violation, and it's the easier of the two rules
  to break by accident — you asked one quick question on Wednesday and had forgotten about
  it by Saturday.

The habit is worth having on its own terms, separately from the policy. A team that writes
down where the help came from can still tell, in December, which parts of its own codebase
anybody actually understands.

---

## 5. Reviewing: what you actually check

Read every PR with one question in mind:

> If I had to maintain this for a year, would I know what it was for?

Then check, in this order:

1. **Does it run?** Pull the branch and run the file. `git fetch origin && git checkout
   week3/marcus-warmer-colder`, then run it. If it doesn't, that's the review — say so kindly
   and stop there.
2. **Do the comments explain the *why*?** "increments i" is not a comment. "step size shrinks
   as error shrinks, so we converge instead of oscillating" is.
3. **This week's specific rule.** Every handout from Week 3 on gives the Review Coordinator a
   short list of what to check that week — that the `*_deriv` functions take the output not the
   input, that dropout is applied forward *and* backward, that the cell update keeps rather than
   overwrites. Read that row before you review. It is almost always where the bug is.
4. **Is it only what it claims to be?** Fourteen unrelated files means something got committed
   by accident.

> Leave a meaningful comment — not just "looks good." Suggest an improvement, ask a question,
> or compliment something specific.

"Looks good" on a PR you didn't open is worth nothing to the author and costs you the two
points. If the code really is clean, say what's clean about it — that's information.

Approve when you'd be happy to maintain it. Request changes when you wouldn't. Neither is
personal; both are the job.

---

## 6. Week 1 repo setup (Repo Lead)

Three settings, in this order. All of it belongs in place **before Thursday's class**.

### 6a. Add your teammates first

Whoever clicks **New repository** owns it, and nobody else can push to it until they're added.
Do this before anything else, or four people spend Tuesday night locked out.

**Settings → Collaborators → Add people** → their GitHub handle → role **Write**.

That's the role you want: Write can push branches and open PRs, but can't delete the repo or
edit its settings. Handles, not names — put everyone's real handle in the README so people can
find each other's PRs.

### 6b. Make it public

Leave the repo **Public**. There is a concrete reason, and it is not that we want your
half-finished code on the internet.

Branch protection — the setting in 6c that stops anyone merging their own work unreviewed —
is only available on public repositories with a free GitHub account. On a private repo you
would have the *convention* of review with nothing enforcing it, and the first time somebody
is up against the Sunday deadline, the convention is what gives way.

Two things follow from being public, and both are on you:

- **Never commit a credential.** No API keys, no tokens, no passwords — not in code, not in a
  notebook output, not in a commit you plan to amend later. Git remembers. Nothing in this
  course needs one, so if you find yourself typing a secret, stop and ask.
- **Other teams can read your repo. Don't read theirs.** You are all building the same
  network from scratch; the point is the building. Copying another team's solution is the
  academic-integrity line, and a public repo makes it an honour system. Treat it as one.

The upside is real, though: in December you will have a public, sixteen-week commit history
showing you built a deep learning framework from nothing. That is a genuinely good thing to be
able to hand someone.

It also means Dr. Joe can open the link you post on Canvas without being added to anything.

### 6c. Branch protection on `main`

The rule the course requires is exactly one:

> Enable branch protection on `main`: require at least one approving review before merging.
> *No one acts alone in the field; no one merges alone in this course.*

On GitHub: **Settings → Branches → Add branch protection rule**, branch name pattern `main`,
then tick:

- [x] **Require a pull request before merging**
- [x] **Require approvals** — set to **1**
- [x] **Do not allow bypassing the above settings**

That third box is the one people miss. By default, branch protection **doesn't apply to
repository admins** — and the person who created the repo is an admin. So without it, the rule
is real for four of you and optional for the fifth, who is also the one most likely to be
merging at 11:40 PM on a Sunday. Tick it and the rule binds everybody, including whoever set it
up. A rule you can quietly step around isn't a rule; it's a preference.

Nothing else is required — no second reviewer, no status checks, no CI, no CODEOWNERS, no
linear-history rule. If someone suggests adding them, it isn't required here.

One consequence worth knowing: **the week branch is *not* protected.** You can push to it by
accident. See §10.

---

## 7. The five roles

Assigned in Week 1, documented in the README, and **rotated every few weeks** so everyone gets
reps in each seat. On a team of four, one person carries two roles for that stretch.

| Role | What it means |
|---|---|
| **Repo Lead** | Manages the shared repo; enforces branch naming conventions; merges approved PRs; maintains branch protection and the `.gitignore`. |
| **Review Coordinator** | Assigns PR reviewers each week; ensures every PR gets at least one review before merge; tracks review turnaround. |
| **Integration Tester** | Pulls merged code and runs end-to-end tests; flags broken builds or conflicts; maintains a simple test script. |
| **Documentation Lead** | Keeps comments and docstrings up to standard; maintains the team wiki/README; writes weekly summary notes. |
| **Standup Lead** | Runs brief weekly check-ins (async via GitHub Issues or sync); tracks who's working on what; flags blockers early. |

Every weekly handout has a role table telling you what your seat does *that week*
specifically. Read your row before you start.

Write your rotation plan into the README in Week 1 — who holds what, and when it changes. It's
graded, and more usefully, it means nobody has to negotiate it again in Week 6.

---

## 8. How the repo is laid out

**One folder per week, and each week's folder stays self-contained and runnable on its own.**

That's why you'll be told to *copy* a helper forward instead of importing it from somewhere
central — `w_sum` from Week 2 into `week4/helpers.py`, the MNIST loader into `week8/`,
`week9/`, `week10/`. It looks like duplication. It's deliberate: it means later weeks are free
to *change* the carried-forward file, which a shared module could never allow.

So: no `src/`, no top-level `helpers/`, no shared package. Don't refactor one in.

**Tests are plain scripts, run from inside the week folder.**

```bash
cd week3
python test_learning.py
```

Not pytest — this course doesn't use it. The test files you write are scripts you run
directly, and they assume they're being run from inside `weekN/`, because that's how the
sibling imports resolve. Run them from the repo root and they'll fail for reasons that have
nothing to do with your code.

---

## 9. Deadlines and late work

Straight from the syllabus:

> Weekly work is due when your team's pull requests are merged — **Sunday at 11:59 PM**, every
> week — with the submission posted on Canvas.

> **Late work is accepted for two days.** Anything turned in after the deadline but within 48
> hours carries a **20% penalty**. Once that window closes, the work is no longer accepted and
> scores a zero.

Two practical notes.

"Merged" means merged — an open PR at 11:58 PM on Sunday is not submitted work. Which means
your *personal* deadline is really Saturday, because a PR still needs a reviewer to read it and
the Repo Lead to merge it.

And this is a team course. A late PR blocks four other people, not just you. If something
serious is going on, the syllabus asks you to talk to Dr. Joe **before** the deadline rather
than after.

---

## 10. When it goes wrong

It will. These three account for most of it, and none of them is a disaster.

### "We both edited the same file and now Git is yelling about a conflict"

Normal, and expected once your team is moving. Git is telling you two people changed the same
lines and it won't guess which one wins.

```bash
git checkout week3/alia-gradient-descent
git merge week3-gradient-descent
# CONFLICT (content): Merge conflict in week3/README.md
```

Open the file. You'll see both versions marked:

```
<<<<<<< HEAD
your version
=======
the version already on the week branch
>>>>>>> week3-gradient-descent
```

Decide what the file should actually say — often it's both halves, not one — and delete all
three marker lines. Then:

```bash
git add week3/README.md
git commit                 # git writes the merge message for you
git push
```

`git merge --abort` puts everything back if you'd rather start over.

The conflicts you'll actually hit are in `README.md` and the assembled notebook, because those
are the files everyone touches. Notebook conflicts are genuinely nasty — that's why the
Documentation Lead assembles the notebook *once*, at the end, from reviewed `.py` files. Resist
the urge to all edit it at the same time.

### "I branched off `main` by mistake"

Barely a problem. The week branch was cut from `main`, so your commits still apply cleanly —
the PR will show exactly your work. Two fixes:

**If the PR is already open:** click **Edit** next to the PR title on GitHub and change the
base branch from `main` to the week branch. Done.

**If the branch name is also wrong**, rename it and push under the right name:

```bash
git branch -m week3-alia-oops week3/alia-gradient-descent
git push -u origin week3/alia-gradient-descent
git push origin --delete week3-alia-oops
```

You almost never need to rebase in this course. If someone tells you that you do, check
whether retargeting the PR solves it first.

### "I committed straight to the week branch"

Easy to do — the week branch isn't protected, so nothing stopped you.

**If you haven't pushed yet**, move the commit onto a proper branch:

```bash
git status                                        # commit or stash anything uncommitted FIRST
git branch week3/alia-gradient-descent            # bookmark the commit where it is
git reset --hard origin/week3-gradient-descent    # rewind the week branch to the shared state
git checkout week3/alia-gradient-descent
git push -u origin week3/alia-gradient-descent    # now open a PR as normal
```

`git reset --hard` throws away uncommitted changes permanently. Run `git status` first and make
sure it's clean.

**If you already pushed it**, stop — do not force-push a branch your teammates are working on.
Tell the Repo Lead. The work is on the week branch and it's staying there; what you've lost is
the review, so get it back: ask a teammate to open the **Commits** tab, read your commit, and
leave a comment on it. Then mention it in the week's final PR so the Repo Lead knows it's in.

### Anything else

Ask in your team channel before you type a command with `--force` in it. Nothing in this course
is worth force-pushing over, and the person you'd be overwriting is sitting two seats away.

---

## 11. Exit tickets

After every class — Tuesday and Thursday, all semester — you post a short exit ticket as a
**GitHub Issue** in your team repo. It takes about a minute. It's individual, not a team
deliverable, and it isn't graded.

**How.** The **Issues** tab → **New issue** → choose the **Exit Ticket** template. The three
prompts are on the last slide of that day's deck and they change every class, so copy them in
as the headings and answer underneath. Fix the title first: week number, and Tuesday or
Thursday.

If you've never opened a GitHub Issue before, that's the whole thing — an Issue is just a
numbered note attached to the repo. It isn't a bug report and nothing breaks when you file one.

**Why it's worth the minute.** Dr. Joe reads all of them and builds the next class around what
they say. *"This still feels fuzzy"* is the single most useful sentence you can send, and it's
the one people talk themselves out of writing. A class where nobody says anything gets taught
as though everybody followed.

The early decks spell these mechanics out on the slide; the later ones just give you the three
prompts and assume you know where they go. This section is the permanent version.

---

You'll be living in this repo for sixteen weeks. Leave it in a state your team can walk into
cold on a Monday morning — that's the whole discipline, and it's the same one you'll be paid
for later.
