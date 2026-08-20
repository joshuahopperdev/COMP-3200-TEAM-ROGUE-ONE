# TODO: Your Team Name

> COMP 3200 — Deep Learning · Fall 2026 · Harding University · Dr. Joe Faith
> *The Cybernetic Heresy*

---

## Week 1: stand this repo up

**Do this once, as a team, before you do anything else.** Five people each
making their own copy produces five disconnected repos, and then team pull
requests are impossible. There is exactly one team repo.

1. **One person — usually the Repo Lead — creates it.** On this template repo,
   click **Use this template → Create a new repository**. Name it something
   meaningful: `temple-crew-alpha`, `team-starcrossing`, whatever your crew
   answers to. Leave it **Public** — branch protection (step 4) only works on
   public repos with a free account, and step 4 is graded. See CONTRIBUTING §6b
   for what being public means for you.
2. **Add the other four as collaborators with Write access**
   (*Settings → Collaborators*).
3. **Everyone else clones that repo.** `git clone <url>`. **Nobody else forks,
   and nobody else uses the template.** If you have your own copy, you are
   working alone by accident.
4. **Turn on branch protection for `main`:** require at least one approving
   review before merging. No one merges alone in this course.

**Week 1 branches straight off `main`** — `week1/yourname-transmission`,
`week1/yourname-reflection`. The two-tier week-branch pattern described further
down starts in Week 2.

**Week 1 has a split deadline.**

| Due | What |
|---|---|
| **Before Thursday's class — Thu, Aug 20** | Repo created, branch protection on `main`, roles assigned, `.gitignore` committed, README filled in |
| **Sun, Aug 23, 11:59 PM** | `padawan_sheet.md` merged (one team PR), every member's `transmissions/yourname.py` merged, every member's `week1/yourname_reflection.md` merged, repo link posted on Canvas |

Late work is accepted up to 48 hours past the deadline for a flat 20% penalty.
After 48 hours it is not accepted and scores a zero.

---

## Where to look next

- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — branch names, commit and PR titles,
  branch protection, and what a reviewable PR looks like. Read this before your
  first branch.
- **[`DATASETS.md`](DATASETS.md)** — every dataset the course uses, where to
  download it, and what to name it.
- **[`numpy_primer.md`](numpy_primer.md)** — about 40 minutes. Run it, don't
  read it. Everything from Week 2 on assumes it.
- **[`week1/reflection_template.md`](week1/reflection_template.md)** — the
  skeleton for Assignment 2. Copy it, answer in your copy.

---

## Canvas is authoritative

This repo is a *copy*. Your team made it from the course template in Week 1, and
it stops there — it does not sync. If Dr. Joe corrects one of these files in
October, your copy still has the old version, and there's no button that fixes
that.

So when this repo and Canvas disagree, **Canvas wins.** Assignment handouts, due
dates, schedule changes and any corrections all live there. What lives here is
the part that doesn't change: the conventions, where the datasets come from, and
your own work.

---

Following in the path of Master Robbie Rotten, we honor the number one as we 
are now a rogue actor without our master.
---

## The Crew


| Name | GitHub handle | Role, 8/18- (current) |
|---|---|---|
| Caleb Hopper | @joshuahopperdev | Repo Lead, Standup Lead |
| Josiah Duvalian | @jduvalian | Review Coordinator |
| Nathanael Williams | @Williams-Nathanael | Documentation Lead |
| Oliver Pham | @Odog0 | Integration Tester |

**Communication channel:** Discord chat

---

## Roles

Every team assigns these five in Week 1. Teams of four carry one person in two
seats for that stretch. Descriptions are from the syllabus.

| Role | What this seat does |
|---|---|
| **Repo Lead** | Manages the shared repo; enforces branch naming conventions; merges approved PRs; maintains branch protection and the `.gitignore`. |
| **Review Coordinator** | Assigns PR reviewers each week; ensures every PR gets at least one review before merge; tracks review turnaround. |
| **Integration Tester** | Pulls merged code and runs end-to-end tests; flags broken builds or conflicts; maintains a simple test script. |
| **Documentation Lead** | Keeps comments and docstrings up to standard; maintains the team wiki/README; writes weekly summary notes. |
| **Standup Lead** | Runs brief weekly check-ins (async via GitHub Issues or sync); tracks who's working on what; flags blockers early. |

### Rotation plan

We will rotate every 1-4 weeks on an ad hoc basis, rotating earlier if the way one role is handled is causing problems and later if it's going well. Most of the time we will rotate every 2-3 weeks. Roles will be assigned on an ad hoc basis as well, generally trying to assign people to roles they have not had before.
---

## How this repo works

**The weekly rhythm.** Same loop every week, from Week 2 to the final project:

1. **Branch.** The Repo Lead cuts one shared week branch off `main`
   (e.g. `week3-gradient-descent`). You cut your own branch off *that* for your
   slice of the week (e.g. `week3/alia-gradient-descent`).
2. **Code & commit.** Clean, well-commented code; commit often, with messages
   that explain *why*.
3. **Open a PR** back into the week branch. Describe what you implemented and
   ask the questions you're sitting on.
4. **Review.** Someone reads it and leaves real feedback — not "looks good."
5. **Merge & test.** The Repo Lead merges your branch into the week branch,
   then the week branch into `main`; the Integration Tester pulls and verifies
   it all still runs.
6. **Document.** The Documentation Lead captures the week's learnings here.

**After every class**, post an exit ticket as a GitHub Issue — *Issues → New
issue → Exit Ticket*. A minute, individual, not graded, and it's what next
class gets built around. See `CONTRIBUTING.md` §11.

**Pull requests must be merged by Sunday, 11:59 PM**, and the submission goes up
on Canvas. (Week 1 is the one exception — see the split deadline above.)

**Where things live.**

| Path | What it is |
|---|---|
| `padawan_sheet.md` | Your team's padawan. Root level. Your first team PR. |
| `transmissions/` | One introduction file per member (Week 1, Part B). |
| `week1/` | One Foundations Reflection per member — `week1/yourname_reflection.md`, Assignment 2, 20 points, individual. Start from `week1/reflection_template.md`. |
| `weekN/` | You create these — one folder per week, flat, self-contained and runnable on its own. Tests run as `python test_something.py` from *inside* the week folder. |
| `data/` | Downloaded corpora. Git-ignored on purpose; see `DATASETS.md`. |

**Setup.** Python 3 and Jupyter, then:

```bash
pip install numpy matplotlib
python -c "import numpy as np; print(np.__version__)"
```

That's the whole toolchain until Week 15 adds one package. We don't import what
we haven't built.

<details>
<summary><b>If Python or Jupyter isn't running yet — open this</b></summary>

<br>

**Getting them installed.** Either route is fine:

- **[python.org](https://www.python.org/downloads/)** — 3.10 or newer. On Windows, tick
  **"Add python.exe to PATH"** on the first installer screen. It is easy to miss and it
  is the cause of most of what goes wrong below. Then `pip install jupyter`.
- **[Anaconda](https://www.anaconda.com/download)** — bigger download, but it bundles
  Python, Jupyter, NumPy and matplotlib together and puts them on PATH for you.

Start a notebook with `jupyter notebook` from your terminal.

**The three things that actually go wrong:**

| Symptom | What's happening | Fix |
|---|---|---|
| `python: command not found`, or Windows opens the Microsoft Store | The command isn't on PATH. Windows ships a placeholder that just opens the Store. | Try `python3` (macOS/Linux) or `py` (Windows). If neither works, reinstall and tick the PATH box. |
| `pip: command not found` | `pip` isn't on PATH even though Python is. | Use `python -m pip install numpy matplotlib` — it always runs the pip belonging to that Python. |
| `pip install` succeeded, but `import numpy` still fails | You have two Pythons, and pip installed into the other one. | `python -c "import sys; print(sys.executable)"` to see which one you're running, then install with `python -m pip` so they match. |

**Still stuck?** Ask in your team channel first — odds are a teammate hit the same thing —
then message Dr. Joe. Do not spend Week 1 fighting a PATH variable; it teaches you nothing
and there's a whole course behind it.

</details>

---

*Nothing here assumes you showed up already knowing. May the Force be with you.*
