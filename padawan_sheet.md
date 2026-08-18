# Padawan Sheet

**Assignment 1, Part A.3 — 4 points. This is a team deliverable.**

Your team shares one padawan. This is who they are.

**Who does the work.** All four fields are a team decision — talk them through together and
agree on the answers. Then **one person commits the finished file and opens the pull
request** on behalf of the team. There is one padawan sheet and one PR for it, not one per
person. (Your individual Week 1 PRs are the transmission and the reflection; those are
separate.)

Fill in the four fields below, in order, replacing each **TODO**. Delete the reference lists
at the bottom once you've chosen — or keep them if you like having them handy. Either is
fine.

---

## 1. Name

**TODO** — the default is **Idrim Sool** (a gender-neutral, melodic name);
change it freely.

> Your padawan's name will appear in your team's assignment write-ups all
> semester. Pick one you'll still like in November.

## 2. Species

**TODO** — choose one from the curated list below.

## 3. Saber color

**TODO** — blue, green, or yellow.

**Why this role:** TODO — one sentence on which role you've chosen.

## 4. Master's lesson

**TODO** — one sentence completing the prompt:

> *"What my master taught me that I remember most…"*

Keep it short. It will be a callback hook later in the season.

---

## Reference: the curated species list

The list is curated so that not every team plays Yoda, and it deliberately
leaves out the species of the Masters who train you — Cerean (Master Velo),
Mirialan (Master Riin), Aleena (Master Korr) — and of Captain Marrek, whose
voice you will hear on comms. Human is the one overlap: it is also Master Vex's
species, so pick it only if you want the contrast.

<!--
  Human is on the list on purpose. It is the one overlap with a Master's
  species (Master Vex) — the handout says so explicitly and invites you to pick
  it "only if you want the contrast." It is not an oversight; please don't
  quietly remove it.
-->

- **Human** — the most common species in the Order; no particular cultural niche.
- **Togruta** — head montrals sense vibration; raised in close-knit communal clans.
- **Zabrak** — horned humanoids of Iridonia; a long tradition of Jedi service.
- **Nautolan** — aquatic, with head-tendrils that read emotional currents; famously calm under pressure.
- **Kel Dor** — masked humanoids (their atmosphere is not yours); a strong cultural tradition of mediation.
- **Cathar** — feline humanoids; known for honor codes and quick reflexes.
- **Pantoran** — blue-skinned humanoids of Pantora; raised in diplomatic culture.

## Reference: saber colors and what they imply

- **Blue — Guardian.** Combat training, physical defense, frontline peacekeeping.
- **Green — Consular.** Force study, diplomacy, healing, deep meditation.
- **Yellow — Sentinel.** Balance of both; investigation, security, undercover work.

---

## Shipping this file

Two things the handout is specific about, plus the house commit format:

- **File name and place:** `padawan_sheet.md`, at the **root** of the repo — where it already
  is. Leave it there.
- **PR title:** `[Week1] Padawan Sheet`.
- **Commit message:** the repo's standard `[WeekN] Short description` form
  (CONTRIBUTING.md §3), so: `[Week1] Add padawan sheet`.

Week 1 has **no week branch** — that two-tier setup starts in Week 2. Whoever ships this cuts
an ordinary personal branch off `main`, the same as any other Week 1 branch:

```bash
git checkout main
git pull origin main
git checkout -b week1/alia-padawan-sheet   # your name, not Alia's
```

The handout doesn't name a branch for this one, so any clear `week1/...` name works. Open the
PR into `main`, get one approving review — branch protection requires it — and merge. All
Week 1 PRs merge by **Sunday 11:59 PM**.

Branch naming, commit messages, what a good PR description looks like, and how to dig yourself
out when something goes sideways: see `CONTRIBUTING.md`.
