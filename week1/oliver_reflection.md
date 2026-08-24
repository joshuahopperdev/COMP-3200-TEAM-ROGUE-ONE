# Foundations Reflection — template

**Assignment 2 · 20 points · individual.** Merged by **Sunday, Aug 23, 11:59 PM.**

> **Copy this file — don't write in it.** Save your answers as
> `week1/yourname_reflection.md` (your real first name, lowercase — e.g.
> `week1/alia_reflection.md`). Leave this template where it is so the next person
> can find it.
>
> The full prompts live in the Assignment 2 handout on Canvas. What's below is the
> skeleton: every part and question, in order, with the point values. It is not a
> substitute for the handout — keep that open, especially for Part 2, where the
> exact wording of each scenario is what carries the answer.
>
> The four parts are graded separately, and a part you forgot to answer is a zero
> for that part — far and away the most common way people lose points on this one.

*In your copy, delete the block above and the italic hints as you go.*

---

## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

*Not just "subset" — what makes each layer different from the one above it?*

AI basically is just artificial intelligence. It's when a human makes something that can make its own decisions.
Machine learning basically takes a bunch of data and finds its own rule to the problem.
Deep Learning uses neural networks for machine learning but far more complex.

### 2. A problem traditional programming can't touch

*Original example, and why machine learning suits it.*

A chatbot would require a lot of direct, human-written answers and have a reply-bank 
or something that an algorithm would pull from. But ML can read from what's online and just write its own.

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- **Supervised or unsupervised?**
- **Parametric or nonparametric?**

Parametric - fixed params

Unsupervised - finds its own labels (playlists)

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- **Supervised or unsupervised?**
- **Parametric or nonparametric?**

Parametric - fixed params/weights 

Supervised - pre-set labels

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- **Supervised or unsupervised?**
- **Parametric or nonparametric?**

Nonparametric - no params ahead of time

Supervised - 10 most similar customers are labels

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario: New Rocket league AI groups players by playstyle and discovers the 'misses open nets' category**
- **Supervised or unsupervised?**
- **Parametric or nonparametric?**

Unsupervised - found the category

Nonparametric - doesn't have fixed set of params but learns from player data


---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob — temp setting you change**
- **(b) the prediction — what temp the stat is currently making**
- **(c) the error — diff between what you want and what you are getting**
- **(d) the learning step — how much you turn the knob**

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` — 1.7**
- **(b) weight set to `0` — 0 no influence on pred** *(and what that means conceptually)*
- **(c) weight negative, `-0.1` — -0.85 opposite influence on pred** *(and what a negative weight could represent)*
- **(d) if the answer should have been `1.7` — pred too low, increase weight** *(too high or too low, and how you know)*

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

Neural networks have lots of weights and there are too many possible combinations to just
predict the perfect one since they all change the overall prediction in so many different ways

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

Predict whether a Rocket League player has a solid chance of hitting a freestyle clip
in a given match based on gameplay data like mechanics used, shot counts, ariels, air dribbles,
and previous attempts.

### 2. Supervised or unsupervised, and why

Supervised - model would train on past attempts where we can give whether the player
successfully hits the clip or not.

### 3. What "success" would look like

*How would you know it was working?*

The model would successfully predict right when a player is likely to hit a clip
and become more accurate than just guessing if the player will hit the clip.

### 4. What could go wrong

*At least one way this could fail or do harm if deployed carelessly.*

Clips don't actually make someone win and shouldn't be the only thing done in Rocket League.
So the model might encourage players to focus too much on hitting clips instead of actually playing
the game competitively, or it could just predict poorly if it doesn't have enough gameplay data.

---

## Before you open the PR

- [ ] All four parts answered — check against the headings above, not your memory
- [ ] Part 2: both axes **and** a justification, for all four scenarios
- [ ] Part 2 #4 uses a combination you didn't already use
- [ ] Part 4 #4 is answered — it's the one people skip
- [ ] Your own words throughout (see below)
- [ ] The file is `week1/yourname_reflection.md` — not this template

Week 1 branches straight off `main`; there's no week branch until Week 2
(`CONTRIBUTING.md` §1).

```bash
git checkout main
git pull origin main
git checkout -b week1/alia-reflection        # your name, not Alia's
git add week1/alia_reflection.md
git commit -m "[Week1] Add reflection for Alia Mehta"
git push -u origin week1/alia-reflection
```

Open the PR into `main`, ask a teammate to review it — they're confirming it's
complete and thoughtful, they are not grading it — and merge before Sunday 11:59 PM.

---

## On AI tools

You're welcome to use them to check your understanding. Every answer has to be in your
own words and reflect your own thinking. If I can paste your answer into a search box
and find it verbatim, that's a problem.

**And say so when you use them.** A line in your PR description naming what you used
and what you used it for is enough — the pull request template asks for it directly.
Acknowledged use is ordinary professional practice. Unacknowledged use is an academic
integrity issue.

Don't let an assistant write your reflection. Write it. *Then* ask it where you've been
unclear — and rewrite.
