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

AI, Artificial Intelligence - "a set of rules," in practice; we can't even say "generated from data," that would leave out a lot. Machine learning is I think where you can reliably say "a set of rules generated from data," built based on feedback or some metric. If I were to say that "anything with a reward/loss function (in practice)" is machine learning, I don't think I'd be too far off, though obviously that includes not-artificial things like businesses or people in some respects. Deep learning is, I believe, literally something with depth, something with layers of neurons in most cases, though I wonder how that counts as a coherent category if we know that a one-layer system can model any such. Ah, but you could say such things about Turing machines in general, e.g. "it can be encoded on paper, therefore...", and that's not so helpful.

### 2. A problem traditional programming can't touch

*Original example, and why machine learning suits it.*

Semantic embedding is what springs to mind. I don't see how one could coherently make a semantic embedding without unthinkable amounts of work (though ironically, putting, say, Claude Code on it makes that amount thinkable) unless one were using machine learning. But I might be wrong; I have no idea how Google Translate worked ten years ago, but I know it did work, and I could imagine it working just by grammatical rules. With machine learning it flows naturally.

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- **Supervised or unsupervised?**
Unsupervised, maybe. It could easily be doing it purely based on a set of tags on each song, but if not, it's unsupervised (presumably based on association, what songs both you and other users listen to together?).
- **Parametric or nonparametric?**
Parametric - it might output 2 playlists or 10 depending on the data, but its guts have a fixed number of parameters.

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- **Supervised or unsupervised?**
Supervised - you have a bunch of true/false labeled data, this is as supervised as it gets.
- **Parametric or nonparametric?**
Parametric- I'm not sure what "fixed weights adjusted during training" would sensibly mean, but you're stuck with a given number of weights.

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- **Supervised or unsupervised?**
Supervised. We're giving it data of "all 1s", but still labeled data of "this was bought"; one could argue that it's a selection mechanism instead of labeling, but I don't think that holds water.
- **Parametric or nonparametric?**
I'm not totally sure we can verify this one, but nonparametric; its parameters aren't decided in advance.

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario:**
k-medoids on a bunch of survey responses encoded by semantic embedding, but you're picking k based on testing a bunch of runs, several of each of several values of k, and going for the one with the lowest average silhouette score.
- **Supervised or unsupervised?**
Unsupervised. It's just raw data, no labels as such.
- **Parametric or nonparametric?**
Nonparametric. k-medoids is already nonparametric, but picking that parameter based on another loss function makes it definitively nonparametric.
---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob —**
The actual result of our "action" of prediction, the thing that converts a number into a real temperature; or, effectively in this case, a single hidden weight such that aligning our actual weight with it leads to victory (I just spent hours trying to figure out what subtracting one equation from another really means in linear algebra or even algebra, so I'm just thinking "aligning vector d with hidden vector x by converting everything else into an identity matrix).
- **(b) the prediction —**
Whatever we're about to set the knob to.
- **(c) the error —**
The difference between where we wanted the knob and where we put the knob.
- **(d) the learning step —**
Multiplying that difference by an arbitrary constant and changing our weight by that much.

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` —**
1.7
- **(b) weight set to `0` —** *(and what that means conceptually)*
0, and it's obliterating the information.
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)*
-0.85, and it's taking information and passing its inverse. In practice, I don't know what that looks like; I'd think that it still provides the same quantity of information, but it's about scaling and what gets wiped out, so... I don't know.
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)*
If I'm reading this right, we predict 0.85, find that it's 0.85 away from our target (1.7-0.85), and increase our weight by learning rate * error.

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*
This system is nice and linear, so here we could! We would know the partial derivative of prediction with respect to weight, and just having a learning rate of 1/input would work; that way we'd find we were 0.85 away and update our knob by 0.85/8.5=0.1 to 0.2 and win. Enough tests with varying inputs and varying weights would find that pretty quickly. But you don't always know that it'll be that smooth and linear; it's never this nice, and hunting for convenient linear patterns like this would be a waste of time.

<!-- your answer -->

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

Image data, a series of pixels; classifying within a set of known mushrooms.

### 2. Supervised or unsupervised, and why

Supervised; you give it all labeled pictures of mushrooms that you can track down.

### 3. What "success" would look like

*How would you know it was working?*

A test set should work well enough here.

### 4. What could go wrong

*At least one way this could fail or do harm if deployed carelessly.*

This one's not hard. Poisonous mushrooms. Probably partially avertable by sufficient documentation, linking to lookalikes, and maybe even having it output a probability spread if that's coherent. And having users take multiple pictures from different angles.

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
