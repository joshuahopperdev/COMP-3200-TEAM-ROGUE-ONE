# Foundations Reflection — template

**Assignment 2 · 20 points · individual.** Merged by **Sunday, Aug 23, 11:59 PM.**

---

## Part 1 — The Landscape (4 pts)

### 1. AI vs. machine learning vs. deep learning

AI is anything that tries to mimic human intelligence.
Machine learning uses various algorithms to discern rules and structure in data.
Deep learning takes machine learning to another level with trying to emulate the human 
brain by using structures like neural networks.

### 2. A problem traditional programming can't touch

Probably something of a copout but the mnist dataset with numbers. I can't comprehend how 
an image classification set would work without a machine learning approach.

---

## Part 2 — Classifying Algorithms (6 pts)

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- **Supervised**
- **Parametric**

I would have to guess that an auto-generated playlist at least have genre labels.
Parametric because of the hint about a fixed set of internal parameters. 
EDIT-next day. Looking back at this, I realize that #2 has label in the example.
I very well could be overthinking this and giving additional details that weren't provided.
But I thought I'd at least explain why I was going with supervised since I foresee there
being labels on this kind of data.

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- **Supervised**
- **Parametric**

The examples being labeled tells that it should be supervised.
Like the example before, the fixed number of weights to be adjusted should be parametric.

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- **Unsupervised**
- **Nonparametric**

Previous Guess - This one has me more stumped, but I think I'm leaning on supervised again just because
how would the customers be similar if there wasn't labeled data to compare. I could also
see it being unsupervised and just searching for connections between values. I think I'm 
going with supervised.
I ended up changing my tune and realized this looks like kmeans, which is unsupervised. 
I'm going with nonparametric since the answers are already there and it's just learning
what's right/wrong. Nothing is really being tuned.

### 4. Future item predictive ordering.

- **The scenario: Training on historical item data for a store to predict what should be ordered in advance.**
- **Supervised?**
- **Nonparametric**

This data would need to be labeled so it's supervised.
This is data that existed and was historical so it's trying to memorize what it looks like. It's not trying to 
learn patterns or have knobs for adjusting.

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

I'm guessing this is us describing what these concepts correlate to?

- **(a) the knob — The dial for setting the heat to hit the target temp.**
- **(b) the prediction — Predicting we hit a nice 70F temp.**
- **(c) the error — The temp being 78F when we wake up, being 8 degrees over what was set.**
- **(d) the learning step — Us adjusting the dial after we wake up sweating.**

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` — The prediction becomes 1.7 since input is the same.**
- **(b) weight set to `0` — Prediction is set to 0, regardless of input. This is a dead neuron.** 
(Isn't there something that can check for 0 weights and set it to a very small decimal so we don't kill neurons?)
- **(c) weight negative, `-0.1` — This would just be -0.85. Though wouldn't we do abs() for most things?** 
I feel like a negative weight is the result of some error in our gradient descent. I guess it could be wanting to
reverse something off an input. Like if an input is high, you want to reverse and push the result lower. But this
could also just be done outside of the weight/input area.
- **(d) if the answer should have been `1.7` — If our answer should have been 1.7, then our weight is too low.**
If I understand correctly, our error should then come back and we would adjust our weight to be higher. Maybe it's 
not high enough the first time and we get 1.3 and need to increase it further. Maybe we overshoot and get 2.0 and
need to decrease the weight.

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

If I had to guess it's because if we have like... 10 knobs and they all have some unknown influence on an output,
those values aren't just known to us outright. Finding a good starting point while also trial and error testing 
what value tweaks and what combination of them do what is part of the process and takes time.

In my mind its just because we are trying to tune a system to an unknown, real life, system that we just don't 
fully understand the fine details of.

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

Optimal loading storage for item restocks to be shipped to stores.
Height of shelf.
Size of shelf.
Item size, weight, quantity in package(up to dozens of parameters here.)
Frequency of item pull.
Distance to restock.
Distance down shelving.
Location in relevance to other items they can't be next to.
I can only list so many inputs here but it's a *lot*.

### 2. Supervised or unsupervised, and why

I feel like this is definitely supervised since all of the inputs above are labeled.
We have a very good idea what our parameters and labels are.

### 3. What "success" would look like

*How would you know it was working?*

Now THIS is the hard part and in complete honesty, I don't completely know. There would need
to be some kind of simulation for what an 'optimal' layout would be given a set of items and shelving.
This is normally something done by a series of reports and engineering intuition and guessing good layouts.
When a layout is found that is more efficient after testing a few runs, it is generally kept. In my mind, 
we would need to train across several premade layouts that are considered good, and maybe let it run to determine
what makes them 'good'. It might actually end up being two separate models, one for figuring out what constitutes
a good layout, and then another on how to achieve it given an input of items, shelving, etc.

### 4. What could go wrong

*At least one way this could fail or do harm if deployed carelessly.*

Obviously, we could simply get it wrong as to what an optimal solution looks like and then we are just handing down
misinformation for what optimal shelving looks like. The nice thing is that we have enough previous layouts to
look at that we can at least verify we are moving in the right path. But I've wrestled with this idea for a while now.

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
