# Foundations Reflection — Josiah Duvalian

---

## Part 1 — The Landscape (4 pts)

### 1. AI vs. machine learning vs. deep learning

Artificial intelligence describes programs that look like they're learning or are otherwise intelligent when they're actually just using clever algorithms to solve clever problems. Machine learning is the part of AI that specializes in finding patterns and acting on those patterns in the form of classification, regression, or true-false/yes-no choices. Deep learning takes supervised machine learning in particular and repeats it in multiple layers to come to more accurate results.

### 2. A problem traditional programming can't touch

The problem of antiswear programs and chat censoring can be achieved fairly well with heavy regularization/sanitization of messages and detection algorithms, but it often misses the nuance of English toxicity. Similar to ham or spam email classification, machine learning can find the nuanced toxic messages that the algorithm can't touch.

---

## Part 2 — Classifying Algorithms (6 pts)

### 1. Music app — auto-generated playlists

- **Supervised or unsupervised?** Unsupervised because people's tastes (the label) are almost as unique as the person; it is impractical to have a fixed set of predetermined playlists or even types of playlists.
- **Parametric or nonparametric?** Parametric because there's a *fixed* of parameters.

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

- **Supervised or unsupervised?** Supervised because there's a clear label--you have it or you don't have it.
- **Parametric or nonparametric?** Parametric because even though there's millions of parameters I don't think the number of parameters change.

### 3. Retail site — recommendations from the 10 most similar past customers

- **Supervised or unsupervised?** Supervised because the website is recommending products, which can represent labels. However, a combination of products may justify an unsupervised answer since there's no fixed set of combinations to choose from.
- **Parametric or nonparametric?** Unparametric, especially at the beginning, because past customers are the parameters and they differ by number based on a time window one may have or a decision to change how many customers to base a recommendation on.

### 4. Your own scenario

- **The scenario:** A school curates a given student's cirriculum to use the best learning method from a fixed set of known and developed methods based on said student's performance (most notably but not exclusively, score) on a fixed number of tests and other diagnostic examinations.
- **Supervised or unsupervised?** Supervised because the developed learning methods are the labels.
- **Parametric or nonparametric?** Parameteric because the exams are the parameters and there are a fixed number of them.

---

## Part 3 — The Knobs Mental Model (6 pts)

### 1. The thermostat analogy

- **(a) the knob —** the lever on the thermostat with which you can adjust the temperature (I chose lever because then you don't know what exact temperature it's supposed to be and thus lose the risk of a technically broken thermostat; this is what we had in our last dorm and I believe it's the case with a few buildings on campus).
- **(b) the prediction —** the room will be a satisfactory temperature in our opinion when the lever is set at a certain position on the thermostat.
- **(c) the error —** our bodies are either too hot or too cold.
- **(d) the learning step —** adjust the lever slightly to find a good room temperature for our bodies.

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` —** prediction doubles to `1.7`; the input essentially doubles in importance
- **(b) weight set to `0` —** *(and what that means conceptually)* prediction becomes `0`; the neuron dies and the input loses all influence
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)* prediction becomes `-0.85`; this could mean that the input contradicts other inputs or is pointing in the opposite direction of what is normal or expected
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)* the prediction is too low because `prediction - true_answer = negative_error` (`0.85 - 1.7 = -0.85`)

### 3. The big picture

Since we don't have rules on how to find knob positions correctly, we can only guess, because another input/answer pair could break any cold calculations we could make about what the weight should be

---

## Part 4 — Your Deep Learning Problem (4 pts)

### 1. The problem

There are so many methods of learning, and each person learns a different way. Additionally, a student may have a method of learning that's best for him depending on the subject (e.g., he could learn best by demonstration in Computer Science but learn literature best by lecture). I would want to determine the best learning method for someone for a given subject using deep learning. There would be a fixed set of concepts that the student would learn using a fixed set of learning methods in different orders so he would use a different learning method fresh for each concept, and the performance from exams using these learning methods would be the input. For each subject, the model would predict which learning method would be best for a student given his performance on a number of tests administered using various learning methods.

### 2. Supervised or unsupervised, and why

It would be supervised because we would have a fixed set of defined learning methods to choose from.

### 3. What "success" would look like

Success would look like A-grade performance in a given subject and positive feedback from the student.

### 4. What could go wrong

Collecting training/testing data would be difficult, and it could be any combination of learning methods that would benefit a student best; this would be fairly intangible to the deep learning algorithm I'm thinking of. If deployed carelessly, this algorithm could possibly recommend a learning method that actually harms the student's performance in school because it either doesn't have enough information or data was collected inaccurately.

---

## Before you open the PR

- [x] All four parts answered — check against the headings above, not your memory
- [x] Part 2: both axes **and** a justification, for all four scenarios
- [ ] Part 2 #4 uses a combination you didn't already use
- [x] Part 4 #4 is answered — it's the one people skip
- [x] Your own words throughout (see below)
- [x] The file is `week1/yourname_reflection.md` — not this template

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
