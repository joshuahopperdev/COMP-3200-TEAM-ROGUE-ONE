# Blade-angle values are larger -- they need alpha!
blade_angle = [8.5, 9.5, 9.9, 9.0]
clean = [1, 1, 0, 1]

"""
HOLOCRON NOTE — the other fix — and why not this week
Some of you will see the real asymmetry here: the blade angle only needs a small α
because we fed it raw degrees. Divide every reading by 10 and the numbers land in the
same range as the balance data, where α = 0.1 already works. You are right — and
you have already met this. Episode II named normalization and gave you both recipes
(min–max and the z-score) on these very readings: blade angle 8.5 against balance 0.65,
the same two numbers, the same shared weight of 0.1. What Episode II could not
show you yet is what that scale gap does once the network starts learning. That is
Thursday’s slide.
Do not use it this week. Part 3 is asking you to feel what α does with your hands, and
rescaling the inputs makes the question disappear before you have answered it. Tune
α against the raw degrees. If you want the extra credit of a good observation, add a
comment block on which of the two fixes you would reach for outside this assignment,
and what each one costs. You build the other fix next week, in Part 1b of Assignment 5.
And if you try it anyway — a warning worth having. Episode II’s min–max recipe
maps the smallest value in a column to exactly 0. Our smallest blade angle is 8.5, so
min–max turns it into a zero input — and a zero input has weight_delta = delta *
0 = 0. That weight can never move again. No error, no warning, no exception: the
prediction simply sits at 0 forever while the other three sensings learn normally. A
technique that is correct in one setting can fail silently in another. Read your numbers,
not just your code.
"""