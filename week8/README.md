# Week 8: Forgetting

This week is learning methods for efficiency or constraining a model to help it better generalize and not overfit to training data.

## Part 1: Memorize

# Oliver
In this section, we split our training and validation data sets, trained on the training data and kept checking how 
it worked against the training and validation datasets for accuracy. The training accuracy quickly exploded to 0.998
almost immediately while the validation set went up quickly and then dropped down and leveled off at around 0.86~.
This is a classic instance of overfitting to the training data and not generalizing well.


## Part 2: Early Stopping

# Josiah
In part 2, we implemented a mechanism to ensure we didn't go too far and lose validation accuracy over time. So we
would stop near 0.89 instead of slowly descending towards 0.86. While this doesn't inherently help fix the overfitting,
it does help when our model would degrade over training iterations and cut us off before wasting compute cycles for 
worse performance.


## Part 3: Dropout

# Nathanael
In part 3 we implemented dropout, a method designed to constrain a model during training and ensure that a single
set of neurons aren't carrying all the training information. This helps to reduce overfitting by not allowing the
environment to become static since the training data doesn't change. Implementing this resulted in a decrease in
training accuracy, but a significant increase in validation accuracy.


## Part 4: Batching

# Caleb
In part 4, is a method that improves efficiency as well as allowing us to have a more stable traning progression
with less harsh jumps in accuracy. We perform weight updates based on an averaged amount that is specified instead
of one at a time with an update.


## Unit Testing

# Nathanael
 PASS: test_part_1_overfitting
 PASS: test_part_2_best_early_stopping
 PASS: test_part_3_correct_dropout
 PASS: test_part_3_dropout_improves
 PASS: test_parts_1234_no_test_leakage

 All of the tests passed after a fix for part 3. 


