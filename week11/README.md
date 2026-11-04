# Week 11 = The Old Tongue

This week we used a new dataset of movie reviews. This contained full sentances, and building a
tokenizer, we started to create a similarity, and how it can be predictive to fill in a blank.

## Part 1 - Preprocess
For preprocessing, we would tokenize the word inputs from the strings and label positive and
negative review values. In here we have regularize and tokenize to be called.

## Part 2 - Sentiment
When training on a bag of words classifier, we can train to get a tone for if it is negative or
positive. One thing that is very noticeable is that it does show signs of overfitting pretty
quickly as there looks to be a pretty heavy difference between training and test accuracy.

## Part 3 - Similarity
Using the sentiment, we can vectorize the words to form a positional similarity that can be shown
and compared to find other nearby words. With a much larger dataset this would be reliable as 
many words still don't quite work with this, but many descriptors do work.

## Part 4 - Filling in The Blank
Without grabbing all words, training instead on just 6 with 1 of them being the right one and 5
being weighted downward. This saves on processing efficiency since the weights are already quite
high.

## Part 5 - Analogy
Using the similarity, we can take two words, as the example, of terrible is to good, as bad is 
to __. Thanks to the vectorization, we can perform computations to get comparisons that can work
in cases like this.

# Testing


