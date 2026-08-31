Week 2: Forward Propagation

As a whole, in five mostly-consecutive parts, this project runs forward propagation on a simple two-layer neural network without an activation function.

In Part 1, we have the code for a single prediction: one weight, one input, four values to test for that input.

In Part 2, we have the code for multiple inputs: three weights, one for each input, and three inputs, with four values to test for each of the three inputs.

In Part 3, we have the code for several outputs at once, but only one input: three weights, one for each neuron, and one input, with four values to test for that input.

In Part 4, we unify Part 2 and Part 3, giving us nine weights, three per neuron, defining how that neuron responds to each of the three inputs (and of course still four values to test for each of the three inputs).

And lastly, in Part 5 we add a hidden layer, running Part 4's code once to generate a hidden layer, and then once again on that hidden layer's output as input into the next layer, with 18 total weights.


part1_single_prediction.py has Part 1, part2_multiple_inputs.py has Part 2, part3_multiple_outputs.py has Part 3, part4_multi_in_multi_out.py has Part 4, part5_two_layers.py has Part 5, and forward_propagation.ipynb has all the code unified in one place with a bit of explanation text.