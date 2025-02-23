#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))
# Network size
N_input = 4
N_hidden = 3
N_output = 2
np.random.seed(42)

# Make some fake data
X = np.random.randn(4)
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

# TODO: Make a forward pass through the network (hidden layer)
hidden_layer_in = np.dot(X, weights_input_to_hidden) # replace with your code
hidden_layer_out = sigmoid(hidden_layer_in) # replace with your code

print('Hidden-layer Output:')
print(hidden_layer_out)

### Notebook grading
hidden_layer_in_ans = np.dot(X, weights_input_to_hidden)
hidden_layer_out_ans = sigmoid(hidden_layer_in_ans)

test_hidden_layer_in = hidden_layer_in_ans == hidden_layer_in
test_hidden_layer_out = hidden_layer_out_ans == hidden_layer_out

if test_hidden_layer_out.all():
    print("Good job! You got the correct calculation on the hidden layer.")
else:
    print('Try again. hidden_layer_out should be {}'.format(hidden_layer_out_ans))


# In[5]:


# TODO: Make a forward pass through the network (output layer)

output_layer_out_answer = [0.49815196,  0.48539772]

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output) # replace with your code
output_layer_out = sigmoid(output_layer_in) # replace with your code
print('Output-layer Output:')
print(output_layer_out)

### Notebook grading
output_layer_in_ans = np.dot(hidden_layer_out_ans, weights_hidden_to_output)
output_layer_out_ans = sigmoid(output_layer_in_ans)

test_output_layer_out = output_layer_out_ans == output_layer_out

if test_output_layer_out.all():
    print("Good job! You got the correct calculation on the output layer.")
else:
    print('Try again. output_layer_out should be {}'.format(output_layer_out_ans))


# In[ ]:




