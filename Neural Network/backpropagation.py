#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-np.array(x, dtype=float)))
x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5
weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])
weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass

## TODO: Calculate output error
error = target - output

### Notebook grading
error_answer = target - output

if error == error_answer:
    print("Well done!")
else:
    print("Try again. Something is wrong in your submission")


# In[2]:


# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output)

### Notebook grading

del_err_output_answer = error_answer * output * (1 - output)

if output_error_term == del_err_output_answer:
    print("Well done!")
else:
    print("Try again. Something is wrong in your submission")


# In[3]:


# TODO: Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * hidden_layer_output * (1 - hidden_layer_output)# replace with your code

### Notebook grading

del_err_hidden_answer = np.dot(del_err_output_answer, weights_hidden_output) * \
                    hidden_layer_output * (1 - hidden_layer_output)

test_h = del_err_hidden_answer == hidden_error_term

if test_h.all():
    print("Well done!")
else:
    print("Try again. Something is wrong in your submission")


# In[6]:


# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output # replace with your code

### Notebook grading
delta_w_h_o_answer = learnrate * del_err_output_answer * hidden_layer_output

test_w_h_o = delta_w_h_o_answer == delta_w_h_o

if test_w_h_o.all():
    print("Well done!")
else:
    print("Try again. Something is wrong in your submission")


# In[10]:


# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:,None] # replace with your code
print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)

### Notebook grading

delta_w_i_h_answer = learnrate * del_err_hidden_answer * x[:, None]


test_w_i_h = delta_w_i_h_answer == delta_w_i_h

if test_w_i_h.all():
    print("Well done!")
else:
    print("Try again. Something is wrong in your submission")


# In[ ]:




