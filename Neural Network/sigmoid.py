#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))
learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight
### Note: Some steps have been consolidated, so there are
###       fewer variable names than in the above sample code
# TODO: Calculate the node's linear combination of inputs and weights
h = np.dot(x, w)

### Notebook grading
import numpy as np
x_test = np.array([1, 2, 3, 4])
y_test = np.array(0.5)
w_test = np.array([0.5, -0.5, 0.3, 0.1])
learnrate_test = 0.5

h_answer = np.dot(x_test, w_test)
if h == h_answer:
    print("Good job!")
else:
    print("Try again. `h` is not correct.")


# In[3]:


# TODO: Calculate output of neural network
nn_output = sigmoid(h)

### Notebook grading
nn_output_answer = 1/(1+np.exp(-np.dot(x_test, w_test)))

if nn_output == nn_output_answer:
    print("Good job!")
else:
    print("Try again. `nn_output` is not correct.")


# In[4]:


# TODO: Calculate error of neural network
error = (y - nn_output)

### Notebook grading
error_answer = y_test - nn_output_answer

if error == error_answer:
    print("Good job!")
else:
    print("Try again. `error` is not correct.")


# In[5]:


# TODO: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
error_term = sigmoid_prime(h) * error

### Notebook grading
error_term_answer = error_answer * nn_output_answer * (1 - nn_output_answer)

if error_term == error_term_answer:
    print("Good job!")
else:
    print("Try again. `error_term` is not correct.")


# In[5]:


# TODO: Calculate change in weights
del_w = learnrate * error * x

# Optional print
print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)

### Notebook grading
del_w_answer = learnrate_test * error_answer * nn_output_answer * (1 - nn_output_answer) * x_test

test = del_w == del_w_answer
if test.all():
    print("Good job!")
else:
    print("Try again. `del_w` is not correct.")

