#!/usr/bin/env python
# coding: utf-8

# # Classifying Fashion-MNIST
# 
# Now it's your turn to build and train a neural network. You'll be using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), a drop-in replacement for the MNIST dataset. MNIST is actually quite trivial with neural networks where you can easily achieve better than 97% accuracy. Fashion-MNIST is a set of 28x28 greyscale images of clothes. It's more complex than MNIST, so it's a better representation of the actual performance of your network, and a better representation of datasets you'll use in the real world.
# 
# <img src='assets/fashion-mnist-sprite.png' width=500px>
# 
# In this notebook, you'll build your own neural network. For the most part, you could just copy and paste the code from Part 3, but you wouldn't be learning. It's important for you to write the code yourself and get it to work. Feel free to consult the previous notebooks though as you work through this.
# 
# First off, let's import our resources and download the Fashion-MNIST dataset from `tensorflow_datasets`. 

# ## Import Resources

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


# In[6]:


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# In[7]:


print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


# In[8]:


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# ## Load the Dataset
# 
# We are now going to load the Fashion-MNIST dataset using `tensorflow_datasets` as we've done before. In this case, however, we are going to omit the `split` argument.  This means that `tensorflow_datasets` will use the default value for `split` which is `split=None`. When `split=None`, `tensorflow_datasets` returns a **dictionary** with all the splits available for the dataset you are loading. However, if the split is given explicitly, such as `split='train'`, then `tensorflow_datasets` returns a `tf.data.Dataset` object.
# 
# In our case, we are going to load the `fashion_mnist` dataset. If we look at the [documentation](https://www.tensorflow.org/datasets/catalog/fashion_mnist#statistics) we will see that this particular dataset has 2 splits, namely a `train` and a `test` slipt. We also see that the `train` split has 60,000 examples, and that the `test` split has 10,000 examples. 
# 
# Now, let's load the `fashion_mnist` dataset and inspect the returned values.

# In[9]:


dataset, dataset_info = tfds.load('fashion_mnist', as_supervised=True, with_info=True)


# In[10]:


# Check that dataset is a dictionary
print('dataset has type:', type(dataset))

# Print the keys of the dataset dictionary
print('\nThe keys of dataset are:', list(dataset.keys()))


# In the cell below, we are going to save the training data and the test data into different variables.

# In[11]:


training_set, test_set = dataset['train'], dataset['test']


# Now, let's take a look at the `dataset_info`

# In[12]:


# Display the dataset_info
dataset_info


# We can access the information in `dataset_info` very easily. As we can see, the `features` and `splits` info are contained in dictionaries. We can access the information we want by accessing the particular key and value in these dictionaries. We start by looking at the values of particular keys in these dictionaries:

# In[13]:


dataset_info.features['image']


# In[14]:


dataset_info.features['label']


# In[15]:


dataset_info.splits['train']


# We can now use dot notation to access the information we want. Below are some examples. 

# In[16]:


shape_images = dataset_info.features['image'].shape
num_classes = dataset_info.features['label'].num_classes

num_training_examples  = dataset_info.splits['train'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples

print(f'There are {num_classes:,} classes in our dataset')
print(f'The images in our dataset have shape: {shape_images}')

print(f'\nThere are {num_test_examples:,} images in the test set')
print(f'There are {num_training_examples:,} images in the training set')


# ## Explore the Dataset
# 
# The images in this dataset are 28 $\times$ 28 arrays, with pixel values in the range `[0, 255]`. The *labels* are an array of integers, in the range `[0, 9]`. These correspond to the *class* of clothing the image represents:
# 
# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th> 
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td> 
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td> 
#   </tr>
#     <tr>
#     <td>2</td>
#     <td>Pullover</td> 
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>Dress</td> 
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>Coat</td> 
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>Sandal</td> 
#   </tr>
#     <tr>
#     <td>6</td>
#     <td>Shirt</td> 
#   </tr>
#     <tr>
#     <td>7</td>
#     <td>Sneaker</td> 
#   </tr>
#     <tr>
#     <td>8</td>
#     <td>Bag</td> 
#   </tr>
#     <tr>
#     <td>9</td>
#     <td>Ankle boot</td> 
#   </tr>
# </table>
# 
# Each image is mapped to a single label. Since the *class names* are not included with the dataset, we create them here to use later when plotting the images:

# In[17]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


# In[18]:


for image, label in training_set.take(1):
    print('The images in the training set have:\n\u2022 dtype:', image.dtype, '\n\u2022 shape:', image.shape)


# In[19]:


for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap= plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is:', label)
print('The class name of this image is:', class_names[label])


# ## Create Pipeline

# In[20]:


def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)


# ## Build the Model
# 
# > **Exercise:** Here you should define your own neural network. Feel free to create a model with as many layers and neurons as you like. You should keep in mind that as with MNIST, each image is 28 $\times$ 28 which is a total of 784 pixels, and there are 10 classes. Your model should include at least one hidden layer. We suggest you use ReLU activation functions for the hidden layers and a softmax activation function for the output layer.

# In[24]:


## Solution
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])


# ## Train the Model
# 
# > **Exercise:** Compile the model you created above using an `adam` optimizer, a `sparse_categorical_crossentropy` loss function, and the `accuracy` metric. Then train the model for 5 epochs. You should be able to get the training loss below 0.4.

# In[27]:


## Solution
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(training_batches, epochs = 5)


# ## Evaluate Loss and Accuracy on the Test Set
# 
# Now let's see how the model performs on the test set. This time, we will use all the examples in our test set to assess the loss and accuracy of our model. Remember, the images in the test are images the model has never seen before.

# In[29]:


loss, accuracy = model.evaluate(testing_batches)

print(f'\nLoss on the TEST Set: {loss:,.3f}')
print(f'Accuracy on the TEST Set: {accuracy:.3%}')


# ## Check Predictions

# In[31]:


for image_batch, label_batch in testing_batches.take(1):
    ps = model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]
    first_label = label_batch.numpy()[0]

fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(first_image, cmap = plt.cm.binary)
ax1.axis('off')
ax1.set_title(class_names[first_label])
ax2.barh(np.arange(10), ps[0])
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(class_names, size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()


# In[ ]:




