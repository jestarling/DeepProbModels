
# coding: utf-8

# # Basics of TensorFlow
# This is a starter for understanding the syntax of TensorFlow.
# 
# First, let's import tensorflow.

# In[1]:

import tensorflow as tf


# ## Operations in the TensorFlow Session
# TensorFlow works with the notion of a Session. When you code mathematical operations, you're really constructing a graph of computations that gets run later in a Session.
# 
# For now, we'll be verbose, but later, we'll be able to use some of the nice built-in shortcuts. This is trying to build your understanding of what's going on "under the hood" to help you get used to it.

# In[2]:

# Here, we start with two constants and add them together.
a = tf.constant(20)
b = tf.constant(5)
result = tf.add(a, b)
result


# You'll note that the output of `result` isn't 25, but is instead an Operation. We've constructed the computation graph, but need to actually run it in a Session:

# In[3]:

sess = tf.Session()
# This gives us the result we'd expect
sess.run(result)


# ## Beyond constants
# Being able to do math with constants is fun and all, but leaves a lot to be desired.
# 
# First off, let's look at placeholders (and use some of the pretty TensorFlow syntax):

# In[4]:

x = tf.placeholder(tf.float64)
y = x + 3
z = y * 2
sess.run(z, feed_dict={x: 4})


# ## Broadcasting
# This should be a familar concept to NumPy users, but you can broadcast Tensors, which is one of the major strengths of TensorFlow (besides being able to run it on a GPU).
# 
# As long as the operations make sense, you can easily apply a function element-wise. Other Tensors in the operation will be broadcast [See the rules here](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html), such as turning scalars into matrices, as in this example:

# In[5]:

sess.run(z, feed_dict={x: [[1, 2], [3, 4]]})


# ## Overriding graph elements
# We can also override values in the graph:

# In[6]:

sess.run(z, feed_dict={x: 4, y: 11})

