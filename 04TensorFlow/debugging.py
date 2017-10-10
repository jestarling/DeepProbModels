
# coding: utf-8

# In[24]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()


# # Debugging
# In normal code (in Python or otherwise), a pretty standard practice is to add conditionals in your code to look for certain behavior when something has gone awry. For example:

# In[28]:

y = np.array([1, 0, 2, 3, 4, 5], dtype=np.float32)
x = np.array([2, 1, 4, 6, 8, 10], dtype=np.float32)
t = x / y
m = np.mean(t)
v = np.var(t)
print((t - m) / np.sqrt(v))


# In the above, we're actually getting some reasonably helpful output from the iPython kernel. To debug this, we'd probably first start by adding some print statements, such as:

# In[29]:

print(np.any(np.isinf(t) | np.isnan(t)))
print(np.isinf(t) | np.isnan(t))


# which tells us that the second element of `t` is `nan` or `inf`. We can then print it:

# In[30]:

print(t[1])


# Or to be more fancy:

# In[31]:

print(t[np.where(np.isinf(t) | np.isnan(t))])


# That is using several thing at once: for a condition on t (whether it is nan or inf), find all the indices where the condition is true, and print all the values of t where it is true.

# # Debugging in TensorFlow
# 
# Doing the same thing in TensorFlow is not as straightforward. In NumPy, we can print and exit in the middle of a program. In TensorFlow, we have to use the computation graph.

# In[34]:

x_tf = tf.constant(x)
y_tf = tf.constant(y)
t_tf = x_tf / y_tf
m_tf = tf.reduce_mean(t_tf)
v_tf = tf.reduce_mean((t_tf - m_tf) ** 2)
final = (t_tf - m_tf) / tf.sqrt(v_tf)


# In[36]:

print(sess.run(final))


# So now we want to go nan/inf hunting again in TensorFlow.

# In[41]:

print(sess.run(tf.reduce_any(tf.logical_or(tf.is_inf(t_tf), tf.is_nan(t_tf)))))
print(sess.run(tf.logical_or(tf.is_inf(t_tf), tf.is_nan(t_tf))))
# Or using the shorthand for tf.logical_or
# print(sess.run(tf.is_inf(t_tf) | tf.is_nan(t_tf)))


# I can still print known elements of Tensors, but conditionals will be challenging mid-way through the computation graph.

# In[42]:

print(sess.run(t_tf[1]))


# What we did in NumPy is not strictly possible in TensorFlow (this will throw a lot of errors). However, we can still use things like `tf.cond` and `tf.where` along with any of the `tf.reduce_*` operations.

# In[57]:

# sess.run(t_tf[tf.where(tf.is_inf(t_tf) | tf.is_nan(t_tf))])


# In[52]:

# If there are any bad elements of t, use x instead for future
# computations.
new_t = tf.cond(
    tf.reduce_any(tf.is_inf(t_tf) | tf.is_nan(t_tf)),
    lambda: x_tf,
    lambda: t_tf)
print(sess.run(new_t))


# In[53]:

# For any bad elements of t, use elements of x instead for future
# computations.
new_t = tf.where(
    tf.is_inf(t_tf) | tf.is_nan(t_tf),
    x_tf,
    t_tf)
print(sess.run(new_t))


# We can even add printing in the graph if we don't mind risking having big log files as we debug (this operation logs to standard error, which doesn't appeart to show up in Jupyter):

# In[56]:

new_t = tf.cond(
    tf.reduce_any(tf.is_inf(t_tf) | tf.is_nan(t_tf)),
    lambda: tf.Print(x_tf,
                     [x_tf, y_tf, t_tf, m_tf, v_tf],
                     "return x_tf, but have side effect of printing x,y,t,m,v: ",
                     # Print up to 100 elements of each tensor in the list
                     summarize=100),
    lambda: t_tf
)
# Prints:
# I tensorflow/core/kernels/logging_ops.cc:79] \
# return x_tf, but have side effect of printing x,y,t,m,v: \
# [2 1 4 6 8 10][1 0 2 3 4 5][2 inf 2 2 2 2][inf][nan]
print(sess.run(new_t))

