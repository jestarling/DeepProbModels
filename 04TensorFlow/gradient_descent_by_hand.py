
# coding: utf-8

# # Linear Regression
# This is a more extended example than `ex02.ipynb` to look at the different ways we can do the same thing in TensorFlow. In particular, this shows one of the major themes in TensorFlow - the trade-off between detailed analytical work done by hand and the abstract pre-fabricated solutions available off-the-shelf.

# In[6]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()


# ## Data

# In[2]:

N = 100
P = 1
slope = 5
intercept = 2
X_train = np.random.normal(size=[N, P])
Y_train = X_train * slope + intercept


# ## Model
# Let's define our prediction as:
# $$\widehat{y}=x\beta+b\left(\begin{array}{c}1\\1\\\vdots\\1\end{array}\right)=x\beta+b\mathbb{1}_N$$

# In[3]:

# Define our data for training / evaluation.
# A tf.placeholder is a recipe for an input. Here, we say that each has the shape [None, 1]
# which means that we are going to provide an arbitrary number of inputs that each are one-dimensional.
x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")

# Define the model. This includes our variables (beta and b) and how they interact with our inputs.
# We'll initialize them to 0 (arbitrarily).
beta = tf.Variable(tf.zeros((1, 1)))
b = tf.Variable(tf.zeros((1, 1)))
Y_pred = tf.matmul(x, beta) + b


# ## Loss and gradient
# The loss is the sum of the squared errors:
# \begin{align}
# \mathcal{L} &= (Y - \hat{y})^\top (Y-\hat{y})
# \\
# &= (Y-(X\beta+b\mathbb{1}_N))^\top(Y-(X\beta+b\mathbb{1}_N))
# \\
# &=Y^\top Y - 2 Y^\top (X\beta+b\mathbb{1}_N)+(X\beta+b\mathbb{1}_N)^\top(X\beta+b\mathbb{1}_N)
# \end{align}
# 
# The gradient of the loss with respect to our variables ($\beta$ and $b$) is:
# $$\frac{\partial \mathcal{L}}{\partial \beta} = - 2 X^\top Y + 2 X^\top X \beta + 2 N b \bar{x}$$
# $$\frac{\partial \mathcal{L}}{\partial b} = - 2 N \bar{Y} + 2 N \bar{x}^\top\beta + 2 N b$$
# 
# Where $\bar{Y}=\frac{1}{N}\sum_i Y_i$ and $\bar{x}_j=\frac{1}{N}\sum_{i=1}^N X_{i,j}$
# 
# (as all good write-ups, the derivation of the above is left to the reader)
# 
# [Note: We could use the mean squared error instead and pick up a multiplicative factor of $\frac{1}{N}$ everywhere (loss function and each component of the gradient), but the minimum will occur at the same place, so it's somewhat moot as long as our batch size stays constant]

# # TensorFlow as calculator
# If we know an answer analytically, we can still use TensorFlow to compute it. Because we're encoding the bias term separately (trying to be more like the CS literature of having separate weights and biases when we get to neural networks, but diverging from the standard column-of-ones treatment we use in statistics), we get a slight update for the standard solution to linear regression (that we can derive by setting the gradient [above] to zero):
# 
# \begin{align}
# \left(X^\top X - N \bar{x}\bar{x}^\top\right)\hat{\beta} &= \left(X^\top Y - N\bar{x}\bar{Y}\right)
# \\
# \hat{b} &= \bar{y}-\bar{x}^\top \hat{\beta}
# \end{align}

# In[7]:

xt = tf.transpose(x)
batch_size = tf.cast(tf.shape(x)[0], tf.float32)
x_mean = tf.reshape(tf.reduce_mean(xt, 1), (-1, 1))
Y_mean = tf.reduce_mean(Y)

# Set up the linear equation to be solved, like you might do in R.
lhs = tf.matmul(xt, x) - batch_size * tf.matmul(x_mean, tf.transpose(x_mean))
rhs = tf.matmul(xt, Y) - batch_size * x_mean * Y_mean
# Solved via LU decomposition internally.
beta_hat = tf.matrix_solve(lhs, rhs)
b_hat = Y_mean - tf.matmul(tf.transpose(x_mean), beta_hat)


# ### Result
# And we get a result within $\epsilon$ of the values used to generate the data set:

# In[8]:

sess.run([beta_hat, b_hat], {x:X_train, Y:Y_train})


# ## (Stochastic) Gradient Descent
# Matrices can be messy, and if our data set is large (especially if $P$, the number of dimensions in the input, is high), we might only be able to do small batches of work at a time. In particular, we resort to gradient descent (or its minibatch cousin stochastic GD) to solve the problem.
# 
# One way to do that is to analytically compute the gradients and do a small update each pass.

# In[9]:

xt = tf.transpose(x)
x_mean = tf.reshape(tf.reduce_mean(xt, 1), (-1, 1))
Y_mean = tf.reduce_mean(Y)
batch_size = tf.cast(tf.shape(x)[0], tf.float32)

# Gradients
dloss_dbeta = - 2 * tf.matmul(xt, Y) + 2 * tf.matmul(tf.matmul(xt, x), beta) + 2 * batch_size * b * x_mean
dloss_db = - 2 * batch_size * Y_mean + 2 * batch_size * tf.matmul(tf.transpose(x_mean), beta) + 2 * b * batch_size

# Updates
learning_rate = 0.002
new_beta = beta - learning_rate * dloss_dbeta
new_b = b - learning_rate * dloss_db

# Group the updates into a single TensorFlow operation. The tf.assign operation will
# Take the second value and store it in the first Variable.
update_rule = tf.group(tf.assign(beta, new_beta), tf.assign(b, new_b))


# In[10]:

sess.run(tf.global_variables_initializer())


# ### Results
# Here, we can run on the full data set (although we could just take slices instead), and show how the parameters quickly converge.

# In[11]:

for _ in xrange(20):
    sess.run(update_rule, feed_dict={x: X_train, Y: Y_train})
    beta_np, b_np = sess.run([beta, b], feed_dict={x: X_train, Y: Y_train})
    print(beta_np[0, 0], b_np[0, 0])


# # There must be a better way
# This may seem like a lot of math for a machine learning system that's supposed to "just work". If you thought that, you'd be right.
# 
# There's a lot we can do to make this whole process easier.

# ## Automatic differentiation
# One of the ways we could do GD/SGD above a lot easier is if we didn't have to compute the derivative by hand. Well, we're in luck: TensorFlow has the `tf.gradients` function which will do symbolic differentiation with respect to variables in the graph.

# In[12]:

loss = tf.reduce_sum((Y - Y_pred) ** 2)
dloss_dbeta, dloss_db = tf.gradients(loss, [beta, b])
# dloss_db = tf.gradients(loss, b)[0]

# Updates
learning_rate = 0.002
new_beta = beta - learning_rate * dloss_dbeta
new_b = b - learning_rate * dloss_db

# Group the updates into a single TensorFlow operation. The tf.assign operation will
# Take the second value and store it in the first Variable.
update_rule = tf.group(tf.assign(beta, new_beta), tf.assign(b, new_b))


# ### Results
# And here we are, getting the same results for the same learning rate, without having to compute the gradient by hand.

# In[13]:

sess.run(tf.global_variables_initializer())


# In[14]:

for _ in xrange(20):
    sess.run(update_rule, feed_dict={x: X_train, Y: Y_train})
    beta_np, b_np = sess.run([beta, b], feed_dict={x: X_train, Y: Y_train})
    print(beta_np[0, 0], b_np[0, 0])


# ## Removing the explicit update
# The next annoying thing in our code above is that we've had to implement gradient descent ourselves. It would be nice if we didn't have to do that ourselves.
# 
# TensorFlow has several built-in optimizers:
# - Gradient descent
# - Adam
# - Adagrad
# - RMSProp
# - etc.
# 
# We can use them to compute gradients in a form that's pairs of gradients and variables. That can then be applied back directly. This also gives us the opportunity to update any of the gradients, such as clipping them if they are too large.

# In[15]:

loss = tf.reduce_sum((Y - Y_pred) ** 2)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.002)
grads_and_vars = opt.compute_gradients(loss, [beta, b])

# Could introduce clipping here.

# This now does all the assignments at once.
train_op = opt.apply_gradients(grads_and_vars)


# ### Results

# In[16]:

sess.run(tf.global_variables_initializer())


# In[17]:

for _ in xrange(20):
    sess.run(train_op, feed_dict={x: X_train, Y: Y_train})
    beta_np, b_np = sess.run([beta, b], feed_dict={x: X_train, Y: Y_train})
    print(beta_np[0, 0], b_np[0, 0])


# ## Simple minimization
# If we don't need clipping, there's a simpler method we can use that handles all the work for us: `minimize`.

# In[18]:

loss = tf.reduce_sum((Y - Y_pred) ** 2)
# This automatically does symbolic differentiation for each
# Variable in the graph, rather than having to specify
# them all individually.
train_op = (
    tf.train.GradientDescentOptimizer(learning_rate=0.002)
        .minimize(loss)
)


# ### Results

# In[19]:

sess.run(tf.global_variables_initializer())


# In[20]:

for _ in xrange(20):
    sess.run(train_op, feed_dict={x: X_train, Y: Y_train})
    beta_np, b_np = sess.run([beta, b], feed_dict={x: X_train, Y: Y_train})
    print(beta_np[0, 0], b_np[0, 0])


# ## Simpler loss
# At this point, the only thing in the graph that's not part of the model or something "magical" like the Optimizer is the loss function. Wouldn't it be great if we didn't have to compute that by hand?

# In[21]:

loss = tf.losses.mean_squared_error(labels=Y, predictions=Y_pred)
train_op = (
    tf.train.GradientDescentOptimizer(learning_rate=0.2)
        .minimize(loss)
)


# ### Results

# In[22]:

sess.run(tf.global_variables_initializer())


# In[23]:

for _ in xrange(20):
    sess.run(train_op, feed_dict={x: X_train, Y: Y_train})
    beta_np, b_np = sess.run([beta, b], feed_dict={x: X_train, Y: Y_train})
    print(beta_np[0, 0], b_np[0, 0])


# ## Customization
# Now that we have all the quick-and-easy built-in functions in play, we can start using some more advanced techniques:

# In[24]:

loss = tf.losses.huber_loss(labels=Y, predictions=Y_pred)
train_op = (
    tf.train.RMSPropOptimizer(learning_rate=0.5)
        .minimize(loss)
)
sess.run(tf.global_variables_initializer())
for _ in xrange(20):
    sess.run(train_op, feed_dict={x: X_train, Y: Y_train})
    beta_np, b_np = sess.run([beta, b], feed_dict={x: X_train, Y: Y_train})
    print(beta_np[0, 0], b_np[0, 0])

