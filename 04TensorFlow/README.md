# tf_intro
Simple introduction to TensorFlow.

This repository contains a gentle introduction to TensorFlow, followed by some more practical examples, and some extensions.

All the code is duplicated: once as a Jupyter notebook (`.ipynb`) and once as a normal Python file (`.py`). If you just want to *view* code on GitHub, take a look at the `.ipynb` version because the formatting will be nice. If you want to run the files but don't have Jupyter/iPython, go ahead and use the `.py` versions -- the code is the same, the Jupyter versions just have a little bit more formatting.

`ex01.(ipynb/py)` and `ex02.(ipynb/py)` are similar in intent to TensorFlow's own "getting started" guide (the first two sections of it, anyway):

- `ex01.(ipynb/py)` is where you should start if you're new to TensorFlow. It goes over the syntax and general way that TensorFlow operates.
- `ex02.(ipynb/py)` is an intermediate step, that shows how to use some of the main workhorses of TensorFlow in linear regression.

`gradient_descent_by_hand.(ipynb/py)` is next, showing the many different ways you can do the same thing in TensorFlow. Often, there are existing tools for almost anything you need, and we'd encourage you to use them. Almost always, you should be able to focus on only what you're doing research in; if you aren't researching new methods of optimization, just stick with the existing Optimizer options.

`edward_getting_started.(ipynb/py)` uses an extension to TensorFlow called [Edward](http://edwardlib.org/) that makes things like variational inference fairly straightforward. This guide is from their documentation, but copied here for reference.