import tensorflow as tf
import sys
import math

# The function to minimize w.r.t. x using
# gradient descent
def function(x):
    return x*x

# this is the learning rate, it decides how mutch the
# variable should be updated in one step of gradient
# descent algorithm
# X = X - lr * df/dX
lr = 0.001

# Random initialization of the X variable
X = tf.Variable(initial_value=4.0,shape=())

# Value of f(X) before the gradient descent step
prev_function_x = function(X)
# Value of f(X) after the gradient descent step (initialized with infinity)
post_function_x = sys.maxsize

# We continue the execution of gradient descent until
# |f(X) - f(X')| < epsilon
epsilon = 0.00001

# iterations counter
it = 0

while abs(prev_function_x - post_function_x) > epsilon:
    # update value of f(X) before applying
    # gradient descent
    prev_function_x = function(X)

    # GradientTape records all the operations that are
    # executed on the watched variables and compute the
    # gradient with respect to those variables
    with tf.GradientTape() as tape:
        tape.watch(X)
        f = function(X)

        # automatic computation of the
        # gradient. The gradient is a vector where
        # the ith component is the derivative w.r.t. the
        # ith watched variable
        g = tape.gradient(f,[X])

        # Update the variable X performing one step of the
        # gradient descent algorithm
        # X = X - lr * df/dX
        X.assign_sub(lr*g[0])

        # update the value of f(X) after applying
        # one step of gradient descent
        post_function_x = function(X)

        it += 1
        print("it: {}, X = {}, f(X) = {}".format(it,X.numpy(),function(X).numpy()))

print("\nLocal Optimum found:\nX = {}, f(X) = {}".format(X.numpy(),function(X).numpy()))   