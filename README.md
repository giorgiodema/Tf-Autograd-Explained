# Gradient Descent with Autograd
This is a simple implementation of Gradient Descent using Tensorflow Autograd to compute the
local optimum of a 1D function. In the example the function is the parabola:
<br>
<b>
f(x) = x<sup>2</sup>
</b>
<br>
And the variable X is initialized to 4. The algorithm stops when after the computation of one step of gradient descent the deltha is less then epsilon, meaning that if x is the value of the variable X before the update and x' is the value of X after the update, the computation stops when:
<br>
<b>
|f(x)-f(x')| < epsilon
</b>
<br>
