<div align="center">
  <p>
    <img src="docs/artifial_neural.png" _style="width:80px;height:80px;">
  </p>

  <h3>Artificial Neural</h3>
</div>

<div align="center">
    Implementation of a Logistic Regression model from scratch using Python and NumPy.
</div>



### Forward Propagation


We consider a dataset containing $m$ observations and $n$ features.

In this example, we have 3 observations and 2 features:

$$
\begin{aligned}
X &=
\begin{pmatrix}
x_1^{(1)} & x_2^{(1)} \\
x_1^{(2)} & x_2^{(2)} \\
x_1^{(3)} & x_2^{(3)}
\end{pmatrix}
\\[1em]
W &=
\begin{pmatrix}
w_1 \\
w_2
\end{pmatrix}
\end{aligned}
$$

- $X$: input data matrix.
- $W$: weights vector.
- $b$: bias.
- $m$: number of observations.
- $n$: number of features.

### Model

We compute the weighted sum for each observation:

$$
Z = X \cdot W + b
$$

Which gives:

$$
Z =
\begin{pmatrix}
x_1^{(1)}w_1 + x_2^{(1)}w_2 + b \\
x_1^{(2)}w_1 + x_2^{(2)}w_2 + b \\
x_1^{(3)}w_1 + x_2^{(3)}w_2 + b
\end{pmatrix}
=
\begin{pmatrix}
z^{(1)} \\
z^{(2)} \\
z^{(3)}
\end{pmatrix}
$$


We apply the sigmoid function to transform the weighted sums into probabilities between 0 and 1:

$$
A = \sigma(Z) = \frac{1}{1 + e^{-Z}}
$$


### Loss Function

We use Binary Cross-Entropy (Log Loss) to measure the difference between predicted probabilities and actual labels.

$$
L = -\frac{1}{m}\sum_{i=1}^{m}
\left[
y^{(i)}\log\left(A^{(i)}\right)
+
\left(1-y^{(i)}\right)\log\left(1-A^{(i)}\right)
\right]
$$

- $y^{(i)}$: actual label of observation $i$.
- $A^{(i)}$: predicted probability for observation $i$.
- $L$: loss value.

A lower loss indicates better predictions.


### Gradient Descent

We calculate the partial derivatives of the loss with respect to the weights and bias.


#### Weights gradient

$$
\frac{\partial L}{\partial W}
=
\frac{1}{m}X^T(A-y)
$$

#### Bias gradient

$$
\frac{\partial L}{\partial b}
=
\frac{1}{m}\sum_{i=1}^{m}
\left(A^{(i)}-y^{(i)}\right)
$$

#### Parameter updates

We update the parameters at each iteration using Gradient Descent:

**Weights update:**

$$
W = W - \alpha \cdot \frac{\partial L}{\partial W}
$$

#### Bias update

$$
b = b - \alpha \cdot \frac{\partial L}{\partial b}
$$

Where $\alpha$ represents the learning rate.