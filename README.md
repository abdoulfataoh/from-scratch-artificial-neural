<p align="center">
  <img src="docs/artifial_neural.png" width="500" height="220" alt="Logo">
</p>

<h3 align="center">Artificial Neural</h3>

<p align="center">
  Implementation of a Logistic Regression model from scratch using Python and NumPy.
</p>

---

### Forward Propagation

We consider a dataset containing <img src="https://latex.codecogs.com/svg.image?m" valign="middle" alt="m"> observations and <img src="https://latex.codecogs.com/svg.image?n" valign="middle" alt="n"> features.

In this example, we have 3 observations and 2 features:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?X=\begin{pmatrix}x_1^{(1)}&x_2^{(1)}\\x_1^{(2)}&x_2^{(2)}\\x_1^{(3)}&x_2^{(3)}\end{pmatrix}" alt="X matrix">
</div>

<br>

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?W=\begin{pmatrix}w_1\\w_2\end{pmatrix}" alt="W vector">
</div>

- **X**: input data matrix.
- **W**: weights vector.
- **b**: bias.
- **m**: number of observations.
- **n**: number of features.

### Model

We compute the weighted sum for each observation:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?Z=X\cdot&space;W&plus;b" alt="Z = X * W + b">
</div>

Which gives:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?Z=\begin{pmatrix}x_1^{(1)}w_1&plus;x_2^{(1)}w_2&plus;b\\x_1^{(2)}w_1&plus;x_2^{(2)}w_2&plus;b\\x_1^{(3)}w_1&plus;x_2^{(3)}w_2&plus;b\end{pmatrix}=\begin{pmatrix}z^{(1)}\\z^{(2)}\\z^{(3)}\end{pmatrix}" alt="Z detailed">
</div>

<br>

We apply the sigmoid function to transform the weighted sums into probabilities between 0 and 1:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?A=\sigma(Z)=\frac{1}{1&plus;e^{-Z}}" alt="Sigmoid">
</div>

### Loss Function

We use Binary Cross-Entropy (Log Loss) to measure the difference between predicted probabilities and actual labels.

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?L=-\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\left(A^{(i)}\right)&plus;\left(1-y^{(i)}\right)\log\left(1-A^{(i)}\right)\right]" alt="Loss Function">
</div>

- <img src="https://latex.codecogs.com/svg.image?y^{(i)}" valign="middle">: actual label of observation *i*.
- <img src="https://latex.codecogs.com/svg.image?A^{(i)}" valign="middle">: predicted probability for observation *i*.
- **L**: loss value.

A lower loss indicates better predictions.

### Gradient Descent

We calculate the partial derivatives of the loss with respect to the weights and bias.

#### Weights gradient

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;L}{\partial&space;W}=\frac{1}{m}X^T(A-y)" alt="Weights gradient">
</div>

#### Bias gradient

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;L}{\partial&space;b}=\frac{1}{m}\sum_{i=1}^{m}\left(A^{(i)}-y^{(i)}\right)" alt="Bias gradient">
</div>

#### Parameter updates

We update the parameters at each iteration using Gradient Descent:

**Weights update:**

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?W=W-\alpha\cdot\frac{\partial&space;L}{\partial&space;W}" alt="Weights update">
</div>

#### Bias update

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?b=b-\alpha\cdot\frac{\partial&space;L}{\partial&space;b}" alt="Bias update">
</div>

Where <img src="https://latex.codecogs.com/svg.image?\alpha" valign="middle"> represents the learning rate.