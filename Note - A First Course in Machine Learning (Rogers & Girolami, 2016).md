# Note - A First Course in Machine Learning (Rogers & Girolami, 2016)

## Chapter 1: Learning Modelling - A least Squares Approach

### Linear Modelling

key words: attribute, response, model, linear relationship, function, parameter, intercept, gradient, squared loss function, average loss, argmin, analytical solution, partial derivative.

summary:

- Create a linear model that encapsulates the relationship between a set of attributes and a set of responses.
  - $f(x_n; w_0, w_1) = w_0 + w_1x$
- Define a loss function to fit / learn this model, as a way of objectively identifying how good a particular model was.
  - squared loss function: $ \mathcal{L_n}(t_n, f(x_n; w_0, w_1)) = (t_n - f(x_n; w_0, w_1))^2$
  - average loss across the whole dataset: $\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \mathcal{L_n}(t_n, f(x_n; w_0, w_1))$
- Derive exact expressions for the values of the model parameters that minimised the loss and therefore corresponded to the best function, using the squared loss.
  - argmin: find the argument that minimises ...
  - Method of turning points:
    - differentiating the loss function
    - equating the derivatives to zero
  - expressions: $$\$$

------

## Chapter 2: Learning Modelling - A Maximum Likelihood Approach

### 2.1 Errors As Noise

### 2.2Random Variables and Probability

key words: random variables, random events, discrete random variables, sample space.

####2.2.1 random variable

- `discrete random variables`: used for random events for which we can systematically list all possible outcomes.

​		`sample space` is the collection of possible outcomes.

- `continue random variables` 

​	***Note***: It is a common convention to use *upper-*case letters to describe random variables and *lower*-case ones for possible values that the random variable can take.

​	***Shorthand***: the vector $\vec{x} = [x_1, x_2, ..., x_N]^T$ can express the values that could be taken on by random variable $X_1, X_2, ..., X_N$. $p(\vec{x}) = p(x_1, x_2, ..., x_N) = P(X_1=x_1, X_2=x_2, ..., X_N=x_N)$

####2.2.2 probability and distributions

- Two important rules governing probabilities：

​		$$0<=P(Y=y)<=1$$

​		$$\sum_y P(Y=y) = 1$$

​	***shorthand***: $P(Y=y) = P(y)$

- probability `distribution`: the set of all of the possible outcomes and their probabilities.

  ​	the total probability (1) is distributed or shared out over all possible outcomes.

#### 2.2.3 adding probabilities

#### 2.2.4 conditional probabilities

#### 2.2.5 joint probabilities

$$P(Y=y, X=x) = P(Y=y) \times P(X=x)$$

$$P(y_1, y_2, ..., y_J) = \prod^J_{j=1}{P(y_j)}$$

$$P(Y=y, X=x) = P(Y=y |X=x) \times P(X=x) = P(X=x|Y=y) \times P(Y=y)$$

#### 2.2.6 Marginalization

- This is done by summing the joint probabilities over all possible values of $X$:

  ​	$$P(Y=y)=\sum_x{P(Y=y, X=x)}$$

- The `marginal` for joint probabilities $P(Y_j=y_j)$ of $J$ random variables:

  ​	$P(Y_j=y_j)=P(y_j)=\sum_{y_1, ..., y_{j-1}, y_{j+1}, ..., y_J} {P(y_1, ..., y_J)}$

#### 2.2.7 aside - Bayes's rule

​	$P(X=x | Y=y) = \cfrac{P(Y=y|X=x)P(X=x)}{P(Y=y)}$

#### 2.2.8 expectations

- An expectation tells us what value we would expect some function $f(X)$ of a random variable $X$ to take and is defined as

  ​	$E_{P(x)} {f(X)} = \sum_x {f(x)P(x)}$

- The expectation of a sum of different functions is equal to a sum of the individual expectations:

  ​	$E_{P(x)} {f(X) + g(X)} = E_{P(x)} {f(X)} + E_{P(x)} {g(X)}$

- The two most common expectations:

  - the `mean` of distribution or the expected value of $X$:

    ​	$E_{P(x)}\{X\} = E_{P(x)} {f(X)} = \sum_x {xP(x)}$

  - the `variance`: a measure of how variable the random variable is defined as the expected squared deviation from the mean

    ​	$var\{ X\} = E_{P(x)} \{ (X-E_{P(x)}\{X\})^2\} = E_{P(x)}\{X^2\} - E_{P(x)}\{X\}^2$

- Expectations computed for vector random variables

  - it is defined as: $E_{P(\vec{x})} {f(\vec{x})} = \sum_{\vec{x}} {f(\vec{x})P(\vec{x})}$$

  - the mean vector

    ​	$E_{P(\vec{x})}  \{\vec{x}\} = \sum_{\vec{x}} {\vec{x} P(\vec{x})}$

  - the variance vector

    - the concept of variance is generalized to a `covariance` matrix, which is defined as

      ​	$cov\{ \vec{x} \} = E _{P(\vec{x{}})} \{ (\vec{x} - E_{P(\vec{x})}  \{\vec{x}\}) (\vec{x} - E_{P(\vec{x})}  \{\vec{x}\})^T \}$

      - If $\vec{x}$ is a vector of length $D$, then $cov\{ \vec{x} \}$ is a $D \times D$ matrix.

    - $cov\{x\} = E_{E_{P(\vec{x})}  \{ \vec{x}\vec{x}^T \}-E_{P(\vec{x})}  \{\vec{x}\} E_{P(\vec{x})}  \{\vec{x}\}^T$ 

    - $cov\{x\} = E_{P(\vec{x})}  {\{ \vec{x}\vec{x}^T \} } - E_{P(\vec{x})}  \{\vec{x}\} E_{P(\vec{x})}  \{\vec{x}\}^T $ 

###2.3Popular Discrete Distributions

### 2.4 Continuous Random Variables - Density Function

### 2.5 Popular Continuous Density Function

### 2.7 Thinking Generatively... Continued

### 2.8 Likelihood

### 2.9 The Bias-Variance Trade-Off

### 2.10 Effect of Noise on Parameter Estimates

### 2.11 Variability in Predictions

##  

------

##Chapter 3: Bayesian Approach



------

##Chapter 4: Bayesian Inference



------

##Chapter 5: Classification



------

##Chapter 6: Clustering



------

##Chapter 7: Principal Components Analysis and Latent Variable Models

------

## Chapter 8: Gaussian Processes

------

## Chapter 9: Markov Chain Monte Carlo Sampling

------

## Chapter 10: Advanced Mixture Modeling