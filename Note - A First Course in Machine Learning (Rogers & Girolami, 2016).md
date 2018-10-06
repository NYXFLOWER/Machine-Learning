# Note - A First Course in Machine Learning (Rogers & Girolami, 2016)

------

------

## Chapter 1: Learning Modelling - A least Squares Approach

### Linear Modelling

key words: attribute, response, model, linear relationship, function, parameter, intercept, gradient, squared loss function, average loss, argmin, analytical solution, partial derivative.

summary:

- Create a linear model that encapsulates the relationship between a set of attributes and a set of responses.
  ​	$f(x_n; w_0, w_1) = w_0 + w_1x$

- Define a loss function to fit / learn this model, as a way of objectively identifying how good a particular model was.

  ​	squared loss function: $ \mathcal{L_n}(t_n, f(x_n; w_0, w_1)) = (t_n - f(x_n; w_0, w_1))^2$

  ​	average loss across the whole dataset: $\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \mathcal{L_n}(t_n, f(x_n; w_0, w_1))$

- Derive exact expressions for the values of the model parameters that minimised the loss and therefore corresponded to the best function, using the squared loss.
  - argmin: find the argument that minimises ...
  - Method of turning points:
    - differentiating the loss function
    - equating the derivatives to zero
  - expressions: $$\$$

------

------

## Chapter 2: Learning Modelling - A Maximum Likelihood Approach

------

### 2.1 Errors As Noise

------

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

​	$$P(Y=y, X=x) = P(Y=y) \times P(X=x)$$

​	$$P(y_1, y_2, ..., y_J) = \prod^J_{j=1}{P(y_j)}$$

​	$$P(Y=y, X=x) = P(Y=y |X=x) \times P(X=x) = P(X=x|Y=y) \times P(Y=y)$$

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

------

### 2.3Popular Discrete Distributions

#### 2.3.1 Bernoulli distribution

#### 2.3.2 Binomial distribution

#### 2.3.3 Multinomial distribution

------

### 2.4 Continuous Random Variables - Density Function



------

### 2.5 Popular Continuous Density Function

#### 2.5.1 The uniform density function

#### 2.5.2 The beta density function

#### 2.5.3 The Gaussian density function

#### 2.5.4 Multivariate Gaussian

------

### 2.7 Thinking Generatively... Continued

------

### 2.8 Likelihood

#### 2.8.1 Dataset likelihood

#### 2.8.2 Maximum likelihood

#### 2.8.3 Characteristics of the maximum likelihood solution

#### 2.8.4 Maximum likelihood favours complex models

------

### 2.9 The Bias-Variance Trade-Off

------

### 2.10 Effect of Noise on Parameter Estimates

#### 2.10.1 Uncertainty in estimates

#### 2.10.2 Comparison with empirical values

#### 2.10.3 Variability in model parameters - Olympic data

------

### 2.11 Variability in Predictions

####  

------

2.11.1 Predictive variability - an example

#### 2.11.2 Expected values of the estimators

------

------

##Chapter 3: Bayesian Approach

### 3.1 A Coin Game

#### 3.1.1 Counting heads

#### 3.1.2 The Bayesian way

------

### 3.2 The Exact Posterior

------

### 3.3 The Three Scenarios

#### 3.3.1 No prior knowledge

#### 3.3.2 The fair coin scenario

#### 3.3.3 A biased coin

#### 3.3.4 The three scenarios - a summary

#### 3.3.5 Adding more data

------

### 3.4 Marginal Likelihoods

#### 3.4.1 Model comparison with the marginal likelihood

------

### 3.5 Hyperparameters

------

### 3.6 Graphical Models

------

### 3.8 A Bayesian Treatment of The Olympic 100m Data

#### 3.8.1 The model

#### 3.8.2 The likelihood

#### 3.8.3 The prior

#### 3.8.4 The posterior

#### 3.8.5 A first-order polynomial

### 3.8.6 Making predictions

------

### 3.9 Marginal Likelihood for Polynomial Model or Der Selection

------

------

##Chapter 4: Bayesian Inference

### 4.1 Non-Conjugate Models

------

### 4.2 Binary Responses

------

### 4.3 A Point Estimate - The Map Solution

------

### 4.4 The Laplace Approximation

#### 4.4.1 Laplace approximation example: Approximating a gamma density

#### 4.4.2 Laplace approximation for the binary response model

------

### 4.5 Sampling Techniques

#### 4.5.1 Playing darts

#### 4.5.2 The Metropolis-Hastings algorithm

#### 4.5.3 The art of sampling

------

------

##Chapter 5: Classification

### 5.1 The General Problem

------

### 5.2 Probabilistic Classifiers

#### 5.2.1 The Bayes classifier

##### 5.2.1.1 Likelihood - class - conditional distributions

##### 5.2.1.2 Prior class distribution

##### 5.2.1.3 Example - Gaussian class-conditionals

##### 5.2.1.4 Making predictions

##### 5.2.1.5 The naive-Bayes assumption

##### 5.2.1.6 Example - classifying text

##### 5.2.1.7 Smoothing

#### 5.2.2 Logistic regression

##### 5.2.2.1 Motivation

##### 5.2.2.2 Non-linear decision functions

##### 5.2.2.3 Non-parametric models - the Gaussian process

------

### 5.3 Non-Probabilistic Classifiers

#### 5.3.1 K-nearest neighbours

#### 5.3.2 Support vector machines and other kernel methods

##### 5.3.2.1 The margin

##### 5.3.2.2 Maximising the margin

##### 5.3.2.3 Making predictions

##### 5.3.2.4 Support vectors

##### 5.3.2.5 Soft margins

##### 5.3.2.6 Kernels

------



------

------

##Chapter 6: Clustering



------

------

##Chapter 7: Principal Components Analysis and Latent Variable Models

------

------

## Chapter 8: Gaussian Processes

------

------

## Chapter 9: Markov Chain Monte Carlo Sampling

------

------

## Chapter 10: Advanced Mixture Modeling

------

------

## Chapter 10: Advanced Mixture Modeling

------

------

