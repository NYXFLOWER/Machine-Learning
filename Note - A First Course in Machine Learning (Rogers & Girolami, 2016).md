# Note - A First Course in Machine Learning (Rogers & Girolami, 2016)

## Chapter 1: Learning Modelling - A least Squares Approach

### Linear Modelling

key words: attribute, response, model, linear relationship, function, parameter, intercept, gradient, squared loss function, average loss, argmin, analytical solution, partial derivative.

summary:

- Create a linear model that encapsulates the relationship between a set of attributes and a set of responses.

  - ```text
    ![](http://latex.codecogs.com/gif.latex?f(x_n; w_0, w_1) = w_0 + w_1x)
    ```
- Define a loss function to fit / learn this model, as a way of objectively identifying how good a particular model was.
  - squared loss function: $ \mathcal{L_n}(t_n, f(x_n; w_0, w_1)) = (t_n - f(x_n; w_0, w_1))^2$
  - average loss across the whole dataset: $\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \mathcal{L_n}(t_n, f(x_n; w_0, w_1))$
- Derive exact expressions for the values of the model parameters that minimised the loss and therefore corresponded to the best function, using the squared loss.
  - argmin: find the argument that minimises ...
  - Method of turning points:
    - differentiating the loss function
    - equating the derivatives to zero
  - expressions: $$\$$

## Chapter 2: Learning Modelling - A Maximum Likelihood Approach

## Chapter 3: Bayesian Approach

## Chapter 4: Bayesian Inference

## Chapter 5: Classification

## Chapter 6: Clustering

## Chapter 7: Principal Components Analysis and Latent Variable Models

## Chapter 8: Gaussian Processes

## Chapter 9: Markov Chain Monte Carlo Sampling

## Chapter 10: Advanced Mixture Modeling