"""
Follow the PCA Example with Scikit Learn to complete this question

3a. Apply PCA to the training data from the MNIST dataset above (x_train_mnist).
Print the top 30 eigenvalues. [10 marks]
3b. Plot the cumulative variances captured by the top 30 PCs (plot 30 values in
total, e.g., the cumulative variance for the top 5 PCs is the summation of variance
captured by the top 5 PCs). Also print out the results (30 values). [10 marks]
3c. Visualise the top 10 eigenvectors as images. Describe what you can observe.
[10 marks]
3d. Use the top 10 PCs to reconstruct all the original images as x_train_mnist_approx.
Compute and print the mean squared error over all images (resulting in a single value).
Show any 10 pairs of reconstructed and original images. [15 marks]
"""
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initial the training / testing data set
mnist = tf.keras.datasets.mnist
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_mnist, x_test_mnist = x_train_mnist / 255.0, x_test_mnist / 255.0

# shape
n_train_sample = x_train_mnist.shape[0]
h, w = x_train_mnist.shape[1], x_train_mnist.shape[2]
n_feature = h * w
n_eigenvalue = 30

# reshape x_train into 2 dimension
x_train_reshape = x_train_mnist.reshape(n_train_sample, n_feature)

# --------------------------------- 3a -----------------------------------
pca = PCA(n_components=n_eigenvalue, svd_solver='randomized', whiten=True)\
    .fit(x_train_reshape)
print("The top 30 eigenvalues are \n", pca.explained_variance_)

# --------------------------------- 3b -----------------------------------
plt.figure(31)
x_axis = np.linspace(1, 30, num=30, dtype=int)
cumsum_vars = np.cumsum(pca.explained_variance_)
plt.plot(x_axis, cumsum_vars)
plt.xticks(np.arange(1, 31, step=1))
plt.title('The cumulative variance captured by the top 30 PCs')
plt.ylabel('Cumulative Variance'), plt.xlabel('The $i^{th}$ top PC')
plt.grid()
plt.show()
print("Their cumulative variances are: \n", cumsum_vars)

# --------------------------------- 3c -----------------------------------
plt.figure(32)
plt.title('The image of top ten eigenvectors')
for i in range(10):
    img = pca.components_[i].reshape((h, w))
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

# --------------------------------- 3d -----------------------------------
pca = PCA(n_components=10, svd_solver='randomized', whiten=True)\
    .fit(x_train_reshape)
x_train_mnist_approx = pca.transform(x_train_reshape)
x_reconstruction = pca.inverse_transform(x_train_mnist_approx)

# compute the mean squared error over all images
mse = (np.subtract(x_train_reshape, x_reconstruction)**2).sum() / n_train_sample/w/h
print('The mean squared error over all images is: ', mse)

# plot paired original and reconstructed images
plt.figure(33, figsize=(8, 20))
for i in range(10):
    # plot original
    plt.subplot(10, 2, 2*i+1)
    plt.title('The ' + str(i+1) + 'th original image')
    plt.xticks([])
    plt.yticks([])
    img = x_train_mnist[i]
    plt.imshow(img)

    # plot reconstructed
    plt.subplot(10, 2, 2*i+2)
    plt.title('The ' + str(i+1) + 'th reconstructed image')
    plt.xticks([])
    plt.yticks([])
    img = x_reconstruction[i].reshape((h, w))
    plt.imshow(img)
plt.show()

