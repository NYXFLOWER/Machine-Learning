# QR Decomposition

The orthonormal matrix $\mathbf{Q}$ has the property $$\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$$

Matrix $\mathbf{R}$ is an upper triangular matrix, such that matrix $\mathbf{X} = \mathbf{Q} \mathbf{R}$

```python
Q, R = np.linalg.qr(X)
```

When we solve equations $\mathbf{R} \mathbf{w} = \mathbf{Q}^\top \mathbf{Y}$

```python
w = scipy.linalg.solve_triangular(R, numpy.dot(Q.T, y))
```



