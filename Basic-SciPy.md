# Basic SciPy

## Package

### [numpy](https://docs.scipy.org/doc/numpy/reference/index.html)

#### About Array

1. [Array Generation](https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html)
   - import data with `genfromtxt` (I/O)
   - ones and zeros: `zeros`
   - numerical ranges: `range`, `linspace`, `logspace`
   - from existing data: 
   - record arrays `numpy.rec`:
   - character arrays `numpy.char`:
   - building matrices: 
2. [Array Manipulation](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)
   - copy: `copyto`
   - changing shape: `reshape`
   - transpose-like operations: `.T`
   - splitting arrays: 
   - adding and removing elements: `delete`, `append`, `unique`
   - rearranging elements: `fliplr`, `flipud`
   - changing number of dimensions: 
   - changing kind of array: `asscalar`
   - joining array:
   - tiling arrays
3. Setting
   - `np.set_printoptions()`
     - parameters: precision=int, suppress=Boolean

#### About Output

1. print
   - 