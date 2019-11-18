import numpy as np

def load_linear_example1():
    """
    >>> X,Y = load_linear_example1()
    >>> X
    array([[ 1,  4],
           [ 1,  8],
           [ 1, 13],
           [ 1, 17]])
    >>> X[0]
    array([1, 4])
    >>> Y
    array([ 7, 10, 11, 14])
    """
    X = np.array([[1,4],[1,8],[1,13],[1,17]])
    Y = np.array([7,10,11,14])
    return X,Y

def load_nonlinear_example1():

    """
    >>> X,Y = load_nonlinear_example1()
    >>> X[0]
    array([1., 0.])
    """
    X = np.array([[1,0.0],[1,2.0],[1,3.9],[1,4.0]])
    Y = np.array([4.0,0.0,3.0,2.0])
    return X,Y

def polynomial2_features(input):
    """
    >>> X,Y = load_nonlinear_example1()
    >>> polynomial2_features(X)
    array([[ 1.  ,  0.  ,  0.  ],
           [ 1.  ,  2.  ,  4.  ],
           [ 1.  ,  3.9 , 15.21],
           [ 1.  ,  4.  , 16.  ]])
    >>> Y
    array([4., 0., 3., 2.])

    """
    poly2 = input[:,1:]**2
    return np.c_[input,poly2]

def polynomial3_features(input):
    """
    >>> X,Y = load_nonlinear_example1()
    >>> polynomial3_features(X)
    array([[ 1.   ,  0.   ,  0.   ,  0.   ],
           [ 1.   ,  2.   ,  4.   ,  8.   ],
           [ 1.   ,  3.9  , 15.21 , 59.319],
           [ 1.   ,  4.   , 16.   , 64.   ]])
    >>> Y
    array([4., 0., 3., 2.])
    """
    poly2 = input[:,1:]**2
    poly3 = input[:,1:]**3
    return np.c_[input, poly2, poly3]


