import numpy as np
import sys
sys.path.append("../ex06")
sys.path.append("../../module00/ex02")
import TinyStatistician as TinyStatistician

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score, standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    mean = float(sum(x) / len(x))
    var = ((sum((x - mean) ** 2)) / len (x)) ** 0.5
    return ((x - mean) / var)
    

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))

    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(zscore(Y))