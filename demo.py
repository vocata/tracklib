from tracklib.math import *
import numpy as np

f = lambda x: np.sqrt(x[0]**2 + x[1]**2)
df = num_diff_hessian([1.2, 2.3], f, 1)
print(df[:, :, 0])