import numpy as np
import scipy.io as io


mat = io.loadmat(r'C:\Users\Ray\Desktop\MTT\matlab.mat')
truePos = mat['truePos'][0].tolist()
measPos = mat['measPos'][0].tolist()
measCov = mat['measCov'][0].tolist()
time = mat['time']
T = np.mean(np.diff(time))
print(T)