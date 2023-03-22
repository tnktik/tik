import numpy as np
import matplotlib.pyplot as plt
import math

x=np.array([1,2,3])
w=np.array([4,5,6])
a=np.dot(x,w)

def sigmoid_function(x):
    return 1/(1+math.exp(-x))
h=sigmoid_function(a)
print(h,h+a)

