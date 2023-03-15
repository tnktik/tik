import numpy as np
import math
import matplotlib.pyplot as plt
class ALD:
    def random_density_function(x,mean,b):
        return (1/(2*b))*math.exp(-(abs(x-mean)/b))
    def cumulative_distribution_function(x,mean,b):
        if x < mean:
            return (1/2)*math.exp((x-mean)/b)
        if x >= mean:
            return 1-(1/2)*math.exp(-(x-mean)/b)
        

x=np.arange(-3,3,0.1)
y=[]
for i in range(len(x)):
    y=np.append(y,ALD.random_density_function(x[i],0,1))
plt.plot(x,y)
plt.show()
y=[]
for i in range(len(x)):
    y=np.append(y,ALD.cumulative_distribution_function(x[i],0,1))
plt.plot(x,y)
plt.show()

