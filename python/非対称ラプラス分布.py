import numpy as np
import math
import matplotlib.pyplot as plt
#非対称ラプラス分布
class ALD:
    #確率密度関数を定義
    def random_density_function(x,mean,b):
        return (1/(2*b))*math.exp(-(abs(x-mean)/b))
    #累積分布関数を定義
    def cumulative_distribution_function(x,mean,b):
        if x < mean:
            return (1/2)*math.exp((x-mean)/b)
        if x >= mean:
            return 1-(1/2)*math.exp(-(x-mean)/b)
    #ALDの関数のグラフを表示する。引数は(関数名,xに入れる配列,mean,b)
    def show(function,x,mean,b):
        #pltの縦の情報が欲しいので空の配列を作成
        y=[]
        #yに確率密度関数のx[i]を代入していく操作
        for i in range(len(x)):
            #mean is 0,b is 1
            y=np.append(y,function(x[i],mean,b))
        plt.plot(x,y)
        plt.show()

#pltでplotするためにxに-3から3の0.1区切りで配列作成
x=np.arange(-3,3,0.1)
ALD.show(ALD.random_density_function,x,2,1)
