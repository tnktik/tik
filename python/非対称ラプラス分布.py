import numpy as np
import random
import math
import matplotlib.pyplot as plt
#非対称ラプラス分布
class ALD:
    x=np.arange(-3,3,0.1)
    #確率密度関数を定義
    def random_density_function(x,mean,b):
        return (1/(2*b))*math.exp(-(abs(x-mean)/b))
    #累積分布関数を定義
    def cumulative_distribution_function(x,mean,b):
        if x < mean:
            return (1/2)*math.exp((x-mean)/b)
        if x >= mean:
            return 1-(1/2)*math.exp(-(x-mean)/b)
    #ALDの関数のグラフを表示する。引数は(関数名,mean,b)
    def show(function,mean,b):
        #pltの縦の情報が欲しいので空の配列を作成
        y=[]
        #yに確率密度関数のx[i]を代入していく操作
        for i in range(len(ALD.x)):
            #mean is 0,b is 1
            y=np.append(y,function(ALD.x[i],mean,b))
        plt.plot(ALD.x,y)
        plt.show()
    #ALDのランダムウォークを作る
    def random_walk(n,mean,b):
        #評価を0と置いておく
        S=0
        #n回試行を繰り返す
        for i in range(n):
            #ALD.xの範囲からランダムに数値をとる。
            x=random.uniform(min(ALD.x),max(ALD.x))
            #xの値が0以上なら確率密度関数に正をかける
            if(x>=0):
                y=ALD.random_density_function(x,mean,b)
                S+=y
            #xの値が0未満なら確率密度関数に負をかける
            else:
                y=ALD.random_density_function(x,mean,b)*-1
                S+=y
        return S
#pltでplotするためにxに-3から3の0.1区切りで配列作成

ALD.show(ALD.random_density_function,0,1)
print(ALD.random_walk(50,0,1))