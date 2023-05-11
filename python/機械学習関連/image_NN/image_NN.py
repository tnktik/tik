import numpy as np
import random
#整数判定
def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

#最初の入力データ
a=np.random.rand(6,6)
asd=np.array([[[1,2,3,1],[4,5,6,1],[7,8,9,1]]])

#活性化関数
class activate_function():
    def ReLU(x):
        if x>0:
            return x
        if x<=0:
            return 0

class Layer():
    #全結合
    class FC():
        #FC(出力データ,活性化関数)
        def __init__(self,x,output_size,activate_func=activate_function.ReLU):
            self.weight=np.random.rand(output_size,len(x.ravel()))
            self.bias_w=np.random.rand(1)
            self.output_size=output_size
            self.activate_func=activate_func
            self.output=np.array([])
        #出力後のデータの形を返す
        def output_data_size(self):
            return self.output_size
        #前方方向
        def forward(self,x):
            for i in range(self.output_size):
                h=np.dot(x.ravel(),self.weight[i])
                y=self.activate_func(h+self.bias_w)
                self.output=np.append(self.output,y)
            return self.output
        #確率pは0～100のドロップアウト
        def Dropout(self,p):
            if p>=0 and p<100:
                for i in range(len(self.weight)):
                    for j in range(len(self.weight[i])):
                        a=random.random()*100
                        if a<=p:
                            self.weight[i][j]=0


    class CNN():
        #畳み込みニューラルネットワーク(kernelの数,kernelの高さ,kernelの横,stride,活性化関数)
        def __init__(self,x,channel,kernel_size_H,kernel_size_W,st=1,activate_func=activate_function.ReLU):
            while(x.ndim<3):
                x=x[np.newaxis]
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((x.shape[-2]-kernel_size_H)/st) and is_integer_num((x.shape[-1]-kernel_size_W)/st):
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(x.shape[-2]-kernel_size_H)/st)
                self.output_W=int(1+(x.shape[-1]-kernel_size_W)/st)
                #重み作成
                self.weight=np.random.rand(channel,x.shape[-3],kernel_size_H,kernel_size_W)
                #バイアスの重みを作成
                self.bias_w=np.random.rand(self.output_H,self.output_W)
                #活性化関数
                self.activate_func=activate_func
                #とりあえず出力結果を作成
                self.output=np.zeros((channel,self.output_H,self.output_W))
                self.st=st
            else:
                print("色々おかしいCNN")
        #出力後のデータの形を返す
        def output_data_size(self):
            return self.output
        #前方方向の計算    
        def forward(self,x):
            while(x.ndim<3):
                x=x[np.newaxis]
            #プログラミングの挙動の為にxの次元数を見てる
            if x.ndim==3:
                #xのchannelの数だけ処理
                for d in range(x.shape[-3]):
                    #出力のchannelの数だけ処理
                    for c in range(self.weight.shape[-4]):
                        #convはたたみ込みをまだするかどうか
                        conv=True
                        #xの何処を処理するかを決めるためのaとb
                        a=0
                        b=0
                        while(conv):
                            #hはたたみ込みの値
                            h=0
                            #カーネルの数だけ処理していく
                            for i in range(self.weight.shape[-2]):
                                for j in range(self.weight.shape[-1]):
                                    #カーネルの中の重みとデータをかけ算
                                    h=h+x[d][i+a][j+b]*self.weight[c][d][i][j]
                            #活性化関数を通す
                            y=self.activate_func(h+self.bias_w[int(a/self.st)][int(b/self.st)])
                            #出力結果に保存
                            self.output[c][int(a/self.st)][int(b/self.st)]=y
                            #データの位置を動かしていく
                            a+=self.st
                            #aが出力結果の形を超える時にbを動かしてaを0にする
                            if a==self.output_H*self.st:
                                b+=self.st
                                a=0
                                #データがそのchannelで全て畳み込みしたら畳み込みを止める。
                                if b==self.output_W*self.st:
                                    a=0
                                    b=0
                                    conv=False
                return self.output
    #プーリング層(kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
    class Pooling():
        def __init__(self,x,kernel_size_H,kernel_size_W,st=1,mode="max"):
            while(x.ndim<3):
                x=x[np.newaxis]
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((x.shape[-2]-kernel_size_H)/st) and is_integer_num((x.shape[-1]-kernel_size_W)/st):
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(x.shape[-2]-kernel_size_H)/st)
                self.output_W=int(1+(x.shape[-1]-kernel_size_W)/st)
                self.kernel_size_H=kernel_size_H
                self.kernel_size_W=kernel_size_W
                #とりあえず出力結果を作成
                self.output=np.zeros((x.shape[-3],self.output_H,self.output_W))
                self.st=st
                self.mode=mode
            else:
                print("色々おかしいpooling")   
        #出力後のデータの形を返す
        def output_data_size(self):
            return self.output         
        def forward(self,x):
                #プログラミングの挙動の為にxの次元数を見てる
                if x.ndim==3:
                    #xのchannelの数だけ処理
                    for d in range(x.shape[-3]):
                        #convはたたみ込みをまだするかどうか
                        pool=True
                        #xの何処を処理するかを決めるためのaとb
                        a=0
                        b=0
                        while(pool):
                            value=0
                            #hはたたみ込みの値
                            #カーネルの数だけ処理していく
                            for i in range(self.kernel_size_H):
                                for j in range(self.kernel_size_W):
                                    if self.mode=="max":
                                        if i==0 and j==0:
                                            value=x[d][i+a][j+b]
                                        else:
                                            if value<x[d][i+a][j+b]:
                                                value=x[d][i+a][j+b]
                                    if self.mode=="avarage":
                                        value=value+x[d][i+a][j+b]
                                if self.mode=="avarage":
                                    value=value/(self.kernel_size_H*self.kernel_size_W)
                                #出力結果に保存
                                self.output[d][int(a/self.st)][int(b/self.st)]=value
                            #データの位置を動かしていく
                            a+=self.st
                            #aが出力結果の形を超える時にbを動かしてaを0にする
                            if a==self.output_H*self.st:
                                b+=self.st
                                a=0
                                #データがそのchannelで全て畳み込みしたら畳み込みを止める。
                                if b==self.output_W*self.st:
                                    a=0
                                    b=0
                                    pool=False
                    return self.output
    def Softmax(x):
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))

def forward(x):

    x=np.pad(x,((1,0),(1,0)))
    #畳み込みニューラルネットワーク(kernelの数,kernelの高さ,kernelの横,stride,活性化関数)
    NN1=Layer.CNN(x,4,2,2,1)
    y=NN1.output_data_size()
    #プーリング層(kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
    NN2=Layer.Pooling(y,2,2,2)
    y=NN2.output_data_size()
    #FC(出力データ,活性化関数)
    NN3=Layer.FC(y,10)

    x=NN1.forward(x)
    x=NN2.forward(x)
    NN3.Dropout(5)
    x=NN3.forward(x)
    x=Layer.Softmax(x)
    print(x)
    print(np.argmax(x))

forward(a)