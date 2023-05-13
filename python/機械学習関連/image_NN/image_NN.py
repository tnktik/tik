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
wakariyasuidata=np.array([[[0,5],[1,1],[5,0]],[[6,2],[2,6],[2,2]]])
wakariyasuiteacher=np.array([[1,0],[0,1]])
#ハイパーパラメータ
learning_rate=0.5

#活性化関数
class activate_function():
    class ReLU():
        #関数
        def func(x):
            if x>0:
                return x
            if x<=0:
                return 0
        #微分
        def dif(x):
            if x>0:
                return 1
            if x<=0:
                return 0
    #ソフトマックス関数 
    class Softmax():
        def func(x,list_x):
            return np.exp(x)/np.sum(np.exp(list_x))
        def dif(x):
            return activate_function.Softmax.func(x)*(1-activate_function.Softmax.func(x))
    #シグモイド関数
    class Sigmoid():
        def func(x):
            return 1/(1+np.exp(-x))
        #本当の微分の結果
        def dif(x):
            return activate_function.Sigmoid.func(x)*(1-activate_function.Sigmoid.func(x))
class Layer():
    #全結合
    class FC():
        #FC(出力データ,活性化関数)
        def __init__(self,x,output_size,activate_func=activate_function.ReLU):
            self.x=x.ravel()
            self.weight=np.random.rand(output_size,len(self.x))
            self.bias_w=np.random.rand(output_size)
            self.output_size=output_size
            self.activate_func=activate_func
            self.h=np.zeros(self.output_size)
            self.output=np.zeros(output_size)
            self.error=np.zeros(self.x.shape)
            self.out_error=np.zeros(output_size)
        #出力後のデータの形を返す
        def output_data_size(self):
            return self.output
        #重みを返す
        def output_weight(self):
            return self.weight
        def output_bias_w(self):
            return self.bias_w
        #誤差データを出力
        def output_error(self):
            return self.error
        def set_weight(self,w,bw):
            self.weight=w
            self.bias_w=bw
        #前方方向
        def forward(self,x):
            self.x=x.ravel()
            for i in range(self.output_size):
                #内積
                h=np.dot(x.ravel(),self.weight[i])
                self.h[i]=h+self.bias_w[i]
            if self.activate_func==activate_function.Softmax:
                for i in range(self.output_size):
                    #活性化関数にバイアスとセットで
                    y=self.activate_func.func(self.h[i],self.h)
                    #出力を保存
                    self.output[i]=y
            else:
                #出力のサイズだけループ
                for i in range(self.output_size):
                    #活性化関数にバイアスとセットで
                    y=self.activate_func.func(self.h[i])
                    #出力を保存
                    self.output[i]=y
            return self.output
        #出力層の逆伝搬につかう
        def first_backward(self,teacher_vector):
            #この層の出力サイズ

            for i in range(self.output_size):
                #誤差を保存
                self.out_error[i]=(self.output[i]-teacher_vector[i])*self.activate_func.dif(self.h[i])
                #重みを更新
                self.weight[i]=self.weight[i]-learning_rate*self.out_error[i]*self.x
                #バイアスの重みを更新
                self.bias_w[i]=self.bias_w[i]-learning_rate*self.out_error[i]
            er=np.zeros(self.x.shape)
            #入力データのサイズ(1次元)
            for j in range(len(self.x)):
                #出力のユニットをみる
                for k in range(self.output_size):
                    er[j]=er[j]+self.out_error[k]*self.weight[k][j]
                self.error[j]=er[j]
            return self.weight
        #中間層の逆伝搬に使う
        def backward(self,forward_error):              
            #以下で重みを更新していく
            for i in range(self.output_size):
                forward_error[i]=forward_error[i]*self.activate_func.dif(self.h[i])
                self.weight[i]=self.weight[i]-learning_rate*forward_error[i]*self.x
                self.bias_w[i]=self.bias_w[i]-learning_rate*forward_error[i]
            er=np.zeros(self.x.shape)
                #この層の誤差を見ていく
            for j in range(len(self.x)):
                #出力のユニットをみる
                for k in range(self.output_size):
                    er[j]=er[j]+self.error[k]*self.weight[k][j]
                self.error[j]=er[j]
            return self.weight            
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
            self.x=x
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
                self.error=np.zeros((channel,self.output_H,self.output_W))
                self.st=st
            else:
                print("色々おかしいCNN")
        #出力後のデータの形を返す
        def output_data_size(self):
            return self.output
        #重みを返す
        def output_weight(self):
            return self.weight
        def output_bias_w(self):
            return self.bias_w
        #誤差データを出力
        def output_error(self):
            return self.error
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
                            y=self.activate_func.func(h+self.bias_w[int(a/self.st)][int(b/self.st)])
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
        def backward(self):
            a=0
    #プーリング層(kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
    class Pooling():
        def __init__(self,x,kernel_size_H,kernel_size_W,st=1,mode="max"):
            while(x.ndim<3):
                x=x[np.newaxis]
            self.x=x
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((x.shape[-2]-kernel_size_H)/st) and is_integer_num((x.shape[-1]-kernel_size_W)/st):
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(x.shape[-2]-kernel_size_H)/st)
                self.output_W=int(1+(x.shape[-1]-kernel_size_W)/st)
                self.kernel_size_H=kernel_size_H
                self.kernel_size_W=kernel_size_W
                self.weight=np.array([])
                #とりあえず出力結果を作成
                self.output=np.zeros((x.shape[-3],self.output_H,self.output_W))
                self.st=st
                self.mode=mode

            else:
                print("色々おかしいpooling")   
        #出力後のデータの形を返す
        def output_data_size(self):
            return self.output  
        #重みを返す
        def output_weight(self):
            return self.weight
        #誤差データを出力
        def output_error(self):
            return self.error
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
                            #カーネルの数だけ処理していく
                            for i in range(self.kernel_size_H):
                                for j in range(self.kernel_size_W):
                                    if self.mode=="max":
                                        self.weight=np.zeros((x.shape[-3],x.shape[-2],x.shape[-1]))
                                        #とりあえず最初のユニットを最大にしておく
                                        if i==0 and j==0:
                                            value=x[d][i+a][j+b]
                                            self.weight[d][i+a][j+b]=1
                                            #カーネル内の他のユニットを見ていく。
                                        else:
                                            #他に最大の物が見つかった時                                            
                                            if value<x[d][i+a][j+b]:
                                                value=x[d][i+a][j+b]
                                                #誤差逆伝搬の為にそのカーネル内にある重みを一旦0に書き換えて、新しく1を置く
                                                for k in range(self.kernel_size_H):
                                                    for l in range(self.kernel_size_W):
                                                        self.weight[d][k+a][l+b]=0
                                                self.weight[d][i+a][j+b]=1
                                    #平均プーリング
                                    if self.mode=="average":
                                        value=value+x[d][i+a][j+b]
                                        self.weight[d][i+a][j+b]=1/(self.kernel_size_H*self.kernel_size_W)
                                if self.mode=="average":
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
        def backward(self,forward_error):
                #前方向の誤差が1次元配列とかなら3次元配列に変える。

                #xのchannelの数だけ処理
                for d in range(self.weight.shape[-3]):
                    #convはたたみ込みをまだするかどうか
                    pool=True
                    #xの何処を処理するかを決めるためのaとb
                    a=0
                    b=0
                    while(pool):
                        value=0
                        #カーネルの数だけ処理していく
                        for i in range(self.kernel_size_H):
                            for j in range(self.kernel_size_W):
                                #forwardで求めた重みで前方向の誤差をかける。
                                self.error=self.weight[d][i+a][j+b]*forward_error[d][a][b]
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
                return self.error
def forward(x):
    x=np.pad(x,((1,0),(1,0)))
    #畳み込みニューラルネットワーク(kernelの数,kernelの高さ,kernelの横,stride,活性化関数)
    NN1=Layer.CNN(x,4,2,2,1)
    y=NN1.output_data_size()
    #プーリング層(kernelの高さ,kernelの横,stride,"max" or "average"のどちらかのモード)
    NN2=Layer.Pooling(y,2,2,2)
    y=NN2.output_data_size()
    #FC(出力データ,活性化関数)
    NN3=Layer.FC(y,10,activate_function.Softmax)
    x=NN1.forward(x)
    x=NN2.forward(x)
    NN3.Dropout(5)
    x=NN3.forward(x)
    t=np.zeros(10)
    t[0]=1
    NN3.first_backward(t)
    er=NN3.output_error()
    w=NN3.output_weight()
    print(er)
    a=NN2.backward(er)

#forward(a)
def machine_learning(output_data,class_size,teacher_index):
    t=np.zeros(class_size)
    t[teacher_index]=1
    z=np.linalg.norm(t-output_data,ord=2)

def wakariyasui_learning():
    count=0
    epoch=1
    max_epoch=2000
    #FC(出力データ,活性化関数)
    NN1=Layer.FC(wakariyasuidata[0][0],2,activate_function.Sigmoid)
    
    y=NN1.output_data_size()
    #FC(出力データ,活性化関数)
    NN2=Layer.FC(y,2,activate_function.Sigmoid)
    #NN1.set_weight(np.array([[-1.1,1.1],[-2.1,-1.1]]),np.array([1.1,2.1]))
    #NN2.set_weight(np.array([[-1.6,1.6]]),np.array([2.3]))

    while(count<5 and epoch<=max_epoch):
        for i in range(len(wakariyasuidata)):
        #for i in range(1):
            for j in range(len(wakariyasuidata[i])):
            #for j in range(1):
                x=NN1.forward(wakariyasuidata[i][j])
                x=NN2.forward(x)
                if np.argmax(x)==i:
                    count+=1
                    if count==len(wakariyasuidata.ravel()):
                        break
                else:
                    count=0
                NN2.first_backward(wakariyasuiteacher[i])
                #w=NN2.output_weight()
                er=NN2.output_error()
                #bw=NN2.output_bias_w()
                NN1.backward(er)
                #w=NN1.output_weight()
                #bw=NN1.output_bias_w()

            if count==len(wakariyasuidata.ravel()):
                break
        epoch+=1
        
    print(epoch)
    w=NN2.output_weight()
    print(w)
    w=NN1.output_weight()
    print(w)


wakariyasui_learning()
