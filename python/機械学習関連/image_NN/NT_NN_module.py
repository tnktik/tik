import numpy as np
import random

#整数判定
def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

#活性化関数
class activate_function():
    class ReLU():
        #関数
        def func(x):
            y=np.array([])
            for i in range(len(x)):
                if x[i]>0:
                    y=np.append(y,x[i])
                if x[i]<=0:
                    y=np.append(y,0)
            return y
        #微分
        def dif(x):
            y=np.array([])
            for i in range(len(x)):
                if x[i]>0:
                    y=np.append(y,1)
                if x[i]<=0:
                    y=np.append(y,0)
            return y
    #ソフトマックス関数 
    class Softmax():
        def func(x):
            max_x=max(x)
            return np.exp(x-max_x)/np.sum(np.exp(x-max_x))
        def dif(x):
            return activate_function.Softmax.func(x)*(1-activate_function.Softmax.func(x))
    #シグモイド関数
    class Sigmoid():
        def func(x):
            return np.array(1/(1+np.exp(-x)))
        #本当の微分の結果
        def dif(x):
            return np.array(activate_function.Sigmoid.func(x)*(1-activate_function.Sigmoid.func(x)))
    class Linear():
        def func(x):
            return x
        def dif(x):
            y=np.array([])
            for i in range(len(x)):
                y=np.append(y,1)
            return y
class Layer():
    #全結合
    class FC():
        #FC(出力データ,活性化関数)
        def __init__(self,x,output_size,activate_func=activate_function.ReLU,learning_rate=0.01):
            self.x=x.ravel()
            self.weight=np.random.normal(0.0,len(self.x)**-0.5,(output_size,len(self.x)))
            self.bias_w=np.random.normal(0.0,len(self.x)**-0.5,(output_size))
            self.output_size=output_size
            self.activate_func=activate_func
            self.learning_rate=learning_rate
            self.h=np.zeros(self.output_size)
            self.output=np.zeros(output_size)
            self.error=np.zeros(self.x.shape)
            self.out_error=np.zeros(output_size)
        #出力後のデータの形を返す
        def output_data(self):
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
            #データを１次元配列にする
            self.x=np.array(x.ravel())
            #重みとデータの内積
            self.h=(self.x@self.weight.T)+self.bias_w
            #上の結果を活性化関数に入れる\\
            self.output=self.activate_func.func(self.h)
            return self.output
        #出力層の逆伝搬につかう
        def first_backward(self,teacher_vector):
            loss = np.mean((teacher_vector-self.out_error)**2)
            #print(loss)
            #誤差を保存
            self.out_error=(self.output-teacher_vector)*self.activate_func.dif(self.h)
            #誤差と重みの計算
            self.error=self.out_error@self.weight
            #重みを更新
            self.weight-=self.learning_rate*(np.array([self.out_error]).T@np.array([self.x]))
            #バイアスの重みを更新
            self.bias_w-=self.learning_rate*self.out_error   
            return self.error
        #中間層の逆伝搬に使う
        def backward(self,forward_error):
            forward_error=forward_error*self.activate_func.dif(self.h) 
            #次の層のために誤差を求める
            self.error=forward_error@self.weight    
            #以下で重みを更新していく
            self.weight-=self.learning_rate*(np.array([forward_error]).T@np.array([self.x]))
            self.bias_w-=self.learning_rate*forward_error
            return self.error         
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
        def __init__(self,x,channel,kernel_size_H,kernel_size_W,st=1,activate_func=activate_function.ReLU,learning_rate=0.01):
            while(x.ndim<3):
                x=x[np.newaxis]
            self.x=x
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((self.x.shape[-2]-kernel_size_H)/st) and is_integer_num((self.x.shape[-1]-kernel_size_W)/st):

                self.channel=channel
                self.kernel_size_H=kernel_size_H
                self.kernel_size_W=kernel_size_W
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(self.x.shape[-2]-kernel_size_H)/st)
                self.output_W=int(1+(self.x.shape[-1]-kernel_size_W)/st)
                #重み作成
                self.weight=np.random.rand(channel,self.x.shape[-3],kernel_size_H*kernel_size_W)
                #バイアスの重みを作成
                self.bias_w=np.random.rand(channel,self.output_H*self.output_W)
                #活性化関数
                self.activate_func=activate_func
                self.learning_rate=learning_rate

                #とりあえず出力結果を作成
                self.output=np.zeros((channel,self.output_H,self.output_W))
                self.error=np.zeros(self.x.shape)
                self.st=st
                self.gen_mat=np.zeros(self.x.shape[-3]*self.output_H*self.output_W*self.kernel_size_W*self.kernel_size_H).reshape(self.output_W*self.output_H,self.kernel_size_W*self.kernel_size_H,self.x.shape[-3])
                self.h=np.zeros((channel,self.output_H,self.output_W))
                self.y=np.zeros((channel,self.output_H,self.output_W))
            else:
                print("色々おかしいCNN")
        #出力後のデータの形を返す
        def output_data(self):
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
        #前方方向の計算    
        def forward(self,x):
            #入力データを３次元にする
            while(x.ndim<3):
                x=x[np.newaxis]
            #畳み込みをするために入力データの形を変えていく
            #重みとgen_matの内積をhhとする
            hh=np.zeros(self.channel*self.output_H*self.output_W*self.x.shape[-3]).reshape(self.channel,self.output_H*self.output_W,self.x.shape[-3])
            for d in range(self.x.shape[-3]):
                for h in range(self.output_H):
                    for w in range(self.output_W):
                        for i in range(self.kernel_size_H):
                            for j in range(self.kernel_size_W):
                                self.gen_mat[w+h*self.output_W][i*self.kernel_size_W+j][d]=x[d][i+h*self.st][j+w*self.st]
            #gen_matと重みの内積
            for i in range(self.channel):
                for j in range(self.output_H*self.output_W):
                    for k in range(self.x.shape[-3]):
                        hh[i][j][k]=(self.weight[i][k]@self.gen_mat[j][:,k])+self.bias_w[i][k]
            #hhのデータの１次元データをまとめる(入力データのチャネルをまとめる)
            bb=np.ones((1,self.x.shape[-3]))
            self.h=hh@bb.T
            #hhのデータを１次元にして活性化関数に入れる。
            self.y=self.activate_func.func(self.h.ravel())
            #データを出力特徴マップの形に変える
            self.output=np.transpose(self.y.reshape(self.channel,self.output_H,self.output_W),(0,1,2))
            return self.output
        def backward(self,forward_error):
            #一旦、、一次元に変える
            #活性化関数と受け取った誤差をかけて適切な誤差にかえる
            #1次元配列なので3次元配列に変える。
            forward_error=(forward_error.ravel()*self.activate_func.dif(self.h.ravel())).reshape(self.output.shape)
            #次の層に渡す誤差の0行列
            self.error=np.zeros(self.x.shape)
            for i in range(self.channel):
                for j in range(self.x.shape[-3]):
                    for k in range(self.output_H):
                        for l in range(self.output_W):
                            print(l+k*self.output_W)
                            print("weight")
                            print(self.weight)
                            print(self.weight[i][j])
                            print("forwarder")
                            print(self.output)
                            print(forward_error)
                            print(forward_error[i][k][l])
                            print("gen_mat")
                            print(self.gen_mat)
                            print(self.gen_mat[:][j][:,l+k*self.output_W])
                            print("inner")
                            print(np.array([forward_error[i][k][l]]).T@np.array([self.gen_mat[:][j][:,l+k*self.output_W]]))
                            self.weight[i][j]=self.weight[i][j]-self.learning_rate*np.array([forward_error[i][k][l]]).T@np.array([self.gen_mat[:][j][:,l+k*self.output_W]])

            return self.error
    #プーリング層(入力データ,kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
    class Pooling():
        def __init__(self,x,kernel_size_H,kernel_size_W,l,st=1,mode="max"):
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
                self.error=np.zeros(self.x.shape)

            else:
                print("色々おかしいpooling")   
        #出力後のデータの形を返す
        def output_data(self):
            return self.output  
        #重みを返す
        def output_weight(self):
            return self.weight
        #誤差データを出力
        def output_error(self):
            return self.error
        def forward(self,x):
            while(x.ndim<3):
                x=x[np.newaxis]
            gen_mat=np.zeros(self.channel*self.output_H*self.output_W*self.kernel_size_W*self.kernel_size_H).reshape(self.channel,self.output_W*self.output_H,self.kernel_size_W*self.kernel_size_H)
            self.weight=np.zeros(self.channel*self.output_H*self.output_W*self.kernel_size_W*self.kernel_size_H).reshape(self.channel,self.output_W*self.output_H,self.kernel_size_W*self.kernel_size_H)
            for d in range(self.channel):
                for h in range(self.output_H):
                    for w in range(self.output_W):
                        min_v=np.inf()
                        min_d=d
                        min_h=w+h*self.output_H
                        min_w=0   
                        for i in range(self.kernel_size_H):
                            for j in range(self.kernel_size_W):
                                if min_v>x[d][i+h*self.st][j+w*self.st]:
                                    gen_mat[d][min_h][min_w]=0
                                    gen_mat[d][w+h*self.output_H][i*self.kernel_size_H+j]=x[d][i+h*self.st][j+w*self.st]
                                    self.weight[min_d][min_h][min_w]=0
                                    self.weight[d][w+h*self.output_H][i*self.kernel_size_H+j]=1
                                    min_d=d                               
                                    min_h=w+h*self.output_H
                                    min_w=i*self.kernel_size_H+j
            self.output=np.transpose(gen_mat.reshape(self.channel,self.x[-3],self.output_H,self.output_W),(1,0,2,3))

            while(x.ndim<3):
                x=x[np.newaxis]
            self.x=x
            #xのchannelの数だけ処理
            self.weight=np.zeros((self.x.shape[-3],self.x.shape[-2],self.x.shape[-1]))
            for d in range(self.x.shape[-3]):
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
                                #とりあえず最初のユニットを最大にしておく
                                if i==0 and j==0:
                                    value=self.x[d][i+a][j+b]
                                    self.weight[d][i+a][j+b]=1                                        
                                    #カーネル内の他のユニットを見ていく。
                                else:
                                    #他に最大の物が見つかった時                                            
                                    if value<self.x[d][i+a][j+b]:
                                        value=self.x[d][i+a][j+b]
                                        #誤差逆伝搬の為にそのカーネル内にある重みを一旦0に書き換えて、新しく1を置く
                                        for k in range(self.kernel_size_H):
                                            for l in range(self.kernel_size_W):
                                                self.weight[d][k+a][l+b]=0
                                        self.weight[d][i+a][j+b]=1
                            #平均プーリング
                            if self.mode=="average":
                                value=value+self.x[d][i+a][j+b]
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
                forward_error=forward_error.reshape(self.output.shape)
                #xのchannelの数だけ処理
                for d in range(self.error.shape[-3]):
                    #convはたたみ込みをまだするかどうか
                    pool=True
                    #xの何処を処理するかを決めるためのaとb
                    a=0
                    b=0
                    while(pool):
                        #カーネルの数だけ処理していく
                        for i in range(self.kernel_size_H):
                            for j in range(self.kernel_size_W):
                                #forwardで求めた重みで前方向の誤差をかける。
                                self.error[d][i+a][j+b]=self.weight[d][i+a][j+b]*forward_error[d][int(a/self.st)][int(b/self.st)]
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

