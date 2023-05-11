import numpy as np
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
        #FC(入力データ,出力データ,活性化関数)
        def __init__(self,input_size,output_size,activate_func=activate_function.ReLU):
            self.weight=np.random.rand(output_size,input_size)
            self.bias_w=np.random.rand(1)
            self.output_size=output_size
            self.activate_func=activate_func
        def forward(self,x):
            output=np.array([])
            for i in range(self.output_size):
                h=np.dot(x.ravel(),self.weight[i])
                y=self.activate_func(h+self.bias_w)
                output=np.append(output,y)
            return output

    class CNN():
        #畳み込みニューラルネットワーク(inputのchannel,inputの高さ,inputの横,kernelの数,kernelの高さ,kernelの横,stride,活性化関数)
        def __init__(self,input_size_channel,input_size_H,input_size_W,channel,kernel_size_H,kernel_size_W,st=1,activate_func=activate_function.ReLU):
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((input_size_H-kernel_size_H)/st) and is_integer_num((input_size_W-kernel_size_W)/st):
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(input_size_H-kernel_size_H)/st)
                self.output_W=int(1+(input_size_W-kernel_size_W)/st)
                #重み作成
                self.weight=np.random.rand(channel,input_size_channel,kernel_size_H,kernel_size_W)
                #バイアスの重みを作成
                self.bias_w=np.random.rand(self.output_H,self.output_W)
                #活性化関数
                self.activate_func=activate_func
                #とりあえず出力結果を作成
                self.output=np.zeros((channel,self.output_H,self.output_W))
                self.st=st
            else:
                print("色々おかしいCNN")
        #前方方向の計算    
        def forward(self,x):
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
                            for i in range(self.weight[-2]):
                                for j in range(self.weight[-1]):
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
            elif x.ndim==2:
                for c in range(self.weight.shape[-4]):
                        conv=True
                        a=0
                        b=0
                        while(conv):
                            h=0
                            for i in range(self.weight.shape[-2]):
                                for j in range(self.weight.shape[-1]):
                                    h=h+x[i+a][j+b]*self.weight[c][0][i][j]
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
    #プーリング層(inputのchannel,inputの高さ,inputの横,kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
    class Pooling():
        def __init__(self,input_size_channel,input_size_H,input_size_W,kernel_size_H,kernel_size_W,st=1,mode="max"):
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((input_size_H-kernel_size_H)/st) and is_integer_num((input_size_W-kernel_size_W)/st):
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(input_size_H-kernel_size_H)/st)
                self.output_W=int(1+(input_size_W-kernel_size_W)/st)
                self.kernel_size_H=kernel_size_H
                self.kernel_size_W=kernel_size_W
                #とりあえず出力結果を作成
                self.output=np.zeros((input_size_channel,self.output_H,self.output_W))
                self.st=st
                self.mode=mode
            else:
                print("色々おかしいpooling")            
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
                elif x.ndim==2:
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
                                        value=x[i+a][j+b]
                                    else:
                                        if value<x[i+a][j+b]:
                                            value=x[i+a][j+b]
                                if self.mode=="avarage":
                                    value=value+x[i+a][j+b]
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
                


a=np.pad(a,((1,0),(1,0)))
#畳み込みニューラルネットワーク(inputのchannel,inputの高さ,inputの横,kernelの数,kernelの高さ,kernelの横,stride,活性化関数)
NN1=Layer.CNN(1,7,7,4,2,2,1)
#プーリング層(inputのchannel,inputの高さ,inputの横,kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
NN2=Layer.Pooling(4,6,6,2,2,2)
#FC(入力データ,出力データ,活性化関数)
NN3=Layer.FC(4*3*3,10)

a=NN1.forward(a)
print(a)
a=NN2.forward(a)
print(a)
a=NN3.forward(a)
print(a)


