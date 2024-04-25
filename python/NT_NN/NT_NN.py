import numpy as np

#整数判定
def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False
def cos_similarity(x, y):
    nx = x/(np.sqrt(np.sum(x ** 2)))
    ny = y/(np.sqrt(np.sum(y ** 2)))
    return nx@ny

#評価関数
class Evaluation_function():
    class MSE():
        def func(x,t):
            return 0.5*np.sum(x-t)**2
        def dif(x,t):
            return x-t
    class Cross_Entropy():
        def func(x,t):
            delta=1e-7
            return np.sum(-t*np.log(x+delta))
        def dif(x,t):
            x=x+1e-7
            return -t/x
    class Softmax_with_Cross_Entropy():
        def func(x,t):
            s=activation_function.Softmax.func(x)
            return Evaluation_function.Cross_Entropy(s,t)
        def dif(x,t):
            return activation_function.Softmax.func(x)-t

#活性化関数
class activation_function():
    class ReLU():
        #関数
        def func(x):
            y=np.maximum(0,x)
            return y
        #微分
        def dif(x):
            y=np.where(x>0,1,0)
            return y
    #ソフトマックス関数 
    class Softmax():
        def func(x):
            x-=np.max(x)
            exp_x=np.exp(x)
            return exp_x/np.sum(exp_x)
        def dif(x):
            s = activation_function.Softmax.func(x)
            return np.sum(np.diagflat(s)-np.outer(s,s),axis=1)
    #シグモイド関数
    class Sigmoid():
        def func(x):
            return np.array(1/(1+np.exp(-x)))
        #本当の微分の結果
        def dif(x):
            s=activation_function.Sigmoid.func(x)
            return np.array(s*(1-s))
    class Linear():
        def func(x):
            return x
        def dif(x):
            return np.ones(x.shape)
class Layer():
    #全結合
    class FC():
        #FC(入力サイズ,出力サイズ,活性化関数,評価関数,学習率,バッチ数)
        def __init__(self,input_size,output_size,activation_func=activation_function.Sigmoid,evaluation_func=None,learning_rate=0.01,batch_num=1):
            self.input_size=input_size
            self.output_size=output_size
            if activation_func==activation_function.ReLU:
                self.weight=np.random.normal(0.0,(2/input_size)**0.5,(output_size,input_size))
                self.bias_w=np.random.normal(0.0,(2/input_size)**0.5,(output_size))
            else:
                self.weight=np.random.normal(0.0,input_size**-0.5,(output_size,input_size))
                self.bias_w=np.random.normal(0.0,input_size**-0.5,(output_size))
            self.learning_rate=learning_rate
            self.h=np.zeros(output_size)
            self.evaluation_func=evaluation_func
            self.activate_func=activation_func
            self.output=np.zeros(output_size)
            self.error=np.zeros(input_size)
            self.out_error=np.zeros(output_size)
            self.teacher_vector=None
            self.batch_num=batch_num
            self.batch_count=0
            self.grad=np.zeros_like(self.weight)
            self.bias_grad=np.zeros_like(self.bias_w)
        def set_weight(self,w,bw):
            self.weight=w
            self.bias_w=bw
        #前方方向
        def forward(self,x):
            self.x=np.array(x.ravel())
            #内積
            self.h=self.x@self.weight.T+self.bias_w
            #活性化関数にバイアスとセットで
            self.output=self.activate_func.func(self.h)
            return self.output
        #出力層の逆伝搬につかう
        def first_backward(self,teacher_vector=None,label=None):
            if teacher_vector is None and label is not None:
                self.teacher_vector=np.zeros(self.output_size)
                self.teacher_vector[label]=1
            else:
                self.teacher_vector=teacher_vector
            #誤差を保存
            self.out_error=self.evaluation_func.dif(self.output,self.teacher_vector)*self.activate_func.dif(self.h)
            self.grad+=np.array([self.out_error]).T@np.array([self.x])
            self.bias_grad+=self.out_error
            self.batch_count+=1
            self.error=self.out_error@self.weight  
            if self.batch_count==self.batch_num:
                #重みを更新
                self.weight=self.weight-self.learning_rate*self.grad/self.batch_num
                #バイアスの重みを更新
                self.bias_w=self.bias_w-self.learning_rate*self.bias_grad/self.batch_num
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
                self.batch_count=0   
            return self.error
        #中間層の逆伝搬に使う
        def backward(self,forward_error):    
            forward_error=forward_error*self.activate_func.dif(self.h) 
            self.grad+=np.array([forward_error]).T@np.array([self.x])
            self.bias_grad+=forward_error
            self.batch_count+=1
            self.error=forward_error@self.weight 
            if self.batch_count==self.batch_num:
                #以下で重みを更新していく
                self.weight=self.weight-self.learning_rate*self.grad/self.batch_num
                self.bias_w=self.bias_w-self.learning_rate*self.bias_grad/self.batch_num
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
                self.batch_count=0
            return self.error
        def last_backward(self,forward_error):    
            forward_error=forward_error*self.activate_func.dif(self.h)    
            self.grad+=np.array([forward_error]).T@np.array([self.x])
            self.bias_grad+=forward_error
            self.batch_count+=1
            if self.batch_count==self.batch_num:
                #以下で重みを更新していく
                self.weight=self.weight-self.learning_rate*self.grad/self.batch_num
                self.bias_w=self.bias_w-self.learning_rate*self.bias_grad/self.batch_num
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
                self.batch_count=0
            return self.error
        def first_grad_backward(self,c):
            self.out_error=self.evaluation_func.dif(self.output[c],1)*self.activate_func.dif(self.h[c])
            self.error=self.out_error*self.weight[c]  
            return self.error
        def grad_backward(self,forward_error):
            forward_error=forward_error*self.activate_func.dif(self.h)  
            self.error=forward_error@self.weight   
            return self.error
        def delete_unit(self,epsilon=0.99):
            for i in range(self.output_size):
                for j in range(i+1,self.output_size):
                    a=cos_similarity(self.weight[i],self.weight[j])
                    if a>=epsilon:
                        print(i,j)
        def high_weight(self,c,x):
            unit_value=self.weight[c]*x
            print(unit_value)

        def loss(self):
            return Evaluation_function.Cross_Entropy.func(self.output,self.teacher_vector)
    class FC_Softmax_CrossEntropy():
        #FC(入力サイズ,出力サイズ,活性化関数,評価関数,学習率,バッチ数)
        def __init__(self,input_size,output_size,learning_rate=0.01,batch_num=1):
            self.input_size=input_size
            self.output_size=output_size
         
            self.weight=np.random.normal(0.0,input_size**-0.5,(output_size,input_size))
            self.bias_w=np.random.normal(0.0,input_size**-0.5,(output_size))
            self.learning_rate=learning_rate
            self.h=np.zeros(output_size)
            self.output=np.zeros(output_size)
            self.error=np.zeros(input_size)
            self.out_error=np.zeros(output_size)
            self.teacher_vector=None
            self.batch_num=batch_num
            self.batch_count=0
            self.grad=np.zeros_like(self.weight)
            self.bias_grad=np.zeros_like(self.bias_w)
        def set_weight(self,w,bw):
            self.weight=w
            self.bias_w=bw
        #前方方向
        def forward(self,x):
            self.x=np.array(x.ravel())
            #内積
            self.h=self.x@self.weight.T+self.bias_w
            #活性化関数にバイアスとセットで
            self.output=activation_function.Softmax.func(self.h)
            return self.output
        #出力層の逆伝搬につかう
        def first_backward(self,teacher_vector=None,label=None):
            if teacher_vector is None and label is not None:
                self.teacher_vector=np.zeros(self.output_size)
                self.teacher_vector[label]=1
            else:
                self.teacher_vector=teacher_vector
            #誤差を保存
            self.out_error=self.output-self.teacher_vector
            self.grad+=np.array([self.out_error]).T@np.array([self.x])
            self.bias_grad+=self.out_error
            self.batch_count+=1
            self.error=self.out_error@self.weight  
            if self.batch_count==self.batch_num:
                #重みを更新
                self.weight=self.weight-self.learning_rate*self.grad/self.batch_num
                #バイアスの重みを更新
                self.bias_w=self.bias_w-self.learning_rate*self.bias_grad/self.batch_num
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
                self.batch_count=0   
            return self.error
        #中間層の逆伝搬に使う
        def backward(self,forward_error):    
            forward_error=forward_error*activation_function.Softmax.func(self.h) 
            self.grad+=np.array([forward_error]).T@np.array([self.x])
            self.bias_grad+=forward_error
            self.batch_count+=1
            self.error=forward_error@self.weight 
            if self.batch_count==self.batch_num:
                #以下で重みを更新していく
                self.weight=self.weight-self.learning_rate*self.grad/self.batch_num
                self.bias_w=self.bias_w-self.learning_rate*self.bias_grad/self.batch_num
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
                self.batch_count=0
            return self.error
        def last_backward(self,forward_error):    
            forward_error=forward_error*self.activate_func.dif(self.h)    
            self.grad+=np.array([forward_error]).T@np.array([self.x])
            self.bias_grad+=forward_error
            self.batch_count+=1
            if self.batch_count==self.batch_num:
                #以下で重みを更新していく
                self.weight=self.weight-self.learning_rate*self.grad/self.batch_num
                self.bias_w=self.bias_w-self.learning_rate*self.bias_grad/self.batch_num
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
                self.batch_count=0
            return self.error
        def first_grad_backward(self,c):
            out_error=self.output[c]-self.teacher_vector[c]
            self.error=out_error*self.weight[c]  
            return self.error
        def grad_backward(self,forward_error):
            forward_error=forward_error*activation_function.Sigmoid.dif(self.h)  
            self.error=forward_error@self.weight   
            return self.error
        def delete_unit(self,epsilon=0.01):
            for i in range(self.output_size):
                for j in range(i,self.output_size):
                    a=cos_similarity(self.weight[i],self.weight[j])
                    if a<=epsilon:
                        print(i,j)
        def high_weight(self,c,x,upper=0.6):
            unit_value=self.weight[c]*x
            argmax_unit_value=np.argmax(unit_value)
            return argmax_unit_value

        def loss(self):
            return Evaluation_function.Cross_Entropy.func(self.output,self.teacher_vector)
    class CNN():
        #畳み込みニューラルネットワーク(データの行,列,チャネル,kernelの行,kernelの列,カーネルのチャネル,stride,活性化関数,学習率)
        def __init__(self,row,column,channel,kernel_row,kernel_column,kernel_channel,st=1,activation_func=activation_function.ReLU,learning_rate=0.01,batch_num=1):
            self.row=row
            self.column=column
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((self.row-kernel_row)/st) and is_integer_num((self.column-kernel_column)/st):

                self.channel=channel
                self.kernel_channel=kernel_channel
                self.kernel_row=kernel_row
                self.kernel_column=kernel_column
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(row-kernel_row)/st)
                self.output_W=int(1+(column-kernel_column)/st)
                #重み作成
                self.weight=np.random.rand(kernel_channel,channel*kernel_row*kernel_column)
                self.grad=np.zeros_like(self.weight)
                self.weight=np.random.normal(0.0,(row*column*channel)**-0.5,(kernel_channel,channel*kernel_row*kernel_column))
                self.bias_w=np.random.normal(0.0,(row*column*channel)**-0.5,(kernel_channel))
                #バイアスの重みを作成
                self.bias_grad=np.zeros_like(self.bias_w)

                self.output_all_unit=kernel_channel*self.output_H*self.output_W
                #活性化関数
                self.activation_func=activation_func
                self.learning_rate=learning_rate

                #とりあえず出力結果を作成
                self.output=np.zeros((kernel_channel,self.output_H,self.output_W))
                self.error=np.zeros((channel,row,column))
                self.st=st
                self.col=np.zeros((self.channel,self.kernel_row,self.kernel_column,self.output_H,self.output_W))
                self.h=np.zeros(self.kernel_channel*self.output_H*self.output_W).reshape(self.kernel_channel,self.output_H*self.output_W)
                self.y=np.zeros((self.kernel_channel,self.output_H,self.output_W))

                self.batch_count=0
                self.batch_num=batch_num
            else:
                print("色々おかしいCNN")
        def set_weight(self,w,bw):
            self.weight=w
            self.bias_w=bw
        def im2col(self,x):
            data_size=len(x)
            #畳み込みをするために入力データの形を変えていく
            col=np.zeros((data_size,self.channel,self.kernel_row,self.kernel_column,self.output_H,self.output_W))
            for d in range(data_size):
                for k in range(self.channel):
                    for h in range(self.kernel_row):
                        for w in range(self.kernel_column):
                            col[d,k,h,w,:,:] = x[d,k,h:h+self.output_H*self.st:self.st,w:w+self.output_W*self.st:self.st]
            col=col.flatten()
            col=col.reshape((-1,self.output_H*self.output_W))
            col=np.split(col,data_size)
            col=np.transpose(col,(0,2,1))
            #col=np.concatenate(col.reshape(data_size,self.channel,self.kernel_row*self.kernel_column,self.output_H*self.output_W)).T.reshape(data_size,self.output_H*self.output_W,self.channel*self.kernel_row*self.kernel_column)
            return col
        def first_forward(self,x):
            self.col=x
            #colと重みの内積とバイアス
            self.h=self.col@self.weight.T+self.bias_w
            #hhのデータを１次元にして活性化関数に入れる。
            self.y=self.activation_func.func(self.h.T.ravel())
            #データを出力特徴マップの形に変える
            self.output=np.transpose(self.y.reshape(self.kernel_channel,self.output_H,self.output_W),(0,1,2))
            return self.output

        #前方方向の計算    
        def forward(self,x):
            #入力データを３次元にする
            while(x.ndim<3):
                x=x[np.newaxis]
            #畳み込みをするために入力データの形を変えていく
            self.col=np.zeros((self.channel,self.kernel_row,self.kernel_column,self.output_H,self.output_W))
            for k in range(self.channel):
                for h in range(self.kernel_row):
                    for w in range(self.kernel_column):
                        self.col[k,h,w,:,:] = x[k,h:h+self.output_H*self.st:self.st,w:w+self.output_W*self.st:self.st]
            self.col=np.concatenate(self.col.reshape(self.channel,self.kernel_row*self.kernel_column,self.output_H*self.output_W)).T.reshape(self.output_H*self.output_W,self.channel*self.kernel_row*self.kernel_column)
            #colと重みの内積とバイアス
            self.h=self.col@self.weight.T+self.bias_w
            #hhのデータを１次元にして活性化関数に入れる。
            self.y=self.activation_func.func(self.h.T.ravel())
            #データを出力特徴マップの形に変える
            self.output=np.transpose(self.y.reshape(self.kernel_channel,self.output_H,self.output_W),(0,1,2))
            return self.output
        def backward(self,forward_error):
            #活性化関数と受け取った誤差をかけて適切な誤差にかえる
            forward_error=(forward_error.ravel()*self.activation_func.dif(self.h.ravel())).reshape((self.kernel_channel,self.output_H*self.output_W,1))
            #次の層の誤差を求める
            conpute_error=forward_error.reshape((self.kernel_channel,self.output_H,self.output_W))
            weight=self.weight.reshape((self.kernel_channel,self.channel,self.kernel_row,self.kernel_column))
            for c in range(self.kernel_channel):
                for oh in range(self.output_H):
                    for ow in range(self.output_W):
                        self.error[:,oh*self.st:oh*self.st+self.kernel_row,ow*self.st:ow*self.st+self.kernel_column]+=weight[c]*conpute_error[c][oh][ow]
            #計算をまとめる
            self.grad+=np.sum(self.col*forward_error,axis=1)
            self.bias_grad+=np.sum(forward_error,axis=1).ravel()
            self.batch_count+=1
            if self.batch_count==self.batch_num:
                #重み修正          
                self.bias_w-=self.learning_rate*self.bias_grad
                self.weight-=self.learning_rate*self.grad
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
            return self.error
        def last_backward(self,forward_error):
            #活性化関数と受け取った誤差をかけて適切な誤差にかえる
            forward_error=(forward_error.ravel()*self.activation_func.dif(self.h.ravel())).reshape((self.kernel_channel,self.output_H*self.output_W,1))
            #計算をまとめる
            self.grad+=np.sum(self.col*forward_error,axis=1)
            self.bias_grad+=np.sum(forward_error,axis=1).ravel()
            #重み修正          
            self.batch_count+=1
            if self.batch_count==self.batch_num:
                self.bias_w-=self.learning_rate*self.bias_grad
                self.weight-=self.learning_rate*self.grad
                self.grad=np.zeros_like(self.weight)
                self.bias_grad=np.zeros_like(self.bias_w)
            return self.error
        def grad_backward(self,forward_error):
            #活性化関数と受け取った誤差をかけて適切な誤差にかえる
            forward_error=(forward_error.ravel()*self.activation_func.dif(self.h.ravel())).reshape((self.kernel_channel,self.output_H,self.output_W))
            weight=self.weight.reshape((self.kernel_channel,self.channel,self.kernel_row,self.kernel_column))
            for c in range(self.kernel_channel):
                for oh in range(self.output_H):
                    for ow in range(self.output_W):
                        self.error[:,oh*self.st:oh*self.st+self.kernel_row,ow*self.st:ow*self.st+self.kernel_column]+=weight[c]*forward_error[c][oh][ow]
            return self.error
        def grad_CAM(self,forward_error):
            #forward_error=forward_error.reshape(self.output)
            grad_CAM_weight=np.sum(forward_error,axis=(1,2)).reshape(self.kernel_channel,1,1)
            #L_grad_cam=activation_function.ReLU.func(np.sum((self.output*grad_CAM_weight),axis=0))
            L_grad_cam=np.sum((self.output*grad_CAM_weight),axis=0)
            L_grad_cam-=np.min(L_grad_cam)
            return L_grad_cam
            


    #プーリング層(入力データ,kernelの高さ,kernelの横,stride,"max" or "avarage"のどちらかのモード)
    class Pooling():
        def __init__(self,row,column,channel,kernel_row,kernel_column,st=1,mode="max"):
            self.row=row
            self.column=column
            self.channel=channel
            #strideとデータとカーネルの都合がいいとたたみ込み可能
            if is_integer_num((row-kernel_row)/st) and is_integer_num((column-kernel_column)/st):
                #最終的に出力の形を求める(output_H,output_W,channel)
                self.output_H=int(1+(row-kernel_row)/st)
                self.output_W=int(1+(column-kernel_column)/st)
                self.kernel_row=kernel_row
                self.kernel_column=kernel_column
                self.weight=np.zeros((self.output_W*self.output_H*channel,self.kernel_column*self.kernel_row))
                self.col=np.zeros((self.channel,self.kernel_row,self.kernel_column,self.output_H,self.output_W))
                self.gen_mat=np.zeros((self.output_W*self.output_H*channel,self.kernel_column*self.kernel_row))
                #とりあえず出力結果を作成
                self.output=np.zeros((channel,self.output_H,self.output_W))
                self.st=st
                self.mode=mode
                self.error=np.zeros((channel,row,column))
            else:
                print("色々おかしいpooling")   
        def forward(self,x):
            while(x.ndim<3):
                x=x[np.newaxis]
                        #畳み込みをするために入力データの形を変えていく
            self.col=np.zeros((self.channel,self.kernel_row,self.kernel_column,self.output_H,self.output_W))
            for k in range(self.channel):
                for h in range(self.kernel_row):
                    for w in range(self.kernel_column):
                        self.col[k,h,w,:,:] = x[k,h:h+self.output_H*self.st:self.st,w:w+self.output_W*self.st:self.st]
            self.col=np.concatenate(self.col.reshape(self.channel,self.kernel_row*self.kernel_column,self.output_H*self.output_W).transpose(0,2,1))
            if self.mode=="max":
                output=np.zeros(self.channel*self.output_H*self.output_W)
                self.weight=np.zeros((self.output_W*self.output_H*self.channel,self.kernel_column*self.kernel_row))
                for i in range(self.output_H*self.output_W*self.channel):
                    max_v=np.NINF
                    max_index=0
                    for j in range(self.kernel_column*self.kernel_row):
                        if max_v<self.col[i][j]:
                            self.weight[i][max_index]=0
                            self.weight[i][j]=1
                            max_v=self.col[i][j]
                            max_index=j
                    output[i]=max_v
            elif self.mode=="average":
                bb=np.ones(self.kernel_row*self.kernel_column)
                self.weight=np.ones((self.output_W*self.output_H*self.channel,self.kernel_column*self.kernel_row))/self.kernel_row*self.kernel_column
                ave_mat=bb@self.col.T
                output=ave_mat/self.kernel_row*self.kernel_column
            self.output=output.reshape(self.channel,self.output_H,self.output_W)
            return self.output
        def backward(self,forward_error):
            #前方向の誤差が1次元配列とかなら3次元配列に変える。
            forward_error=forward_error.reshape(self.output.shape)
            er=forward_error.ravel().T*self.weight.T
            er=er.T
            self.error=np.zeros((self.channel,self.row,self.column))
            for d in range(self.channel):
                for oh in range(self.output_H):
                    for ow in range(self.output_W):
                        for i in range(self.kernel_row):
                            for j in range(self.kernel_column):
                                self.error[d][i+oh*self.st][j+ow*self.st]+=er[ow+oh*self.output_W+d*self.output_H*self.output_W][j+i*self.kernel_column]
            return self.error
    
    class Dropout:
        def __init__(self,p):
            if self.p>=0 and self.p<100:
                self.p=p
            else:
                print("Dropout error")
        #確率pは0～100のドロップアウト
        def forward(self,x):
            x_shape=x.shape
            x=x.raval()
            for i in range(len(x)):
                a=np.random.random()*100
                if a<=self.p:
                    x[i]=0
            x=x.reshape(x_shape)
            return x
    

        
import pickle
def save_object(file_name,object):
    with open(file_name+".pickle","wb") as f:
        pickle.dump(object,f)

def load_object(file_name,object):
    with open(file_name+".pickle","rb") as f:
        object=pickle.load(f)