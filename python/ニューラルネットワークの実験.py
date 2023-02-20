import numpy as np
import math
weight=np.array([[1.1,-1.1,1.1],[2.1,-2.1,-1.1],[2.3,-1.6,1.6]])
data=np.array([[0,5],[1,1],[5,0],[6,2],[2,6],[2,2]])
teacher_vector=np.array([1,1,1,0,0,0]) 
#中間層のεの保存
djdw2=np.array([1.0,0.0])
#学習係数
alpha=0.5

#重みをランダムに定義
def make_weight():
 w=np.random.rand(3,3)
 return w
 
#def make_data(d):

#微分定義(関数,数値,リミットのやつ)使わない
def differential(f,x,h=0.1**6):
 df_dx=(f(x+h)-f(x-h))/2*h
 return df_dx
#シグモイド関数
def sigmoid_function(x):
 return 1/(1+math.exp(-x)) 
 #(データ,重み,活性化関数,最大エポック数)
def neural_network(d,w,activate_function,max_epoch=1000):
 #最大エポック数回まで処理を行う(再起関数を使うことは良くないことがわかったためfor文を利用してます。
 for m_e in range(max_epoch):
  #最初の1回目(m_eが0)だけはそれぞれのデータに対して1を先頭に付け加える
  if m_e==0:
   d=np.insert(d,0,1,axis=1)
  #教師ベクトルと出力層のデータ初期設定
  teacher_vector=np.array([1,1,1,0,0,0])
  output=np.array([])
  #データのパターン分処理していく
  for j in range(d.shape[0]):
   #中間層のデータの初期化
   middle_data=np.array([1.0])
   
   #次の中間層のユニット数2個分処理
   for i in range(w.shape[0]-1):
    #内積
    h=np.dot(w[i],d[j])
    #活性化関数
    f=activate_function(h)
    #中間層にデータを挿入
    middle_data=np.append(middle_data,f)
    #中間層から出力層への線形和
   h=np.dot(w[2],middle_data)
   #それの活性化
   f=activate_function(h)
   #それぞれのパターンの出力層のデータを保存
   output=np.append(output,f)
   #本で言うところの出力層のε
   djdw3=(output[j]-teacher_vector[j])*output[j]*(1-output[j])
   #出力層の重み修正
   w[2]=w[2]-alpha*djdw3*middle_data
   #中間層のεを求める
   for k in range(middle_data.shape[0]-1):
    djdw2[k]=(djdw3*w[2][k+1])*middle_data[k+1]*(1-middle_data[k+1])
   #中間層の重み修正
   for k in range(w.shape[0]-1):
    w[k]=w[k]-alpha*djdw2[k]*d[j]
  #全てのデータに関して正しく識別できた数をカウント。6になれば全て成功
  count=0
  for j in range(d.shape[0]):
   if teacher_vector[j]==1 and output[j]>teacher_vector[j]-0.1:
    count+=1
   elif teacher_vector[j]==0 and output[j]<teacher_vector[j]+0.1:
    count+=1
  #countが6になった時に学習終了する
  if count==d.shape[0]:
   m_e=max_epoch
   break
 #重みを返す
 return w
  #j=0

#新しい入力データがどのクラスに分類されるかを判別する関数
#(新しい入力データ,データ,重み,活性化関数,最大エポック数)
def test(x,d=data,w=weight,activate_function=sigmoid_function,max_epoch=1000):
 #中間層を初期化
 middle_data=np.array([1])
 #新しい入力データの先頭に1を加える。
 x=np.insert(x,0,1,axis=0)
 #新しい入力データから中間層への処理をしていく。
 
 for i in range(w.shape[0]-1):
  h=np.dot(w[i],x)
  f=activate_function(h)
  middle_data=np.append(middle_data,f)
 #中間層から出力層への処理
 h=np.dot(w[2],middle_data)
 #活性化
 f=activate_function(h)
 #この出力層から得られた値を0.5より大きいか小さいかでクラスを識別。0.5ピッタリの時はエラー表示
 if h>0.5:
  print("これはクラス1に属します")
 elif h<0.5:
  print("これはクラス2に属します")
 else:
  print("error")
 
#新しい入力データを定義している
test_data=np.array([2,2])
#重みをランダムに作る。
weight=make_weight()

print("ニューラルネットワークを通して得られる重みは",neural_network(data,weight,sigmoid_function))

weight=neural_network(data,weight,sigmoid_function)

#test(test_data)
#neural_network(data,weight,sigmoid_function,100000000)