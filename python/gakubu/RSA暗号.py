import numpy as np
import math
import random
#文章を数値化するために配列に入れる。aは0,bは1
alphabet=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
#2のi乗の結果を羅列。2の16乗まで書いてある。
pow_2=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]


#文字列を数値化する
def sentence_digit(sentence):
 #空の配列
 se_num=[] 
 #文字列を一文字ずつ配列化
 sp_se=list(sentence)
 #以下は文字列を一文字ずつ数値化していく
 for i in range(len(sp_se)):
  for j in range(len(alphabet)):
   #se_se[i]の文字がalphabetの何番目にあるかを検索
   if alphabet[j] in sp_se[i]:
    #se_numに文字列の数値jを入れる
    se_num.append(j)
    break
 return se_num
#数値から文字列にする
def digit_sentence(digit):
 #空の配列
 num_se =[]
 #数値を一つずつ文字列にしていく
 for i in range(len(digit)):
  #num_seに文字を配列として入れていく
  num_se.append(alphabet[digit[i]])
  #num_seの配列の文字列を結合して一つの文章にする
 sentence="".join(num_se)
 return sentence
#フェルマー法
def felmats_factorization_algorithm(n):
 #nの平方根よりでかい最小の整数
 x=math.ceil(math.sqrt(n))
 #y=1にしたらダメだと思う。
 y=0
 #nより左辺がでかい時、常にyに1を足し続ける
 while x**2-y**2>n:
  y=y+1
 #左辺=nになったときaとbを定義
 if x**2-y**2==n:
  a=x-y
  b=x+y
  #返すのはaだけ
  return a
 #上のリターンがおこなわれない時に左辺=nでないとき
 while x**2-y**2!=n:
  #左辺がnより大きい時
  while x**2-y**2>n:
   #yに1を足す
   y=y+1
  #左辺がnより小さい時xに1を足し、yを0に戻す
  if x**2-y**2<n:
   x=x+1
   y=0
  #以上の流れで左辺=nになった時aとbを定義
  if x**2-y**2==n:
   a=x-y
   b=x+y
   #aだけを返す。aが1であることが大事だから。
   return a

#繰り返し二乗法
def repeated_square_method(m):
 #mを2の何乗かを羅列していく配列
 rsm=[]
 #rsmを探していく。
 while m>0:
  for i in range(len(pow_2)):
   #mよりpow_2[i+1]よりでかい時にiが求めたい値としてrsmに入れていく
   if pow_2[i+1]>m:
     m=m-pow_2[i]
     rsm.append(i)
     break
 
 return rsm
#(nのm乗のmod)
def modulo(n,m,mod):
 rsmm=[n]
 rsm=repeated_square_method(m)
 #nの2^k乗までのmodを求めて配列に入れる。
 for i in range(max(rsm)):  
  rsmm.append((rsmm[i]**2)%mod)
 a=1
 #aがnのm乗のmodを求める。
 for i in range(len(rsm)):
  a=rsmm[rsm[i]]*a
  a=a%mod
 return a
#最大公倍数
def gcd(x,y):
 switch=0
 count=0
 a=[]
 if x==y:
  return x
 elif x>y:
  a.append(y)
  a.append(x)
 elif y>x:
  a.append(y)
  a.append(x)
 while switch==0:
  a.append(a[count]%a[count+1])
  if a[count+2]==0:
   switch=1
  count+=1
 return a[count]
#ユークリッドの互助法
def euclidean(x,y):
 switch=0
 r=0
 a=[]

 while switch==0 or switch==1:
  s=(-x*r+gcd(x,y))/y
  if s.is_integer():
   switch+=1
   a.append((r,s))
  r+=1
 rr=a[0][0]-a[1][0]
 ss=a[0][1]-a[1][1]

 return ss+a[0][1]
#RSA暗号
def rsa_cryptography(plaintext,p,q):
 #plaintextを数値化
 digit=sentence_digit(plaintext)
 #暗号キー、空の配列用意
 Encode_number=[]
 #複合キー、空の配列用意
 Decode_number=[]
 #暗号文の数値、空の配列用意
 ciphertext=[]
 #複合文章の数値、空の配列用意
 Encodetext_d=[]
 #p*qしたものをn
 n=p*q
 #eulerのファイ関数
 m=(p-1)*(q-1) 
 #各文字の数値の暗号キーを作る。
 for i in range(len(digit)):
  #暗号キーを2から1002までの数値でランダムに取る
  E=math.floor(random.random()*1000)+2
  #Eが奇数になるまでEを定義する。
  while E%2==0:
   E=math.floor(random.random()*1000)+2
  #フェルマー法によりEが素数か判定
  a=felmats_factorization_algorithm(E)
  #aが1ならば素数、1以外ならばもう一度同じ手順を繰り返す
  while a!=1:
   E=math.floor(random.random()*1000)+2
   while E%2==0:
    E=math.floor(random.random()*1000)+2
   a=felmats_factorization_algorithm(E)
  #それぞれの文字の暗号キーを配列に入れる
  Encode_number.append(E)
 #複合キーを作る
 for i in range(len(Encode_number)):
  #ユークリッドの互助法を用いて複合キーを作る。
  Decode_number.append(euclidean(m,Encode_number[i])%m)
 #数値化した文章を暗号化する。
 for i in range(len(digit)):
  ciphertext.append(modulo(digit[i],Encode_number[i],n))
 #暗号化した文章を複合化して数値に戻す。
 for i in range(len(Decode_number)):
  Encodetext_d.append(modulo(ciphertext[i],Decode_number[i],n))
 #複合化した数値を文章に戻す
 Encodetext=digit_sentence(Encodetext_d)
 #以下はそれぞれの数値の確認
 print(f"文章{plaintext}")
 print(f"文章を数値化{digit}")
 print(f"暗号文{ciphertext}")
 print(f"暗号キー{Encode_number}")
 print(f"複合キー{Decode_number}")
 print(f"暗号文を複合した文章の数値{Encodetext_d}")
 print(f"暗号文を複合した文章{Encodetext}")
 
 return ciphertext

rsa_cryptography("hello",23,29)