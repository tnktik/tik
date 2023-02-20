import numpy as np
import matplotlib.pyplot as plt

#パターン行列
large_x=np.array([[1,1.2],[1,0.2],[1,-0.2],[1,-0.5],[1,-1],[1,-1.5]])
#教師ベクトル
teacher_vector=np.array([1,1,-1,1,-1,-1])
#実験、行列の積
print(np.dot(large_x.T,large_x))
print(np.linalg.inv(np.dot(large_x.T,large_x)))


XX_I_X=np.dot(np.linalg.inv(np.dot(large_x.T,large_x)),large_x.T)
print(XX_I_X)
#大域的最適解のw
global_optimal_solution_w=np.dot(XX_I_X,teacher_vector)
print(global_optimal_solution_w[0]/global_optimal_solution_w[1]*(-1))
plt.scatter(large_x[:,0],large_x[:,1])
plt.hlines(global_optimal_solution_w[0]/global_optimal_solution_w[1]*(-1),1.1,0.9)
plt.show()
print(f"大域的最適解は{global_optimal_solution_w}")
#xに数値を入れて以下を動かすとクラスを判定してくれる。
def pattern_x(x):
 p_x=np.array([1,x])
 g_x=np.inner(p_x,global_optimal_solution_w)
 if g_x<0:
  print(f"x={x}はクラス2に属する")
 else:
  print(f"x={x}はクラス1に属する")