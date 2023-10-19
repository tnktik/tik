import numpy as np
from create_maze import maze

def find(a,x):
    if a.ndim==2:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i][j]==x:
                    return (j,i)

class Q_learning:
    def __init__(self, Panel,learning_rate, discount_rate):
        # 迷路
        self.Panel=Panel
        # 状態（=エージェントの位置）は迷路の盤面数
        self.height,self.width=self.Panel.shape
        # 行動の数は上下左右の4種類
        self.num_action = 4
        # Qは 状態数 x 行動数(y,x)
        self.Q = np.zeros((self.num_action,self.Panel.shape[1],self.Panel.shape[0]))
        # 現在の状態(x,y)
        self.state=find(Panel,2)
        print(self.state[0])
        #学習率
        self.learning_rate=learning_rate
        #割引率
        self.discount_rate=discount_rate


    def select_action(self,random_rate=0.01):
        """一定の確率で、ベストでない動きをする"""
        if np.random.rand()<random_rate:
            return np.random.randint(self.num_action)
        """評価値の最も高い行動を探すaaaa"""
        return self.Q[:,self.state[0],self.state[1]].argmax()

    def reward(self,maze_clear):
        """報酬"""
        return 0 if maze_clear else -1
    def move(self,ob,action):
        return ob.move(action)

    def learning(self,ob,random_rate):
        # 行動の選択。ベストアクションとは限らない。
        action=self.select_action(random_rate)
        print(action)
        # 選択された行動に従い動く。ただし、壁がある場合は無視される
        self.Panel=self.move(ob,action)
        print(self.Panel)
        # 移動後の状態を取得
        next_state=np.where(self.Panel==2)
        # ベストアクションを選択
        next_action = self.Q[:,self.state[0],self.state[1]].argmax()
        # Q[s][a] += 学習率 * ( 報酬 + 割引率 * ( max_{s'} Q[s'][a'] ) - Q[s][a] )aaaa
        self.Q[action][self.state[0]][self.state[1]]+=self.learning_rate*(self.reward()+
                                                                          self.discount_rate*self.Q[next_action][next_state[0]][next_state[1]]-
                                                                          self.Q[action][self.state[0]][self.state[1]])
        # 移動後の状態を現在の状態に記録
        self.state = next_state

    