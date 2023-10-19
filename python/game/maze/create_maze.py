import numpy as np
import shelve
class maze:
    #(高さ,横,迷路作成に対する最大試行回数)
    def __init__(self,height,width,iteration=1000):
        self.height=height
        self.width=width
        self.Panel=np.ones((self.height*2+1,self.width*2+1))
        self.Panel=np.pad(self.Panel,((1,1),(1,1)))
        self.iteration=iteration
        self.clear=False
        self.step=0
    #迷路を消す
    def clear(self):
        self.Panel=np.ones((self.height*2+1,self.width*2+1))
        self.Panel=np.pad(self.Panel,((1,1),(1,1)))
    #初期地点からやりなおし
    def reset(self):
        self.Panel[self.y][self.x]=0
        self.set_player(self.init_x,self.init_y)
        self.set_goal_pos(self.goal_x,self.goal_y)
        self.clear=False
        self.step=0

    #スタート地点を設定
    def set_player(self,x,y):
        self.init_x=x
        self.init_y=y
        self.x=x
        self.y=y
        self.Panel[y][x]=2
        return self.Panel
    #プレイヤーの座標を取得
    def get_player_pos(self):
        player_pos=(self.y,self.x)
        return player_pos
    #ゴール地点の設定
    def set_goal_pos(self,x,y):
        self.goal_x=x
        self.goal_y=y
        self.Panel[y][x]=3
        return self.Panel
    #壁を壊す
    def break_wall(self,count):
        for i in range(count):
            while(True):
                x=np.random.randint((self.width-1))*2+3
                y=np.random.randint((self.height-1))*2+2
                if self.Panel[y][x]==1:
                    print(x,y)
                    self.Panel[y][x]=0
                    break
        return self.Panel
        
    #迷路とステップをセーブ
    def save(self):
                #マップを保存
        with shelve.open("maze_map") as shelf_file:
            shelf_file["maze1"]=self.Panel
        
    def load(self):
        #重みをロードする
        with shelve.open("maze_map") as shelf_file:
            self.Panel=shelf_file["maze1"]
            return self.Panel
       #移動
    def move(self,action):
        if action==0:
            #移動先が0(空白)あるいは3(ゴール)の時
            if self.Panel[self.y+1][self.x]==0 or self.Panel[self.y+1][self.x]==-1:
                self.Panel[self.y][self.x]=0
                #移動する
                self.y+=1
                self.Panel[self.y][self.x]=2
            elif self.Panel[self.y+1][self.x]==3:
                #クリアにする
                self.Panel[self.y][self.x]=0
                self.y+=1
                self.Panel[self.y][self.x]=2
                self.clear=True

        if action==1:
           #移動先が0(空白)あるいは3(ゴール)の時
            if self.Panel[self.y][self.x+1]==0 or self.Panel[self.y][self.x+1]==-1:
                self.Panel[self.y][self.x]=0
                #移動する
                self.x+=1
                self.Panel[self.y][self.x]=2
            elif self.Panel[self.y][self.x+1]==3:
                #クリアにする
                self.Panel[self.y][self.x]=0
                self.x+=1
                self.Panel[self.y][self.x]=2
                self.clear=True

        if action==2:
            #移動先が0(空白)あるいは3(ゴール)の時
            if self.Panel[self.y][self.x-1]==0 or self.Panel[self.y][self.x-1]==-1:
                self.Panel[self.y][self.x]=0
                #移動する
                self.x-=1
                self.Panel[self.y][self.x]=2
            elif self.Panel[self.y][self.x-1]==3:
                #クリアにする
                self.Panel[self.y][self.x]=0
                self.x-=1
                self.Panel[self.y][self.x]=2
                self.clear=True

        if action==3:
            #移動先が0(空白)あるいは3(ゴール)の時
            if self.Panel[self.y-1][self.x]==0 or self.Panel[self.y-1][self.x]==-1:
                self.Panel[self.y][self.x]=0
                #移動する
                self.y-=1
                self.Panel[self.y][self.x]=2
            elif self.Panel[self.y-1][self.x]==3:
                #クリアにする
                self.Panel[self.y][self.x]=0
                self.y-=1
                self.Panel[self.y][self.x]=2
                self.clear=True
    #迷路作成(穴掘り法)
    def create(self,start_x,start_y):
        #スタート地点を空気にする
        self.Panel[start_y][start_x]=0
        #スタート地点から穴掘り法
        x=start_x
        y=start_y
        #道をつくった座標を保存
        street=[[y,x]]
        for iteration in range(self.iteration):
            #進む方向を初期化
            direction_array=[0,1,2,3]
            for i in range(len(direction_array)):
                #進む方向をきめる
                direction=direction_array[np.random.randint(len(direction_array))]
                
                #方向に基づいて処理
                if direction==0:
                    #進む方向が現在地からニマス分壁なら空気にして、穴を掘る人がその場所に移動、道を保存
                    if self.Panel[y][x+1]==1 and self.Panel[y][x+2]==1:
                        self.Panel[y][x+1]=0
                        self.Panel[y][x+2]=0
                        x=x+2
                        street.append([y,x])
                        break
                    else:
                        direction_array.remove(direction)


                
                if direction==1:
                    if self.Panel[y][x-1]==1 and self.Panel[y][x-2]==1:
                        self.Panel[y][x-1]=0
                        self.Panel[y][x-2]=0
                        x=x-2
                        street.append([y,x])
                        break
                    else:
                        direction_array.remove(direction)


                if direction==2:
                    if self.Panel[y+1][x]==1 and self.Panel[y+2][x]==1:
                        self.Panel[y+1][x]=0
                        self.Panel[y+2][x]=0
                        y=y+2
                        street.append([y,x])
                        break
                    else:
                        direction_array.remove(direction)

                if direction==3:
                    if self.Panel[y-1][x]==1 and self.Panel[y-2][x]==1:
                        self.Panel[y-1][x]=0
                        self.Panel[y-2][x]=0
                        y=y-2
                        street.append([y,x])
                        break
                    else:
                        direction_array.remove(direction)
            
            #もし、どの方向にも進めなくなった場合
            if len(direction_array)==0:
                #保存した道をランダムに取得
                xy=street[np.random.randint(len(street))]
                x=xy[1]
                y=xy[0]
            #道にするべき場所をすべて空気にしたら終わり
            if len(street)==self.height*self.width:
                break
        #迷路のパネルを返す
        return self.Panel







