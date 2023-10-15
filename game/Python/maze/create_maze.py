import numpy as np
class maze:
    #(高さ,横,迷路作成に対する試行回数)
    def __init__(self,height,width,start_x=2,start_y=2,iteration=1000):
        self.height=height
        self.width=width
        self.Panel=np.ones((self.height*2+1,self.width*2+1))
        self.Panel=np.pad(self.Panel,((1,1),(1,1)))
        self.iteration=iteration
        self.start_x=start_x
        self.start_y=start_y

        #迷路作成(穴掘り法)
    def clear(self):
        self.Panel=np.ones((self.height*2+1,self.width*2+1))
        self.Panel=np.pad(self.Panel,((1,1),(1,1)))
    def create(self):
        self.Panel[self.start_y][self.start_x]=0
        x=self.start_x
        y=self.start_y
        street=[[y,x]]
        for iteration in range(self.iteration):
            direction_array=[0,1,2,3]
            for i in range(len(direction_array)):
                direction=direction_array[np.random.randint(len(direction_array))]
                if direction==0:
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
                
            if len(direction_array)==0:
                xy=street[np.random.randint(len(street))]
                x=xy[1]
                y=xy[0]
            if len(street)==self.height*self.width:
                print(iteration)
                break
        self.Panel[self.start_y][self.start_x]=2
        
        return self.Panel







