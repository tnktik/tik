import sys
import pygame 
import json
from pygame.locals import *
import numpy as np
from create_maze import maze

height=15
width=15
panel_size=15
player_x=2
player_y=2

def main():
    player_x=2
    player_y=2
    #画面描く
    pygame.init()
    screen=pygame.display.set_mode((800,600))
    #タイトル
    pygame.display.set_caption("迷路")
    surface=pygame.Surface(screen.get_size(),pygame.HWACCEL)
    #現在のパネルの座標
    m=maze(height,width,player_y,player_x)
    Panel=m.create()

    while(True):
        #操作するもの
      
        screen.blit(surface,(0,0))
        #枠組み(10,20)ブロック
        #1ブロックあたり20ピクセル
        #色をつける。
        for i in range(len(Panel)):
            for j in range(len(Panel[i])):
                if Panel[i][j]==0:
                    color=(0,0,0)
                if Panel[i][j]==1 or Panel[i][j]==-1:
                    color=(255,255,255)
                if Panel[i][j]==2 or Panel[i][j]==-2:
                    color=(100,100,0)
                pygame.draw.rect(surface,color,(panel_size*j+panel_size,panel_size*i+panel_size,panel_size,panel_size))

        pygame.time.wait(10)
        pygame.display.update()

        #操作キー
        for event in pygame.event.get():
            if event.type==QUIT:
                pygame.quit()
                sys.exit()
            if event.type==KEYDOWN:
                if event.key==K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                #移動キー
                if event.key==K_DOWN:
                    if Panel[player_y+1][player_x]==0:
                        Panel[player_y][player_x]=0
                        player_y+=1
                        Panel[player_y][player_x]=2

                if event.key==K_RIGHT:
                    if Panel[player_y][player_x+1]==0:
                        Panel[player_y][player_x]=0
                        player_x+=1
                        Panel[player_y][player_x]=2


                if event.key==K_LEFT:
                    if Panel[player_y][player_x-1]==0:
                        Panel[player_y][player_x]=0
                        player_x-=1
                        Panel[player_y][player_x]=2

                if event.key==K_UP:
                    if Panel[player_y-1][player_x]==0:
                        Panel[player_y][player_x]=0
                        player_y-=1
                        Panel[player_y][player_x]=2


if __name__=="__main__":
    main()