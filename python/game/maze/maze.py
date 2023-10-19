import sys
import pygame 
from pygame.locals import *
import numpy as np
from create_maze import maze

height=6
width=6
panel_size=15

def main():
    #操作キャラの現在位置
    player_x=2
    player_y=2
    #ゴール地点
    goal_x=width*2
    goal_y=height*2
    #移動回数
    step=0
    #クリア後の移動数
    clear_step=0
    #クリアしていない
    clear=False
    #画面描く
    pygame.init()
    screen=pygame.display.set_mode((800,600))
    #タイトル
    pygame.display.set_caption("迷路")
    surface=pygame.Surface(screen.get_size(),pygame.HWACCEL)
    font=pygame.font.SysFont("Arial",50)
    
    #迷路ぱらめーた(高さ,横)
    m=maze(height,width)
    #作成
    m.create(player_x,player_y)
    m.set_goal_pos(goal_x,goal_y)
    m.set_player(player_x,player_y)
    m.break_wall(3)
    m.load()

    Panel=m.Panel
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
                if Panel[i][j]==1:
                    color=(255,255,255)
                if Panel[i][j]==2:
                    color=(100,100,0)
                if Panel[i][j]==3:
                    color=(100,0,100)
                if Panel[i][j]==-1:
                    color=(0,50,0)
                pygame.draw.rect(surface,color,(panel_size*j+panel_size,panel_size*i+panel_size,panel_size,panel_size))
        #ステップ表示
        text=font.render(str(step),True,(0,0,200))
        surface.blit(text,(10,10))
        #ゴールした場合
        if m.clear==True:
            text=font.render("CLEAR!  STEP:"+str(clear_step),True,(250,250,0))
            surface.blit(text,(200,300))

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
                if m.clear==False:
                    if event.key==K_DOWN:
                        m.move(0)
                    if event.key==K_RIGHT:
                        m.move(1)
                    if event.key==K_LEFT:
                        m.move(2)
                    if event.key==K_UP:
                        m.move(3)

if __name__=="__main__":
    main()