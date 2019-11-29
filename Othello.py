#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.29
#   Intro: h黑白棋人机游戏
#####################################

# 导入相关模块
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time
from Othello_AI import *


class Othello(object):
    def __init__(self, board_size=8):
        '''
            初始化函数
            参数：
                board_size: 棋盘大小
            返回值：
                None
        '''
        super().__init__()

        # 参数设置
        self.board_size = board_size
        self.SCREEN_SIZE = 820
        self.BOARD_LEN = 780
        self.CELL_SIZE = self.BOARD_LEN // self.board_size
        self.BACKGROUND_COLOR = pygame.Color('seagreen')
        self.LINE_COLOR = pygame.Color('black')
        self.PLAYER_COLOR = pygame.Color('black')
        self.AI_COLOR = pygame.Color('white')

        # 确定棋盘四条边的位置
        self.left = (self.SCREEN_SIZE - self.BOARD_LEN) // 2
        self.right = self.left + self.CELL_SIZE * self.board_size
        self.top = (self.SCREEN_SIZE - self.BOARD_LEN) // 2
        self.bottom = self.top + self.CELL_SIZE * self.board_size

        # 执棋方，黑棋为先手，白棋为后手；1代表玩家，-1代表AI；默认玩家先手
        self.turn = 1

    
    def initialize_game(self):
        '''
            初始化游戏
            参数：
                None
            返回值：
                None
        '''
        # 初始化棋盘
        self.board = np.zeros((self.board_size, self.board_size))
        # 添加开局便存在的棋子
        self.add_init_coins()


    def draw_board(self):
        '''
            绘制棋盘
            参数：
                None
            返回值：
                None
        '''
        # 绘制方框
        for i in range(self.board_size+1):
            pygame.draw.line(self.screen, self.LINE_COLOR, \
                            (self.top+i*self.CELL_SIZE, self.left), (self.top+i*self.CELL_SIZE, self.right))
            pygame.draw.line(self.screen, self.LINE_COLOR, \
                            (self.top, self.left+i*self.CELL_SIZE), (self.bottom, self.left+i*self.CELL_SIZE))
        
        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    # 玩家的棋子
                    position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                                self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
                    pygame.draw.circle(self.screen, self.PLAYER_COLOR, position, int(self.CELL_SIZE*0.4))
                elif self.board[i, j] == -1:
                    # AI的棋子
                    position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                                self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
                    pygame.draw.circle(self.screen, self.AI_COLOR, position, int(self.CELL_SIZE*0.4))


    def add_init_coins(self):
        '''
            添加开局便存在的棋子
            参数：
                None
            返回值：
                None:
        '''
        mid = self.board_size // 2
        self.board[mid, mid] = -self.turn
        self.board[mid-1, mid-1] = -self.turn
        self.board[mid, mid-1] = self.turn
        self.board[mid-1, mid] = self.turn


    def play(self):
        '''
            开始游戏
            参数：
                None
            返回值:
                None
        '''
        # 初始化
        pygame.init()
        self.initialize_game()
        # 屏幕设置
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        # 标题设置
        pygame.display.set_caption('Othello Game')
        # 背景颜色设置
        self.screen.fill(self.BACKGROUND_COLOR)

        # 读取先手方
        self.turn = input('First(1) or second(-1)?')
        while self.turn not in ('1', '-1'):
            self.turn = input('Error Input! Input Again: ')
        self.turn = int(self.turn)

        # 默认玩家先手，而先手执黑；所以AI先手要更换颜色
        if self.turn == -1:
            self.PLAYER_COLOR, self.AI_COLOR = self.AI_COLOR, self.PLAYER_COLOR

        # 初始化四邻居统计，只有有邻居的空格的值才大于0
        self.valid_board = np.zeros_like(self.board)
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                    if self.board[i, j] == 0 \
                        and np.sum(self.board[max(0, i-1):min(self.board_size, i+2), \
                            max(0, j-1):min(self.board_size, j+2)]!=0)>0:
                        self.valid_board[i, j] += 1

        # 初始化AI
        self.ai = Othello_AI(self.board, self.valid_board)

        # 表示连续pass的次数，连续两次pass，说明双方都无棋可下，游戏结束
        self.pass_time = 0

        print("#---------------------- Game Start ----------------------#")
        while self.pass_time < 2:
            # 绘制棋盘
            self.draw_board()
            # 刷新
            pygame.display.update()
            
            # 获取当前可下的棋子
            step_list = self.get_steps()
            # 无可下棋子则直接pass
            if len(step_list) == 0:
                self.pass_time += 1
                self.turn = -self.turn
                continue
            else:
                self.pass_time = 0

            if self.turn == -1:
                # AI执棋
                cur_time = time.time()

                # 获取落子位置
                row, col = self.ai.search()

                # 翻转棋子
                for i, j in step_list[(row, col)]:
                    self.board[i, j] = self.turn

                # 落子
                self.board[row, col] = self.turn

                # 更新邻居统计
                for i in range(max(0, row-1), min(self.board_size, row+2)):
                    for j in range(max(0, col-1), min(self.board_size, col+2)):
                        if self.board[i, j] == 0:
                            self.valid_board[i, j] += 1
                self.valid_board[row, col] = 0

                # 转换执棋方
                self.turn = -self.turn

                # 输出记录
                score = 0
                print('AI turn: {}, score: {}, time cost: {}'.format((row, col), \
                    score, time.time()-cur_time))
            else:
                # 先清空事件队列
                pygame.event.clear()
                while self.turn == 1:
                    # 处理事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            # 关闭窗口
                            pygame.quit()
                            print("#----------------------- Game End -----------------------#")
                            return
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            # 鼠标按键
                            # 计算落子位置
                            row = (event.pos[1] - self.top) // self.CELL_SIZE
                            col = (event.pos[0] - self.left) // self.CELL_SIZE
                            # 添加棋子
                            if (0<=row<self.board_size) and (0<=col<self.board_size) \
                                and ((row, col) in step_list):
                                # 落子
                                self.board[row, col] = self.turn

                                # 翻转棋子
                                for i, j in step_list[(row, col)]:
                                    self.board[i, j] = self.turn

                                # 转换执棋方
                                self.turn = -self.turn

                                # 输出记录
                                score = 0
                                print('Player turn: {}, score: {}'.format((row, col), score))
                                
                                # 更新邻居统计
                                for i in range(max(0, row-1), min(self.board_size, row+2)):
                                    for j in range(max(0, col-1), min(self.board_size, col+2)):
                                        if self.board[i, j] == 0:
                                            self.valid_board[i, j] += 1
                                self.valid_board[row, col] = 0

                                # 跳出等待事件循环
                                break

        # 结束后计算双方的棋子数
        ai_count = np.sum(self.board==-1)
        player_count = np.sum(self.board==1)
        print('You : AI = {} : {}, with {} empty cell(s) left!'.format(\
            player_count, ai_count, self.board_size**2-player_count-ai_count))

        # 判断胜负
        if ai_count > player_count:
            print('AI Win!')
        elif ai_count < player_count:
            print('You Win!')
        else:
            print('Draw!')
        print('#----------------------- Game End -----------------------#')

        while True:
            # 等待关闭窗口
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return


    def get_steps(self):
        '''
            获取可落子的格子列表
            参数：
                None
            返回值：
                step_list: {可下空格：相应要翻转的棋子列表}
        '''
        step_list = {}
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 只考虑有邻居的空格
                if self.valid_board[i, j] > 0:
                    # 要翻转的棋子列表
                    flip_list = []

                    # 向上探索
                    k = 1
                    while i-k > 0 and self.board[i-k, j] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i-k, j] == self.turn:
                        flip_list += [(i-kk, j) for kk in range(1, k)]
                    
                    # 向下探索
                    k = 1
                    while i+k < self.board_size-1 and self.board[i+k, j] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i+k, j] == self.turn:
                        flip_list += [(i+kk, j) for kk in range(1, k)]

                    # 向左探索
                    k = 1
                    while j-k > 0 and self.board[i, j-k] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i, j-k] == self.turn:
                        flip_list += [(i, j-kk) for kk in range(1, k)]

                    # 向右探索
                    k = 1
                    while j+k < self.board_size-1 and self.board[i, j+k] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i, j+k] == self.turn:
                        flip_list += [(i, j+kk) for kk in range(1, k)]

                    # 向左上探索
                    k = 1
                    while j-k > 0 and i-k > 0 and self.board[i-k, j-k] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i-k, j-k] == self.turn:
                        flip_list += [(i-kk, j-kk) for kk in range(1, k)]

                    # 向左下探索
                    k = 1
                    while j-k > 0 and i+k < self.board_size-1 and self.board[i+k, j-k] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i+k, j-k] == self.turn:
                        flip_list += [(i+kk, j-kk) for kk in range(1, k)]

                    k = 1
                    while j+k < self.board_size-1 and i-k > 0 and self.board[i-k, j+k] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i-k, j+k] == self.turn:
                        flip_list += [(i-kk, j+kk) for kk in range(1, k)]

                    # 向右下探索
                    k = 1
                    while j+k < self.board_size-1 and i+k < self.board_size-1 and self.board[i+k, j+k] == -self.turn:
                        k += 1
                    if k > 1 and self.board[i+k, j+k] == self.turn:
                        flip_list += [(i+kk, j+kk) for kk in range(1, k)]

                    # 如果有可以翻转的棋子，则此空格可落子
                    if len(flip_list) > 0:
                        step_list[(i, j)] = flip_list

        return step_list


if __name__ == "__main__":
    game = Othello()
    game.play()
