#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.29
#   Intro: 黑白棋人机游戏
#####################################

# 导入相关模块
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
#import pygame
import time
from Othello_AI import *
from MCTS import *
from policy_value_net import *


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
        # 执棋方，黑棋为先手(1)，白棋为后手(-1)
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
        # 表示连续pass的次数，连续两次pass，说明双方都无棋可下，游戏结束
        self.pass_time = 0
        # 游戏状态
        self.if_gameover = False
        # 记录最近一步落子之后的落子情况
        self.record_1 = []
        self.record_2 = []
        # 添加开局便存在的棋子
        self.add_init_coins()

        # 初始化四邻居统计，只有有邻居的空格的值才大于0
        self.valid_board = np.zeros_like(self.board)
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                    if self.board[i, j] == 0 \
                        and np.sum(self.board[max(0, i-1):min(self.board_size, i+2), \
                            max(0, j-1):min(self.board_size, j+2)]!=0)>0:
                        self.valid_board[i, j] += 1

        # 获取可下的棋子列表
        self.step_list = self.get_steps()


    def draw(self):
        '''
            绘制棋盘
            参数：
                None
            返回值：
                None
        '''
        # 覆盖画面
        self.screen.fill(self.BACKGROUND_COLOR)

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
                    # 玩家1的棋子
                    position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                                self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
                    pygame.draw.circle(self.screen, self.PLAYER_1_COLOR, position, int(self.CELL_SIZE*0.4))
                elif self.board[i, j] == -1:
                    # 玩家2的棋子
                    position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                                self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
                    pygame.draw.circle(self.screen, self.PLAYER_2_COLOR, position, int(self.CELL_SIZE*0.4))

        # 提示下一步可落子的位置
        for i, j in self.step_list.keys():
            position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                        self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
            pygame.draw.circle(self.screen, self.NEXT_STEP_COLOR, position, int(self.CELL_SIZE*0.1))

        # 刷新
        pygame.display.flip()


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
        
        
    def initialize_draw(self):
        '''
            初始化绘制流程
            参数：
                None
            返回值:
                None
        '''
        # 参数设置
        self.SCREEN_SIZE = 820
        self.BOARD_LEN = 780
        self.CELL_SIZE = self.BOARD_LEN // self.board_size
        self.BACKGROUND_COLOR = pygame.Color('seagreen')
        self.LINE_COLOR = pygame.Color('black')
        self.PLAYER_1_COLOR = pygame.Color('black')
        self.PLAYER_2_COLOR = pygame.Color('white')
        self.NEXT_STEP_COLOR = pygame.Color('pink')

        # 确定棋盘四条边的位置
        self.left = (self.SCREEN_SIZE - self.BOARD_LEN) // 2
        self.right = self.left + self.CELL_SIZE * self.board_size
        self.top = (self.SCREEN_SIZE - self.BOARD_LEN) // 2
        self.bottom = self.top + self.CELL_SIZE * self.board_size

        # 初始化
        pygame.init()
        # 屏幕设置
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        # 标题设置
        pygame.display.set_caption('Othello Game')
        # 背景颜色设置
        self.screen.fill(self.BACKGROUND_COLOR)


    def get_board(self):
        '''
            返回棋局
            参数：
                None
            返回值：
                board: 棋局
        '''
        return self.board


    def step(self, row, col):
        '''
            下一步棋（要事先保证有效性）
            参数：
                row, col: 下棋的位置
            返回值：
                step_list: 当前执棋方可下的棋子
        '''
        # 进行记录
        if self.turn == 1:
            self.record_1.append((row, col))
        else:
            self.record_2.append((row, col))

        # 重置pass次数
        self.pass_time = 0

        # 落子
        self.board[row, col] = self.turn

        # 翻转棋子
        for i, j in self.step_list[(row, col)]:
            self.board[i, j] = self.turn

        # 更新邻居统计
        for i in range(max(0, row-1), min(self.board_size, row+2)):
            for j in range(max(0, col-1), min(self.board_size, col+2)):
                if self.board[i, j] == 0:
                    self.valid_board[i, j] += 1
        self.valid_board[row, col] = 0

        # 转换执棋方
        self.turn = -self.turn

        # 获取可下的位置列表
        self.get_steps()

        # 自动pass
        while self.if_gameover == False and self.step_list == {}:
            self.pass_turn()

        return self.step_list


    def pass_turn(self):
        '''
            跳过当前回合，并判断游戏是否结束
            参数：
                None
            返回值：
                if_gameover: 布尔值
        '''
        self.pass_time += 1
        self.turn = -self.turn
        self.if_gameover = self.pass_time == 2
        self.get_steps()

        return self.if_gameover


    def get_state(self):
        '''
            获取当前棋局状态（棋子个数比）
            参数：
                None
            返回值：
                num1, num2: 玩家1、2的棋子数
        '''
        num2 = np.sum(self.board==-1)
        num1 = np.sum(self.board==1)

        return num1, num2


    def ai_vs_ai(self, ai_1, ai_2):
        '''
            ai与ai对弈
            参数：
                ai_1, ai_2: 两个ai
            返回值：
                winner: ai_1胜利为1,否则为-1
        '''
        self.initialize_game()
        
        # 下到终局
        while self.if_gameover == False:
            if self.turn == 1:
                row, col = ai_1.search(self)
                self.record_2 = []
                self.step(row, col)
            else:
                row, col = ai_2.search(self)
                self.record_1 = []
                self.step(row, col)

        # 判断胜负
        count_1, count_2 = self.get_state()
        winner = 1 if count_1 >= count_2 else -1

        return winner


    def play_with_ai(self, ai):
        '''
            与AI进行一场游戏
            参数：
                None
            返回值:
                None
        '''
        # 游戏初始化
        self.initialize_game()
        # 绘制初始化
        self.initialize_draw()

        # 读取先手方
        player_turn = input('First(1) or second(-1)?')
        while player_turn not in ('1', '-1'):
            player_turn = input('Error Input! Input Again: ')
        player_turn = int(player_turn)

        print("#---------------------- Game Start ----------------------#")
        while self.if_gameover == False:
            # 绘制棋盘
            self.draw()
            
            if self.turn != player_turn:
                # AI执棋
                cur_time = time.time()

                # 获取落子位置
                row, col = ai.search(self)
                if self.turn == 1:
                    self.record_2 = []
                else:
                    self.record_1 = []

                # 落子
                self.step(row, col)

                # 输出记录
                score = self.get_state()
                print('AI turn: {}, score: {}, time cost: {}'.format((row, col), \
                    score, time.time()-cur_time))
            else:
                # 先清空事件队列
                pygame.event.clear()
                while self.turn == player_turn:
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
                                and ((row, col) in self.step_list):
                                self.step(row, col)

                                # 输出记录
                                score = self.get_state()
                                print('Player turn: {}, score: {}'.format((row, col), score))

                                # 跳出等待事件循环
                                break

        self.draw()                      
        # 结束后计算双方的棋子数
        if player_turn == -1:
            ai_count, player_count = self.get_state()
        else:
            player_count, ai_count = self.get_state()
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
        self.step_list = {}
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
                        self.step_list[(i, j)] = flip_list

        return self.step_list


if __name__ == "__main__":
    game = Othello()
    net = PolicyValueNetwork(file_path='model.pkl')
    ai_1 = MCTS(50, 5, net.application)
    ai_2 = MCTS(500, 5, net.application)
    game.play_with_ai(ai_1)
