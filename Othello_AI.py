#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.29
#   Intro: 黑白棋人机游戏AI
#####################################


# 导入相关模块
import numpy as np
import time
import random


class Othello_AI(object):
    def __init__(self, board, valid_board):
        '''
            初始化函数
            参数：
                board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
            返回值：
                None
        '''
        super().__init__()
        self.board = board
        self.board_size = len(board)
        self.valid_board = valid_board


    def search(self):
        '''
            搜索函数
            参数：
                None
            返回值：
                (row, col): 选择的下一步棋
        '''
        time.sleep(2)
        # 此时轮到AI执棋
        self.turn = -1
        # 获取落子列表
        step_list = self.get_steps()
        # 随机返回一个
        return random.sample(step_list.keys(), 1)[0]


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

                    # 向右上探索
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