# 导入相关包 
import os
import random
import numpy as np
import torch
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt


def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []
    # -----------------请实现你的算法代码--------------------------------------
    visited = {}
    path_t = []
    direction = ['u', 'r', 'd', 'l']
    
    
    def reverse(d):
        idx = direction.index(d)
        return direction[(idx + 2) % 4]
    
    
    def dfs():
        nonlocal path
        if path:
            return
        location = maze.sense_robot()
        if location == maze.destination:
            path = [_ for _ in path_t]
        
        visited[location] = True
        for d in maze.can_move_actions(location):
            maze.move_robot(d)
            location = maze.sense_robot()
            if location not in visited:
                path_t.append(d)
                dfs()
                del path_t[-1]
            maze.move_robot(reverse(d))
            
            
    dfs()
    # -----------------------------------------------------------------------
    return path


class Robot(TorchRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 10.,
            "destination": -maze.maze_size ** 2 * 4.,
            "default": 1.,
        })
        self.maze = maze
        self.epsilon = 0
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        self.loss_list = self.train()
        

    def train(self):
        loss_list = []
        batch_size = len(self.memory)
        
        # 训练，直到能走出这个迷宫
        while True:
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            success = False
            self.reset()
            for _ in range(self.maze.maze_size ** 2 - 1):
                a, r = self.test_update()
            #     print("action:", a, "reward:", r)
                if r == self.maze.reward["destination"]:
                    return loss_list
            

        
    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        
        # batch_size = len(self.memory)
        # for _ in range(10):
        #     self._learn(batch=batch_size)
        
        

        """---update the step and epsilon---"""
        # self.epsilon = max(0.01, self.epsilon * 0.995)

        return action, reward
    
    
    def test_update(self):
        state = np.array(self.sense_state(), dtype=np.int16)
        state = torch.from_numpy(state).float().to(self.device)

        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()

        action = self.valid_action[np.argmin(q_value).item()]
        reward = self.maze.move_robot(action)
        return action, reward

