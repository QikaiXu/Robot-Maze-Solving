from QRobot import QRobot

"""
题目要求:  编程实现 DQN 算法在机器人自动走迷宫中的应用
输入: 由 Maze 类实例化的对象 maze
必须完成的成员方法：train_update()、test_update()
补充：如果想要自定义的参数变量，在 \_\_init\_\_() 中以 `self.xxx = xxx` 创建即可
"""

class Robot(QRobot):
    
    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        self.maze = maze

    def train_update(self):
        """ 
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """

        action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------

        # -----------------------------------------------------------------------

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """

        action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------

        # -----------------------------------------------------------------------

        return action, reward
