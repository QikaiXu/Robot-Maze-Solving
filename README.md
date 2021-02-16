# Robot-Maze-Solving
>浙江大学《人工智能与系统》课程作业，机器人走迷宫。

深搜和宽搜没什么好说的，这里主要是用 DQN 实现。

关于这个项目的描述可以查看 `main.ipynb`，然后我实现的机器人在 `my_robot.ipynb` 中。

这里我的机器人没有采用探索的策略，而是直接获取整个世界的信息，并从所有数据中直接开始学习。

修改了原有的 `torch_py/QNetwork.py` 中的 QNetwork 类的网络结构，将原有的 state_size -> 512 -> 512 -> action_size 改成了 state_size -> 128 -> 64 -> action_size。

修改奖励策略，将终点奖励设置为和迷宫大小相关，否则在过大的迷宫中，机器人不一定能「看得到」太远的终点。

训练结果如下图所示：

<img src="/Users/xuqikai/Documents/GitHub/Robot-Maze-Solving/images/image-20210216211011077.png" alt="image-20210216211011077" style="zoom:50%;" />

