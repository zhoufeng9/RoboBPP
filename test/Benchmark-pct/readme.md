python 3.7 跑的

python main.py

* pybullet中 irb2400 的路径规划使用的是预编译好的 ompl，目前在python 3.7下试了
> https://github.com/lyfkyle/pybullet_ompl

机器人控制在 ：irb2400_pybullet_env/irb2400_envs/envs.py

detector 加识别模块，这里直接使用pybullet的segment mask

packer 做装箱决策算法