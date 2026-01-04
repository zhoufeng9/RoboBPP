metric有效性验证实验
| pg          | Feasible Num | Static Stability | Stability Score | Position Offset |
|-------------|--------------|------------------|-----------------|-----------------|
| TAPNet++    | 38.4         | 0.400            | 15.360          | 0.011           |
| TAPNet++_no | 23.6         | 0.413            | 9.750           | 0.006           |
|             |              |                  |                 |                 |
| PCT         | 35.3         | 0.385            | 13.591          | 0.013           |
| PCT_no      | 27.5         | 0.396            | 10.890          | 0.011           |
|             |              |                  |                 |                 |
| PackE       | 35.1         | 0.419            | 14.707          | 0.013           |
| PackE_no    | 27.0         | 0.425            | 11.475          | 0.008           |
|             |              |                  |                 |                 |
|             |              |                  |                 |                 |
|             |              |                  |                 |                 |
| deli        | Feasible Num | Static Stability | Stability Score | Position Offset |
| PCT         | 20.8         | 0.420            | 8.736           | 0.012           |
| PCT_no      | 11.4         | 0.429            | 4.891           | 0.011           |
|             |              |                  |                 |                 |
| TAPNet++    | 9.9          | 0.449            | 4.445           | 0.003           |
| TAPNet++_no | 9.7          | 0.454            | 4.404           | 0.004           |
|             |              |                  |                 |                 |
| PackE       | 7.9          | 0.472            | 3.729           | 0.001           |
| PackE_no    | 7.1          | 0.467            | 3.316           | 0.001           |


介绍：
选取TAPNet++、PCT、PackE三种算法来进行metric有效性验证。model 采用pybullet仿真环境，并以Static Stability和Position Offset这两个指标作为reward训练；model_no 直接采用理想条件来训练。其中稳定性总得分（Stability Score）为Static Stability 对均值乘以Feasible Num，也就是所有放置box的stability累加。

结论：
以我们提出的两个指标引入reward训练，模型会考虑box选取位置的稳定性和偏移距离（进行约束），从而训练出更加稳健的model，最终模型在setting2真实物理仿真环境中的测试结果可摆放的数量明显更多，在保证Static Stability和Position Offset相差不大的情况下增加可摆放的数量。

代码：
test、test-no分别对两类model进行测试，例如cd RobotPackingBenchmark-main-v9后可以直接运行demo.sh进行测试，相关配置在configs文件夹中修改，配置加载的模型路经和测试的数据路经。

train中存放了三种算法的训练文件，参考demo.txt来训练。主要修改如下：PCT、PackE中的bin3D.py中修改step()中的reward，reward = reward + stability_reward * 0.01 + positions_offset * (-1)，同理修改TAPNet++中的env.py中的step()，将Static Stability和Position Offset这两个指标引入reward。stability_reward与positions_offset在仿真环境中得到。
相应的model-no采用不考虑Static Stability和Position Offset这两个指标的原始训练结果（之前setting1理想条件下的训练结果）
