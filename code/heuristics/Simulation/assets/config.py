import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================
# 
# connection
# 
# ==========================
class Connection:
    
    server_ip = 'localhost'

    heuristic_port = 12345

    # tap_port = 12344
    # detect_port = 22344

    # tap_port = 32344
    # detect_port = 42344
    tap_port_dict = {
        'INIT': 22335,
        'CNN': 32335,
        'ATTN': 43225,
        'RNN': 53225,
        
        'CNN2': 34335,
        'ATTN2': 44225,
        'RNN2': 54225,

        'CNN3': 14335,

        'HER_vol': 50225,
        'HER_rew': 64225,

        'ROS': 9002,

        'ATTN3': 24225,

        # 'CNN2': 34337,
        # 'ATTN2': 44227,
        # 'RNN2': 54227,
    }
    detect_port_dict = {
        'INIT': 22334,
        'CNN': 32334,
        'ATTN': 43224,
        'RNN': 53224,

        # 'CNN2': 34336,
        # 'ATTN2': 44226,
        # 'RNN2': 54226,

        'CNN2': 34334,
        'ATTN2': 44224,
        'RNN2': 54224,

        'CNN3': 14334,
        
        'HER_vol': 50224,
        'HER_rew': 64224,

        'ROS': 9000,

        'ATTN3': 24224,
    }


image_path = "/home/ubuntu/ws/TAP_app/TAP-all"

# ==========================
# 
# blocks and container
# 
# ==========================
class TAP:
    # 机器人位移
    plane_offset = [-0, 0, -0.001]


    unit_len = 1
    # unit_len = 10

    blocks_num = 10
    unit_height = 0.03
    # unit_height = 0.05
    # unit_height = 0.04

    # 初始生成一堆立方体的容器大小和位置
    init_container_size = [ 7, 7, 50 ] # width, length, height

    # 数量
    # blocks_num = 20
    # unit_height = 0.02
    # init_container_size = [ 10, 10, 50 ] # width, length, height

    # blocks_num = 30
    # unit_height = 0.02
    # init_container_size = [ 12, 12, 50 ] # width, length, height

    # 目标容器的位置和大小
    # target_width = 10
    target_width = 5
    # target_origin = np.array([0.0, 0.6, 0])
    # target_origin = np.array([0.0, 0.4, 0])
    # target_origin = np.array([0.35, -target_width / 2.0 * unit_height - 0.03, 0])


    if target_width == 5:
        x = 0.35
    else:
        x = 0.3
        
    target_origin = np.array([x, 0.28, 0])
    target_origin = np.array([x, 0.28, 0.05])
    target_origin = np.array([x, 0.28, 0.025])
    target_origin = np.array([x, 0.28, 0.025])
    
    # target_origin = np.array([x, -target_width / 2.0 * unit_height, 0])

    target_base_size = np.array([ target_width * unit_height, target_width * unit_height, 0.02])
    target_container_size = np.array([target_width, target_width, 50])


    # init_container_origin = [-0.25, 0.1, 0.2]
    init_container_origin = [-0.25, 0.1, 0.25]
    
    wait_postition = [0.25, 0.1, 0.3]
    wait_postition = [0.3, 0.1, 0.8]
    wait_quaternion = list( R.from_euler('XYZ', [0, -np.pi/2, 0]).as_quat() )
    
    # 聚齐物体后先放到一个中间位置进行第二次物体大小识别
    tmp_position = np.array([0.5, 0.2, 0.15])
    preset_tmp_pos = np.array((0.5520267130164717, 0.24386956319982722, 0.045988591668631944))
    
    # 临时摆放物体的候选位置，如果抓取的物体大小是错误的，而且更新物体大小后
    # 去重新让TAP-Net选择的物体不是当前抓取物体的情况下，
    # 我们得把当前抓取的物体摆放到一个候选位置等下一次选到它
    free_position = np.array([0.5, 0.5, 0.0])
    free_block = None
    free_block_pose = None
    
    # 无法摆放的物体扔掉的位置，或者直接代码删除该物体也好
    remove_position = np.array([0.11, -0.4, 0])

    # wall_height = 0.1
    wall_width = 0.005
    wall_height = (unit_height * 5 + 0.01)
    
    # 长度范围
    block_min_size, block_max_size = 1, 4
    # block_min_size, block_max_size = 20, 70

    # block_gap = 0.003
    # block_gap = 0.004 #* globalScaling
    block_gap = 0.004 #* globalScaling

    # 一个用于生成立方体的变量，可忽视
    loop_num = 1

    data_type = 'mess' # mess or neat
    # net_type = 'cnn'

    @property
    def is_mess(self):
        return self.data_type == 'mess'
    
    def data_folder(self):
        return "./data/%d_[%d]_[%d_%d]_%03d_%s_real" % (
            self.blocks_num, self.init_container_size[0],
            self.block_min_size, self.block_max_size,
            self.unit_height * 100, self.data_type
        )
        # return "./data/%d_[%d_%d]_%03d_%s_scale" % (
        #     self.blocks_num,
        #     self.block_min_size, self.block_max_size,
        #     self.unit_height * 100, self.data_type
        # )

        # return "./data/%d_[%d_%d]_%03d_unit_%d" % (
        #     self.blocks_num,
        #     self.block_min_size, self.block_max_size,
        #     self.unit_height * 100, self.unit_len
        # )
        # return "./data/mess_real"

    # unit_length to tolerant
    tolerant_nums = {
        20: 0,
        10: 0,
        5: 0,
        1: 0,
        # 5: 1,
        # 1: 4,
    }
    
# ==========================
# 
# task
# 
# ==========================
class Task:
    debug = False

    TRANSPORT_TYPES = [
        'tap',
        # 'rand',
        # 'precedence',
        # 'area',
    ]

    PRECEDENCE_TYPES = [
        'norm',
        # 'hard', 
        # 'soft'
    ]


    tap_type = 'none'

    save_result = True

    task_num = 1000
    # task_num = 1

    is_spawn = False
    

