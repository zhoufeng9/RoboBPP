# detect the block in scene / image, then compute 3D pose and size

from detector.cube_matcher import block_match


class Detector():
    # TODO
    def __init__(self) -> None:
        # 用于二次检测的，取吸盘下方的像素点，判断是不是箱子
        # self.tmp_pos = [220, 280]
        self.tmp_pos = [130, 300]

    def img_segment(self, rgb, dep, idmap):

        pass

    def detect_blocks(self, rgb, dep, seg, cam_intrinsic, cam_to_world, max_block_size, block_unit, is_tmp=False, tolerant_num=0):
        """
        Args:
            rgb: np.array [w, h, 3]
            dep: np.array [w, h]
            cam_intrinsic: np.array [3, 3]
            cam_to_world: np.array [4, 4]
            block_unit: float

        Returns:
            block_poses: List[ np.array [4, 4] ]
            block_sizes: List[ np.array [3] ]
        """
        # state = 'DEP-CUBE'

        # mask = self.segers.segment( rgb, dep )

        # 直接用pybullet的分割结果，当前场景下箱子的id都是大于等于8的
        first_box_id = 8
        seg[seg < first_box_id] = 0
        mask = seg  # 得到掩码，即对应的id

        if is_tmp:
            # 二次检测的，只匹配吸取的箱子，取吸盘下方的像素点，判断是不是箱子
            x, y = self.tmp_pos
            label = seg[x,y]
            seg[seg != label] = 0
        
        ret = block_match(mask, dep, cam_intrinsic, cam_to_world, max_block_size, block_unit)

        blocks_pose, blocks_size, blocks_mask = ret[0], ret[1], ret[2]

        for i in range(len(blocks_size)):
            # 加上一定的扩大长度
            blocks_size[i][0] += tolerant_num * block_unit
            blocks_size[i][1] += tolerant_num * block_unit
        
        return blocks_pose, blocks_size, blocks_mask

    def close(self):
        self.client.close([0, 0, "STOP"] )
        