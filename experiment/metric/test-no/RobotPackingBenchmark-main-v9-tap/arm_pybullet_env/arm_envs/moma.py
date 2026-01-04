
import os
import time
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import pybullet as p
# from irb2400_pybullet_env.irb2400_envs import models
# from irb2400_pybullet_env.irb2400_envs import pb_ompl
# from irb2400_pybullet_env.irb2400_envs.grippers import Suction
# from irb2400_pybullet_env.utils import pybullet_utils, utils

from scipy.spatial.transform import Rotation as R

# PLACE_STEP = 0.0003
# PLACE_DELTA_THRESHOLD = 0.005

# IRB2400_URDF_PATH = "irb2400/irb2400.urdf"

class MOMA:

    def __init__(self, ee_type='suction', speed=0.01, offset=[0,0,0], server=None):
        
        self.homej = np.array([-1, -0.7, 0.5, -0.5, -0.5, 0]) * np.pi

        self.homej2 = np.array([-3.81010793, -1.9823318 ,  1.66832743, -1.25692833, -1.57112184, -0.66827464])

        self.offset = offset

        if server is not None:
            self.server = server

    def reset(self, show_gui=True):
        pass

    @property
    def is_static(self):
        return True

    def add_object_to_list(self, category, obj_id):
        pass

    #---------------------------------------------------------------------------
    # Robot Movement Functions
    #---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        
        return True

    def movep(self, pose, speed=0.01):
        
        return self.movej(targj, speed)

    def solve_ik(self, pose, max_num_iterations=100):
        # TODO        
        joints = None
        pass
        return joints

    #============----------------   Move and grasp   ----------------============#
    
    def move_to(self, pose, offset=(0,0,0)):
        """Move end effector to pose.

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        return True

    def pick(self, pose):
        """Move to pose to pick.

        Args:
            pose: SE(3) picking pose.

        Returns:
            pick_success: pick success if True
        """

        offset = (0, 0, 0.1)
        timeout = self.move_to(pose, offset)
        
        if timeout:
            return False

        # Activate end effector, move up, and check picking success.
        self.ee.activate()
        
        postpick_to_pick = ( offset, (0,0,0,1) )
        
        postpick_pose = utils.multiply(pose, postpick_to_pick)

        timeout |= self.movep(postpick_pose, self.speed)

        return pick_success

    def place(self, pose):
        """Move end effector to pose to place.

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        offset = (0, 0, 0.1)
        timeout = self.move_to(pose, offset)
        if timeout:
            return True
        
        self.ee.release()

        postplace_to_place = (offset, (0,0,0,1.0))
        postplace_pose = utils.multiply(pose, postplace_to_place)
        timeout |= self.movep(postplace_pose)

        return timeout

    #============----------------   robot tools   ----------------============#
    
    def get_ee_pose(self):
        pos, orient, *_ = p.getLinkState(self.arm, self.ee_tip)
        return list(pos), list(orient)

    def get_current_joints(self):
        # TODO
        return joints

    def set_joints_state(self, joints):

        # TODO
        pass
        