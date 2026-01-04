# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera configs."""

import numpy as np
import pybullet as p

from scipy.spatial.transform import Rotation as R

class RealSenseD415():
  """Default configuration with 3 RealSense RGB-D cameras."""

  # Mimic RealSense D415 RGB-D camera parameters.
  image_size = (480, 640)
  intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

  # Set default camera poses.
  front_position = (1., 0, 0.75)
  front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
  front_rotation = p.getQuaternionFromEuler(front_rotation)
  left_position = (0, 0.5, 0.75)
  left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
  left_rotation = p.getQuaternionFromEuler(left_rotation)
  right_position = (0, -0.5, 0.75)
  right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
  right_rotation = p.getQuaternionFromEuler(right_rotation)

  result_position = (0.5, 0.5, 0.75)
  result_rotation = (np.pi / 4, 4 * np.pi / 4, -1 * np.pi / 4)
  result_rotation = p.getQuaternionFromEuler(result_rotation)

  # Default camera configs.
  CONFIG = [{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': front_position,
      'rotation': front_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': left_position,
      'rotation': left_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': right_position,
      'rotation': right_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  },{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': result_position,
      'rotation': result_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  },
  ]

class Camera:
    def __init__(self, config, random_state:np.random.RandomState = None) -> None:
        self.config = config
        self.random_state = random_state

    def take_images(self):
        """Render RGB-D image with specified camera configuration."""

        config = self.config
        random_state = self.random_state

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (config['image_size'][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # Note when use_egl is toggled, this option will not actually use openGL
            # but EGL instead.
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (config['image_size'][0], config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        if config['noise']:
            color = np.int32(color)
            color += np.int32(random_state.normal(0, 3, config['image_size']))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth

        if config['noise']:
            depth += random_state.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm
    
    def get_intrinsics(self):
        intrinsics = self.config['intrinsics']
        intrinsics = np.array(intrinsics).reshape((3,3))
        return intrinsics
    
    def get_pose(self):
        pos = self.config['position']
        rot = self.config['rotation']

        # w, h, view, proj, up, forward, hori, ver, yaw, pitch, dist, target = p.getDebugVisualizerCamera()
        # pos = target - np.array(forward) * dist

        rot = R.from_quat(rot)
        mat = rot.as_matrix()

        pose = np.identity(4)
        pose[:3,:3] = mat
        pose[:3,3] = pos

        return pose