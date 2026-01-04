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

"""PyBullet utilities for loading assets."""
import os
import time

import pybullet as p

import tempfile
import string
import random
import numpy as np

from scipy.spatial.transform import Rotation as R


def scale_pos(pos, scale):
    ret = []

    for p in pos:
        ret.append(p * scale)
    return ret

# BEGIN GOOGLE-EXTERNAL
def load_urdf(pybullet_client, file_path, *args, **kwargs):
  """Loads the given URDF filepath."""
  # Handles most general file open case.
  try:
    return pybullet_client.loadURDF(file_path, *args, **kwargs)
  except pybullet_client.error:
    pass


# END GOOGLE-EXTERNAL

def simulate_step(step_num, sleep_time=0):
  for _ in range(step_num):
    p.stepSimulation()  
    if sleep_time > 0:
      time.sleep(sleep_time)


def draw_pose(pose:list, length=0.05, width=0.03, life_time=0):
  return
  pos, quat = pose
  
  rot = R.from_quat(quat)

  mat = rot.as_dcm()

  x = mat[:,0] * length
  y = mat[:,1] * length
  z = mat[:,2] * length
  
  p.addUserDebugLine( pos, pos+x, [1,0,0], width, lifeTime=life_time  )
  p.addUserDebugLine( pos, pos+y, [0,1,0], width, lifeTime=life_time  )
  p.addUserDebugLine( pos, pos+z, [0,0,1], width, lifeTime=life_time  )


def key_event(keys):
  # keys: p.getKeyboardEvents()

  def check_key(key):
      '''
      Args:
          - `key`: char
              * the key you enter
      
      Returns:
          - `the_key_press`: boolean
      '''
      
      key = ord(key)
      return key in keys and keys[key] & p.KEY_WAS_TRIGGERED
  
  if check_key('p'):
    print("")
    print("")
    print("-- pybullet key map --")
    print("    [g] to switch the gui")
    print("    [v] to hide scene")
    print("    [s] to switch light/shadow")
    print("    [p] to print these text")
    print("")
    

def fill_template(assets_root, template, replace):
  """Read a file and replace key strings."""
  full_template_path = os.path.join(assets_root, template)
  with open(full_template_path, 'r') as file:
    fdata = file.read()
  for field in replace:
    for i in range(len(replace[field])):
      fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
  alphabet = string.ascii_lowercase + string.digits
  rname = ''.join(random.choices(alphabet, k=16))
  tmpdir = tempfile.gettempdir()
  template_filename = os.path.split(template)[-1]
  fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
  with open(fname, 'w') as file:
    file.write(fdata)
  return fname

def color_random(color):
  shade = np.random.rand() + 0.5
  color = np.float32([shade * color[0] * 255, shade * color[1] * 255, shade * color[2] * 255, 255]) / 255
  return color

def pose_to_mat(pose):
  pos, quat = pose
  mat = np.identity(4)

  # rot = R.from_quat(quat)
  # mat[:3,:3] = rot.as_matrix()
  mat[:3,:3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
  mat[:3,3] = pos
  return mat

def mat_to_pose(mat):
  pos = mat[:3,3]
  rot = R.from_matrix(mat[:3,:3])
  quat = rot.as_quat()
  return [pos, quat]

def get_link_pose(body_id, link_id):
    
    link_info = p.getLinkState(body_id, link_id)
    link_pos = np.array(link_info[0])
    link_quat = np.array(link_info[1])
    return [link_pos, link_quat]

def mask_coll_off( obj_id, link_id=-1 ):
  collisionFilterGroup = 0
  collisionFilterMask = 0
  # p.setCollisionFilterGroupMask(cubeId, -1, collisionFilterGroup, collisionFilterMask)
  p.setCollisionFilterGroupMask( obj_id, link_id, collisionFilterGroup, collisionFilterMask)

def mask_coll_on( obj_id, link_id=-1 ):
  collisionFilterGroup = 1
  collisionFilterMask = 1
  p.setCollisionFilterGroupMask( obj_id, link_id, collisionFilterGroup, collisionFilterMask)