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

"""Classes to handle gripper dynamics."""

import os

import numpy as np
from arm_pybullet_env.utils import pybullet_utils
from scipy.spatial.transform import Rotation as R

import pybullet as p

SPATULA_BASE_URDF = 'irb6700/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'irb6700/suction/suction_base.urdf'
SUCTION_HEAD_URDF = 'irb6700/suction/suction_head.urdf'
# SUCTION_BASE_URDF = 'ur5/suction/suction_base_scale.urdf'
# SUCTION_HEAD_URDF = 'ur5/suction/suction_head_scale.urdf'


class Gripper:
  """Base gripper class."""

  def __init__(self, assets_root):
    self.assets_root = assets_root
    self.activated = False

  def step(self):
    """This function can be used to create gripper-specific behaviors."""
    return

  def activate(self, objects):
    del objects
    return

  def release(self):
    return


class Spatula(Gripper):
  """Simulate simple spatula for pushing."""

  def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
    """Creates spatula and 'attaches' it to the robot."""
    super().__init__(assets_root)

    # Load spatula model.
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))


class Suction(Gripper):
  """Simulate simple suction dynamics."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    """Creates suction and 'attaches' it to the robot.

    Has special cases when dealing with rigid vs deformables. For rigid,
    only need to check contact_constraint for any constraint. For soft
    bodies (i.e., cloth or bags), use cloth_threshold to check distances
    from gripper body (self.body) to any vertex in the cloth mesh. We
    need correct code logic to handle gripping potentially a rigid or a
    deformable (and similarly for releasing).

    To be clear on terminology: 'deformable' here should be interpreted
    as a PyBullet 'softBody', which includes cloths and bags. There's
    also cables, but those are formed by connecting rigid body beads, so
    they can use standard 'rigid body' grasping code.

    To get the suction gripper pose, use p.getLinkState(self.body, 0),
    and not p.getBasePositionAndOrientation(self.body) as the latter is
    about z=0.03m higher and empirically seems worse.

    Args:
      assets_root: str for root directory with assets.
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all suctionable objects in the env.
    """
    super().__init__(assets_root)

    # Load suction gripper base model (visual only).
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))

    self.base_urdf = os.path.join(self.assets_root, SUCTION_BASE_URDF)
    self.base = pybullet_utils.load_urdf(p, self.base_urdf, pose[0], pose[1])
    base_constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))

    # Load suction tip model (visual and collision) with compliance.
    # urdf = 'assets/irb2400/suction/suction-head.urdf'
    pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))

    self.body_urdf = os.path.join(self.assets_root, SUCTION_HEAD_URDF)
    self.body = pybullet_utils.load_urdf(p, self.body_urdf, pose[0], pose[1])
    head_constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.08))
        # childFramePosition=(0, 0, -0.12))
    p.changeConstraint(head_constraint, maxForce=50)

    # self.constraints = [ world_constraint, base_constraint, head_constraint ]
    self.constraints = [ base_constraint, head_constraint ]

    # Reference to object IDs in environment for simulating suction.
    self.obj_ids = obj_ids

    self.ee = ee

    self.attach_obj_ids = None
    self.obj_to_body = None

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Defaults for deformable parameters, and can override in tasks.
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    self.def_threshold = 0.030
    self.def_nb_anchors = 1

    # Track which deformable is being gripped (if any), and anchors.
    self.def_grip_item = None
    self.def_grip_anchors = []

    # Determines release when gripped deformable touches a rigid/def.
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self):
    """Simulate suction using a rigid fixed constraint to contacted object."""
    # TODO(andyzeng): check deformables logic.
    # del def_ids

    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
      # print(points)
      if points:

        # Handle contact between suction with a rigid object.
        for point in points:
          obj_id, contact_link = point[2], point[4]
        if obj_id in self.obj_ids['rigid']:
          body_pose = p.getLinkState(self.body, 0)
          obj_pose = p.getBasePositionAndOrientation(obj_id)
          world_to_body = p.invertTransform(body_pose[0], body_pose[1])
          obj_to_body = p.multiplyTransforms(world_to_body[0],
                                             world_to_body[1],
                                             obj_pose[0], obj_pose[1])
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=0,
              childBodyUniqueId=obj_id,
              childLinkIndex=contact_link,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=obj_to_body[0],
              parentFrameOrientation=obj_to_body[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))
          
          self.attach_obj_ids = obj_id
          self.obj_to_body = obj_to_body
          
        self.activated = True

  def release(self):
    """Release gripper object, only applied if gripper is 'activated'.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    To handle deformables, simply remove constraints (i.e., anchors).
    Also reset any relevant variables, e.g., if releasing a rigid, we
    should reset init_grip values back to None, which will be re-assigned
    in any subsequent grasps.
    """
    if self.activated:
      self.activated = False
      self.attach_obj_ids = None
      self.obj_to_body = None
      
      # Release gripped rigid object (if any).
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass
        self.init_grip_distance = None
        self.init_grip_item = None

      # Release gripped deformable object (if any).
      if self.def_grip_anchors:
        for anchor_id in self.def_grip_anchors:
          p.removeConstraint(anchor_id)
        self.def_grip_anchors = []
        self.def_grip_item = None
        self.def_min_vetex = None
        self.def_min_distance = None

  def detect_contact(self):
    """Detects a contact with a rigid object."""
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    # print(points)
    # exit()
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False

  def check_grasp(self):
    """Check a grasp (object in contact?) for picking success."""

    suctioned_object = None
    if self.contact_constraint is not None:
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
    return suctioned_object is not None

  def reset_constraints(self, robot):
    for constraint in self.constraints:
      p.removeConstraint(constraint)
    
    base_constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=self.ee,
        childBodyUniqueId=self.base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))

    head_constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=self.ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.08))
    p.changeConstraint(head_constraint, maxForce=50)

    self.constraints = [ base_constraint, head_constraint ]
  
  def get_links_dict(self, robot_id):
      
      gripper_id = robot_id

      link_name_to_index = {p.getBodyInfo(gripper_id)[0].decode('UTF-8'):-1,}
      joint_name_to_index = {}
      joint_name_to_parent = {}

      for _id in range(p.getNumJoints(gripper_id)):
          joint_info = p.getJointInfo(gripper_id, _id)
          joint_index = joint_info[0]
          joint_name = joint_info[1].decode('UTF-8')
          joint_parent_link_index = joint_info[-1]

          joint_name_to_index[joint_name] = joint_index
          joint_name_to_parent[joint_name] = joint_parent_link_index

          link_name = joint_info[12].decode('UTF-8')
          link_name_to_index[link_name] = _id
      
      self.joint_name_to_index = joint_name_to_index
      self.joint_name_to_parent = joint_name_to_parent
      self.link_name_to_index = link_name_to_index

      return joint_name_to_index, joint_name_to_parent, link_name_to_index

  def get_joint_pose(self, robot_id, joint_id):
      joint_info = p.getJointInfo(robot_id, joint_id)
      
      pos_in_parent = joint_info[-3]
      # quat_in_parent = joint_info[-2]
      link_id = joint_info[-1]

      link_pos, link_quat = self.get_link_pose(robot_id, link_id)

      joint_m = np.identity(4)
      link_m = np.identity(4)

      joint_m[:3,3] = pos_in_parent
      link_m[:3,3] = link_pos

      link_m[:3,:3] = R.from_quat(link_quat).as_matrix()

      joint_m = link_m @ joint_m

      joint_pos = joint_m[:3, 3]
      joint_quat = R.from_matrix(joint_m[:3,:3]).as_quat()
      return [joint_pos, joint_quat]
  
  def get_link_pose(self, robot_id, link_id):

      return pybullet_utils.get_link_pose(robot_id, link_id)