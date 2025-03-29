import numpy as np
import genesis as gs
import torch
from numpy import random 


#TODOS TOMORROW 
#   implement replay buffer elements 
#   
class FrankaPickPlaceDDPG_Env:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 4 # end effector x, y, z, finger disp.
        #   gripper pos, block pos, block to gripper, joint displacement of gripper -> if this is bad consider velocities
        self.state_dim = 11 + 3 # 11 for observation, 3 for goal 

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=vis,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04), # block
                pos=(0.65, 0.0, 0.02),
            )
        )
        
        #TODO tomorrow test if this thing can actually have a random position within the box.
        self.goal_target = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                visualization=True,
                collision=False,
                requires_jac_and_IK=False,
                fixed=True,
                radius=0.04
            )
        )
        
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to(self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

        
        #goal_target pos  -> TODO TEST THIS OUT 
        default_pos = np.array([0.7, 0.0, 0])
        self.target_poses = []
        for _ in range(50):
            offset =np.array([random.rand() * 0.2, random.rand() * 0.6 - 0.3, 0.35 * random.rand() + 0.1])
            target_pos = default_pos + offset
            target_pos = np.repeat(target_pos[np.newaxis], self.num_envs, axis=0)
            self.target_poses.append(target_pos)
        self.goal_index = 0
        
        
    # give the sphere a random position
    def reset(self):
        print("RESETTING THE ENVIRONMENT")
        self.build_env()
        # fixed cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        
        
        #seems like these lines cause a lot of lag
        
        #TODO: Use pregenerated points
        goal_pos = self.target_poses[self.goal_index] #np.array([0.7, -0.2, 0.3])    # first one is forward, second one is lateral, third is z (height)
        # goal_pos = np.repeat(goal_pos[np.newaxis], self.num_envs, axis=0)

        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.goal_target.set_pos(goal_pos, envs_idx=self.envs_idx)
        self.goal_index += 1
        obs3 =self.goal_target.get_pos()
        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2 
        return observation
    
    # state should have format [cube pos, finger pos, target pos]
    #used for calculating reward
    #   
    def get_state_info(self, state):
        # print(state.shape)
        # print(state)
        info = {}
        cube_pos = state[0][0:3]
        gripper_pos = state[0][3:6]
        
        grip_diff = self.franka.get_link("right_finger").get_pos() - self.franka.get_link("left_finger").get_pos()
        info["grip_width"] = grip_diff.select(dim=-1, index=1)
        # print("GRIPPER: " + str(gripper_pos))
        # print("CUBE POS: " + str(cube_pos))
        info['cube_distance_to_gripper'] = torch.norm(gripper_pos - cube_pos)
        info['goal_distance'] = torch.norm(self.goal_target.get_pos() - gripper_pos)
        # print(info)
        return info
        
    
    
    
    #ideas if this continues to not work:
    #   make sure the rewards are higher for the in air state than the other ones
        #2 tiered system old one works like the original reward system
        # new one is only based on gripper's position to the goal pos
        # start implementing DPPG based on the paper
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        Sparse reward: 
        - 0 if block is within 0.05m of goal
        - -1 otherwise
        """
        distances = torch.norm(achieved_goal - desired_goal, dim=1)
        rewards = torch.where(distances < 0.05, 
                            torch.zeros_like(distances), 
                            -torch.ones_like(distances))
        return rewards        
     
    def _get_obs(self):
        """Returns observation compatible with HER"""
        left_pos = self.franka.get_link("left_finger").get_pos()
        right_pos = self.franka.get_link("right_finger").get_pos()
        grip_width = (right_pos[0][1] - left_pos[0][1]) / 2  # Half of the grip width
        right_displacement = right_pos[0][1] - grip_width  # Adjust right finger
        left_displacement = left_pos[0][1] + grip_width  # Adjust left finger
        displacement = [right_displacement, left_displacement] 
        return torch.cat([
            (left_pos + right_pos) / 2,  # Gripper pos (3)
            self.cube.get_pos(),          # Block pos (3)
            self.goal_target.get_pos(),   # Goal pos (3)
            [right_pos[0][1] - left_pos[0][1]], # Grip width (1)
        ], dim=1)    
        
    def _check_done(self):
        goal_pos = self.goal_target.get_pos()
        block_position = self.cube.get_pos()
        distance = torch.norm(torch.tensor(block_position) - torch.tensor(goal_pos), p=2)
        return torch.tensor([distance < 0.05], dtype=torch.bool)
        
        
    def step(self, actions):
        # actions is now a continuous tensor of shape [num_envs, 4] in [-1, 1]
        # Dim 0: X movement (-1=left, +1=right)
        # Dim 1: Y movement (-1=backward, +1=forward)
        # Dim 2: Z movement (-1=down, +1=up)
        # Dim 3: Gripper (-1=close, +1=open)
        
        # Scale to real-world units
        delta_pos = actions[:, :3] * 0.05  # 5cm max movement
        gripper_cmd = actions[:, 3]  # [-1, 1]
        
        # Update position continuously
        self.pos += delta_pos
        
        # Continuous gripper control (0=closed, 0.04=open)
        finger_width = (gripper_cmd + 1) * 0.02  # Map [-1,1]→[0,0.04]
        finger_pos = torch.stack([finger_width, finger_width], dim=1)  # Both fingers
        
        # Inverse kinematics
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )
        
        # Execute movements
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()
        
        # Get new state
        new_obs = self._get_obs()
        reward = self._compute_reward(new_obs)
        done = self._check_done(new_obs)
        
        return new_obs, reward, done

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = FrankaPickPlaceDDPG_Env(vis=True)
    
    
    
    
    
# TODOS TMRW:
#   modify the original simulation to show the sphere with random pos
#   if that works modify this script to start testing the reward function
#   GOAL: 
#       have a basic training thing and have the environment correctly show the sphere
#       show up at random locations. ->DONE
#       *BONUS: make sure everything like state, observation space matches the original 
#           start tuning the algorithm

#TODO TMRW: FIGURE OUT HOW TO HAVE THE thing run a lot faster using the extra piece

#in this current code it runs fast without any of the code for goal_target