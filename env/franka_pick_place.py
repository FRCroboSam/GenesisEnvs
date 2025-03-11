import numpy as np
import genesis as gs
import torch
from numpy import random 

class FrankaPickPlaceEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 8  
        self.state_dim = 9 # before it was 6 with 2 args 

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
        default_pos = np.array([0.8, 0.0, 0.2])
        self.target_poses = []
        for _ in range(50):
            offset =np.array([random.rand() * 0.3, random.rand() * 0.5 - 0.5, 0.3 * random.rand() + 0.2])
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
        self.goal_target.set_pos(self.target_poses[self.goal_index], envs_idx=self.envs_idx)
        self.goal_index += 1

        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2 
        state = torch.concat([obs1, obs2, obs2], dim=1) # self.goal_target.get_pos()], dim=1)
    
        
        return state
    
    # state should have format [cube pos, finger pos, target pos]
    #used for calculating reward
    #   
    def get_state_info(self, state):
        # print(state)
        info = {}
        cube_pos = state[0][0:3]
        gripper_pos = state[0][3:6]
        
        grip_diff = self.franka.get_link("right_finger").get_pos() - self.franka.get_link("left_finger").get_pos()
        info["grip_width"] = grip_diff.select(dim=-1, index=1)
        info['cube_distance_to_gripper'] = torch.norm(gripper_pos - cube_pos)
        info['goal_distance'] = torch.norm(self.goal_target.get_pos() - gripper_pos)
        # print(info)
        return info
        
    
    def calculate_reward(self, start_state, end_state):
        #First step, print the correct thing based on the state.
        cube_pos = self.cube.get_pos()
        goal_pos = self.goal_target.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2

        
        # gripped = (torch.norm(block_position - gripper_position, dim=1) < 0.02)  # Close to block
        # lifted = self.cube.get_pos()[:, 2] > 0.1  # Block is lifted
        # placed = torch.norm(self.goal_target.get_pos() - block_position, dim=1) < 0.05  # Block is near goal
        
        reward = 0
        #case 1: block is not close to the gripper -> reward being close to the block, punish closing gripper
        # print("START STATE")
        # print(start_state)
        # print("END STATE")
        # print(end_state)
        if end_state["cube_distance_to_gripper"].squeeze() > 0.02:
            # punish being far away 
            reward = -torch.norm(cube_pos - gripper_position, dim=1) 
            #punish gripper being too close (0.08 is open pos)
            reward -= 0.5 * (6e-2 * end_state['grip_width'])
            
        #case 2: block is within the range of the gripper but not gripped
        elif end_state["cube_distance_to_gripper"].squeeze() <= 0.02 and start_state['grip_width'].squeeze() > 5e-2:
            reward += 2.0  # Encourage reaching the grasping position
            reward += 2.0 * (0.1- torch.abs(end_state['grip_width']) ) # Reward closing when cube is within range
        else: # cube distance to gripper < 0.02 and gripper closed 
            reward += 5.0  # Strong reward for successful grasp
            #reward -= torch.norm(cube_pos - goal_pos, dim=1)  # Encourage moving cube toward the goal
            if torch.norm(end_state['goal_distance'], dim=1) < 0.05:  # If very close to the goal
                reward += 10.0  # Large reward for successful placement
        print(reward)
        return reward
        
        
        
    def step(self, actions):
        block_position = self.cube.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        start_state = torch.concat([block_position, gripper_position, gripper_position]) #self.goal_target.get_pos()], dim=1)
        # start_state_info = self.get_state_info(start_state)
        # actions is a tensor with the action its taking, ie. [5]
        action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 1 # Close gripper
        action_mask_2 = actions == 2 # Lift gripper
        action_mask_3 = actions == 3 # Lower gripper
        action_mask_4 = actions == 4 # Move left
        action_mask_5 = actions == 5 # Move right
        action_mask_6 = actions == 6 # Move forward
        action_mask_7 = actions == 7 # Move backward

        finger_pos = torch.full((self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device)
        finger_pos[action_mask_1] = 0
        finger_pos[action_mask_2] = 0
        
        pos = self.pos.clone()
        pos[action_mask_2, 2] = 0.4
        pos[action_mask_3, 2] = 0
        pos[action_mask_4, 0] -= 0.05
        pos[action_mask_5, 0] += 0.05
        pos[action_mask_6, 1] -= 0.05
        pos[action_mask_7, 1] += 0.05

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )
        
        # move the arm thing
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)
        #FIGURE OUT WHY ITS NOT MOVING PROPERLY 
        # move the gripper
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        block_position = self.cube.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        end_state = torch.concat([block_position, gripper_position, gripper_position], dim=1) #self.goal_target.get_pos()], dim=1)
        # end_state_info = self.get_state_info(end_state)
        #TODO improve this reward function.
        rewards =  -torch.norm(block_position - gripper_position, dim=1) + torch.maximum(torch.tensor(0.02), block_position[:, 2]) * 100
        # rewards = torch.tensor(0) #self.calculate_reward(start_state_info, end_state_info)
        dones = block_position[:, 2] > 0.35
        return end_state, rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = FrankaPickPlaceEnv(vis=True)
    
    
    
    
    
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