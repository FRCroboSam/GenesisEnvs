import numpy as np
import argparse

import genesis as gs
#reference: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/inverse_kinematics_motion_planning.html
########################## create a scene ##########################
#ACTUALLY FINAL TASK IS THIS: https://robotics.farama.org/envs/fetch/pick_and_place/
#TODO: Implement this reinforcement learning: https://www.youtube.com/watch?v=XF7QoENL5I8
# or this one: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/locomotion.html
# scene = gs.Scene(
#     viewer_options = gs.options.ViewerOptions(
#         camera_pos    = (0, -3.5, 2.5),
#         camera_lookat = (0.0, 0.0, 0.5),
#         camera_fov    = 30,
#         res           = (960, 640),
#         max_FPS       = 60,
#     ),
#     sim_options = gs.options.SimOptions(
#         dt = 0.01,
#     ),
#     show_viewer = True,
# )

# ########################## entities ##########################
# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )
# franka = scene.add_entity(
#     gs.morphs.MJCF(
#         file  = 'xml/franka_emika_panda/panda.xml',
#     ),
# )
    
# jnt_names = [
#     'joint1',
#     'joint2',
#     'joint3',
#     'joint4',
#     'joint5',
#     'joint6',
#     'joint7',
#     'finger_joint1',
#     'finger_joint2',
# ]
# dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

############ Optional: set control gains ############
# scene = gs.Scene(
#     show_viewer    = True,
#     viewer_options = gs.options.ViewerOptions(
#         res           = (1280, 960),
#         camera_pos    = (3.5, 0.0, 2.5),
#         camera_lookat = (0.0, 0.0, 0.5),
#         camera_fov    = 40,
#         max_FPS       = 60,
#     ),
#     vis_options = gs.options.VisOptions(
#         show_world_frame = True, # visualize the coordinate frame of `world` at its origin
#         world_frame_size = 1.0, # length of the world frame in meter
#         show_link_frame  = False, # do not visualize coordinate frames of entity links
#         show_cameras     = False, # do not visualize mesh and frustum of the cameras added
#         plane_reflection = True, # turn on plane reflection
#         ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
#     ),
#     renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
# )
# ########################## entities ##########################
# plane = scene.add_entity(gs.morphs.Plane())
# r0 = scene.add_entity(
#     gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
# )
jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
def main():
    gs.init(backend=gs.cpu)
    # scene = gs.Scene(
    #     viewer_options = gs.options.ViewerOptions(
    #         camera_pos    = (0, -3.5, 2.5),
    #         camera_lookat = (0.0, 0.0, 0.5),
    #         camera_fov    = 30,
    #         res           = (960, 640),
    #         max_FPS       = 60,
    #     ),
    #     sim_options = gs.options.SimOptions(
    #         dt = 0.01,
    #     ),
    #     show_viewer = True,
    # )
    scene = gs.Scene(
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer    = True,
        viewer_options = gs.options.ViewerOptions(
            res           = (1280, 960),
            camera_pos    = (3.5, 0.0, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 40,
            max_FPS       = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True, # visualize the coordinate frame of `world` at its origin
            world_frame_size = 1.0, # length of the world frame in meter
            show_link_frame  = False, # do not visualize coordinate frames of entity links
            show_cameras     = False, # do not visualize mesh and frustum of the cameras added
            plane_reflection = True, # turn on plane reflection
            ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
        ),
        renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size = (0.04, 0.04, 0.04),
            pos  = (0.65, 0.0, 0.02),
        )
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file  = 'xml/franka_emika_panda/panda.xml',
        ),
    )
        
    scene.build()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # set control gains
    # Note: the following values are tuned for achieving best behavior with Franka
    # Typically, each new robot would have a different set of parameters.
    # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, franka, args.vis))

    scene.viewer.start()
    # set positional gains
    # franka.set_dofs_kp(
    #     kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    #     dofs_idx_local = dofs_idx,
    # )
    # # set velocity gains
    # franka.set_dofs_kv(
    #     kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    #     dofs_idx_local = dofs_idx,
    # )
    # # set force range for safety
    # franka.set_dofs_force_range(
    #     lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    #     upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    #     dofs_idx_local = dofs_idx,
    # )

# def run_sim(scene, enable_vis):
#     from time import time

#     t_prev = time()
#     i = 0
#     while True:

#         i += 1

#         scene.step()

#         t_now = time()
#         print(1 / (t_now - t_prev), "FPS")
#         t_prev = t_now
#     if enable_vis:
#         scene.viewer.stop()




def run_sim(scene, franka, enable_vis):
    
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]    

    from time import time

    t_prev = time()
    print("ruNNING SIMMMMMMM")
    i = 0
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    while(True):
        # get the end-effector link
        end_effector = franka.get_link('hand')

        # move to pre-grasp pose
        qpos = franka.inverse_kinematics(
            link = end_effector,
            pos  = np.array([0.65, 0.0, 0.25]),
            quat = np.array([0, 1, 0, 0]),
        )
        # gripper open pos
        # release gripper
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        qpos[-2:] = 0.04
        
        path = franka.plan_path(
            qpos_goal     = qpos,
            num_waypoints = 200, # 2s duration
        )
        # execute the planned path
        for waypoint in path:
            franka.control_dofs_position(waypoint)
            scene.step()

        # allow robot to reach the last waypoint
        for i in range(100):
            scene.step()
       # reach
        qpos = franka.inverse_kinematics(
            link = end_effector,
            pos  = np.array([0.65, 0.0, 0.130]),
            quat = np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        for i in range(100):
            scene.step()

        # grasp
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

        for i in range(100):
            scene.step()

        # lift
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.28]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        

        for i in range(200):
            scene.step()



if __name__ == "__main__":
    main()