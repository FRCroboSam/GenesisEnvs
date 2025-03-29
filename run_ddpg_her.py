import argparse
import genesis as gs
import torch
from algo.dqn_agent import DQNAgent
from algo.ddpg_her_agent import DDPG_HER_AGENT
from env import *
import os

gs.init(backend=gs.gpu, precision="32")

task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv,
    'WaterFranka': WaterFrankaEnv,
    'ShadowHandBase': ShadowHandBaseEnv,
    'FrankaPickPlace': FrankaPickPlaceEnv,
    'FrankaPickPlaceDDPGHer': FrankaPickPlaceDDPG_Env
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]  
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")


def train_dqn(args):
    if args.load_path == "default":
        load = True
        checkpoint_path = f"logs/{args.task}_dqn_checkpoint_released.pth"
    elif args.load_path: 
        load = True
        checkpoint_path = args.load_path
    else:
        load = False
        checkpoint_path = f"logs/{args.task}_dqn_checkpoint.pth"
    print("ARGS LOAD IS: " + str(load))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print("CREATING AN ENV")
    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created environment: {env}")
    
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    replay_size = args.replay_size if args.replay_size else max(100000, 10 * batch_size) 
    agent = DDPG_HER_AGENT(state_dim=env.state_dim, output_dim=env.action_space, goal_dim=env.goal_dim, lr=1e-3, gamma=0.99, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.01, \
                     device=args.device, load=load, num_envs=args.num_envs, hidden_dim=args.hidden_dim, \
                        checkpoint_path=checkpoint_path, batch_size=batch_size, replay_size=replay_size)
    if args.device == "mps":
        print("RUNNING THE AGENT with mps")
        gs.tools.run_in_another_thread(fn=run, args=(env, agent))
        env.scene.viewer.start()
    else:
        print("RUNNING THE AGENT")
        run(env, agent)



#TODO CHANGE THIS TO BE LIKE REPLAY BUFFER ALGO 
def run(env, agent):
    best_reward = -float('inf')
    num_episodes = 500
    target_update_interval = 10

    for episode in range(num_episodes):
        # 1. Sample a goal and initial state
        state, goal = env.reset()  # Modified to return goal
        episode_transitions = []  # Stores (s_t, a_t, s_{t+1}, done) for the episode

        # 2. Run episode (NO TRAINING HERE)
        for t in range(50):
            action = agent.select_action(np.concatenate([state, goal]))  # π_b(s_t || g)
            next_state, _, done = env.step(action)  # Original reward ignored (recomputed later)
            
            # Store transition (without reward, since it depends on the goal)
            episode_transitions.append((state.copy(), action, next_state.copy(), done))
            
            if done.all():
                break

        # 3. Process transitions and store in replay buffer (with original + hindsight goals)
        for t, (state, action, next_state, done) in enumerate(episode_transitions):
            # (A) Store original transition (s_t || g, a_t, r_t, s_{t+1} || g)
            original_reward = agent.reward_fn(state, action, goal)
            agent.memory.add(
                torch.cat([state, goal]),
                action,
                original_reward,
                torch.cat([next_state, goal]),
                done
            )

            # (B) HER: Sample additional goals and store relabeled transitions
            additional_goals = agent.sample_goals(episode_transitions, t)  # Strategy S (e.g., future states)
            for g_prime in additional_goals:
                relabeled_reward = agent.reward_fn(state, action, g_prime)
                relabeled_done = agent.is_done(next_state, g_prime)  # Check if g_prime achieved
                
                agent.memory.add(
                    torch.cat([state, g_prime]),
                    action,
                    relabeled_reward,
                    torch.cat([next_state, g_prime]),
                    relabeled_done
                )

        # 4. Train N steps on random minibatches (AFTER storing all transitions)
        for _ in range(len(episode_transitions)):  # Or some fixed number of updates
            agent.train()  # Samples from replay buffer (includes original + hindsight transitions)

        # 5. Periodically update target network & save
        if episode % target_update_interval == 0:
            agent.update_target_network()
            agent.save_checkpoint()
        print(f"Episode {episode}, Total Reward: {total_reward}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load_path", type=str, nargs='?', default=None, help="Path for loading model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-r", "--replay_size", type=int, default=None, help="Size of replay buffer for DQN")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64, help="Hidden dimension for the network")
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_dqn(args)
