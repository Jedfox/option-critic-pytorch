import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import argparse
import torch
from copy import deepcopy

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from soft_option_critic import SoftOptionCriticFeatures, SoftOptionCriticConv
from soft_option_critic import critic_loss as critic_loss_Soft_fn
from soft_option_critic import actor_loss as actor_loss_Soft_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import Logger
from datetime import datetime
from collections import deque

import time

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--model', default='option_critic', help='Model to use: option_critic | soft_option_critic') 
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='scratch/runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')


def run(args):
    env, is_atari = make_env(args.env)
    is_atari = 'NoFrameskip' in args.env or 'ALE/' in args.env or 'Seaquest' in args.env
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    soft_option_critic = SoftOptionCriticConv if is_atari else SoftOptionCriticFeatures
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")

    if args.model == 'option_critic':

        option_critic = option_critic(
            in_features=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            num_options=args.num_options,
            temperature=args.temp,
            eps_start=args.epsilon_start,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            eps_test=args.optimal_eps,
            device=device
        )
        # Create a prime network for more stable Q values
        option_critic_prime = deepcopy(option_critic)

    else:
        option_critic = soft_option_critic(
            in_features=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            num_options=args.num_options,
            temperature=args.temp,
            eps_start=args.epsilon_start,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            eps_test=args.optimal_eps,
            device=device
        )
        # Create a prime network for more stable Q values
        option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.env != "LunarLander-v2":
        env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    if is_atari:
        best_reward = -float('inf')
        patience = 0
        patience_limit = 500000
        patience_start = 3000000
    if args.model == 'option_critic':
        logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    else:
        logger = Logger(logdir=args.logdir, run_name=f"{SoftOptionCriticFeatures.__name__}-{args.env}-{args.exp}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    steps = 0 ;
    recent_rewards = deque(maxlen=100)
    if args.switch_goal: print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}

        obs   = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            if args.model == 'option_critic':
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            f'scratch/models/option_critic_seed={args.seed}_1k')
            else:
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            f'scratch/models/soft_option_critic_seed={args.seed}_1k')
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            if args.model == 'option_critic':
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            f'scratch/models/option_critic_seed={args.seed}_2k')
            else:
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            f'scratch/models/soft_option_critic_seed={args.seed}_2k')
            break

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
    
            action, logp, entropy = option_critic.get_action(state, current_option)

            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                if args.model == 'option_critic':
                    actor_loss = actor_loss_fn(obs, current_option, logp, entropy,\
                    reward, done, next_obs, option_critic, option_critic_prime, args)
                else:
                    actor_loss = actor_loss_Soft_fn(obs, current_option, logp, entropy, \
                        reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    if args.model == 'option_critic':
                        critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    else:
                        critic_loss = critic_loss_Soft_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            
            if steps % 10000 == 0:
                print(steps, rewards)

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
        if args.env == 'CartPole-v0' or args.env == "LunarLander-v2":
            recent_rewards.append(rewards)
            if len(recent_rewards) == 100:
                avg_reward = np.mean(recent_rewards)
                print(f"Average reward last 100 eps: {avg_reward:.3f}")
                if args.env == 'CartPole-v0':
                    if avg_reward >= 195.0:
                        print("Environment solved in CartPole-v0!")
                        if args.model == 'option_critic':
                            torch.save({'model_params': option_critic.state_dict()},
                                   f'scratch/models/option_critic_cartpole_solved')
                        else:
                            torch.save({'model_params': option_critic.state_dict()},
                                   f'scratch/models/soft_option_critic_cartpole_solved')
                        break
                if args.env == 'LunarLander-v2':
                    if avg_reward >= 250.0:
                        print("Environment solved in LunarLander-v2!")
                        if args.model == 'option_critic':
                            torch.save({'model_params': option_critic.state_dict()},
                                   f'scratch/models/option_critic_lunarlander_solved')
                        else:
                            torch.save({'model_params': option_critic.state_dict()},
                                   f'scratch/models/soft_option_critic_lunarlander_solved')
                        break
        elif is_atari:
            recent_rewards.append(rewards)
            if len(recent_rewards)  == 100 and steps >= patience_start:
                avg_reward = np.mean(recent_rewards)
                if avg_reward > best_reward:
                    best_reward = rewards
                    patience = 0
                else:
                    patience += ep_steps
                if patience >= patience_limit:
                    print("Environment solved in Atari!")
                    if args.model == 'option_critic':
                        torch.save({'model_params': option_critic.state_dict()},
                                    f'scratch/models/option_critic_{args.env}_solved')
                    else:
                        torch.save({'model_params': option_critic.state_dict()},
                                    f'scratch/models/soft_option_critic_{args.env}_solved')
                    break
        
        if steps >= args.max_steps_total:
                    print("Step Limit Reach")
                    if args.model == 'option_critic':
                        torch.save({'model_params': option_critic.state_dict()},
                                    f'scratch/models/option_critic_{args.env}_{steps}')
                    else:
                        torch.save({'model_params': option_critic.state_dict()},
                                    f'scratch/models/soft_option_critic_{args.env}_{steps}')
        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)

if __name__=="__main__":
    args = parser.parse_args()
    run(args)
