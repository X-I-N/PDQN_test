import os
import numpy as np
import gym
import torch
import datetime
import argparse
import gym_platform
from gym.wrappers import Monitor
from torch.utils.tensorboard import SummaryWriter
from agents.pdqn import PDQNAgent_v1
from wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper, PlatformFlattenedActionWrapper

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return act, params


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir', default='results/platform', type=str)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--max_steps', default=250, type=int)
    parser.add_argument('--train_eps', default=8000, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--epsilon_start', default=0.95, type=float)
    parser.add_argument('--epsilon_decay', default=5000, type=int)
    parser.add_argument('--epsilon_end', default=0.02, type=float)
    parser.add_argument('--epsilon_steps', default=1000, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--param_net_lr', default=1e-4, type=float)
    parser.add_argument('--memory_size', default=1e5, type=int)
    parser.add_argument('--initialise_params', default=True)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--render-freq', default=100, type=int)
    parser.add_argument('--scale_actions', default=True, type=bool)
    parser.add_argument('--visualise', default=False, type=bool)
    parser.add_argument('--layers', default=[128, ])
    parser.add_argument('--title', default="PDQN_nstep", type=str)
    parser.add_argument('--invert_gradients', default=True, type=bool)
    parser.add_argument('--initial_memory_threshold', default=128, type=int)

    config = parser.parse_args()
    return config


def run(cfg):
    env = gym.make('Platform-v0')
    initial_params = [3., 10., 400.]
    if cfg.scale_actions:
        for i in range(env.action_space.spaces[0].n):
            initial_params[i] = 2. * (initial_params[i] - env.action_space.spaces[1].spaces[i].low) / (
                    env.action_space.spaces[1].spaces[i].high - env.action_space.spaces[1].spaces[i].low) - 1.

    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    if cfg.scale_actions:
        env = ScaledParameterisedActionWrapper(env)

    save_dir = os.path.join(cfg.save_dir, cfg.title)
    env = Monitor(env, directory=os.path.join(save_dir, str(cfg.seed)), force=True)
    env.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # agent = RandomAgent(env.action_space, cfg.seed)
    agent = PDQNAgent_v1(env.observation_space.spaces[0], env.action_space,
                         epsilon_initial=cfg.epsilon_start,
                         epsilon_steps=cfg.epsilon_steps,
                         epsilon_final=cfg.epsilon_end,
                         batch_size=cfg.batch_size,
                         inverting_gradients=cfg.invert_gradients,
                         learning_rate_actor=1e-3,
                         learning_rate_actor_param=1e-4,
                         device=cfg.device,
                         gamma=cfg.gamma,
                         seed=cfg.seed,
                         actor_kwargs={"hidden_layers": cfg.layers},
                         replay_memory_size=cfg.memory_size,
                         actor_param_kwargs={"hidden_layers": cfg.layers},
                         )

    #  initialise parameters for parameter pass through layer

    if cfg.initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)

    rewards = []
    moving_avg_rewards = []
    eps_steps = []
    log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/train/platform/" + str(cfg.seed) + "_ " + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_eps in range(1, 1 + cfg.train_eps):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0.

        for i_step in range(cfg.max_steps):
            act, act_param, all_action_param = agent.act(state)
            action = pad_action(act, act_param)
            (next_state, steps), reward, done, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            agent.replay_memory.push(state, np.concatenate(([act], all_action_param)).ravel(), reward, next_state, done)

            state = next_state
            episode_reward += reward
            agent._optimize_td_loss()

            if cfg.visualise and i_eps % cfg.render_freq == 0:
                env.render()
            if done:
                break

        if i_eps % cfg.target_update == 0:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.actor_param_target.load_state_dict(agent.actor_param.state_dict())

        rewards.append(episode_reward)
        eps_steps.append(i_step)
        if i_eps == 1:
            moving_avg_rewards.append(episode_reward)
        else:
            moving_avg_rewards.append(episode_reward * 0.1 + moving_avg_rewards[-1] * 0.9)

        if i_eps % 100 == 0:
            print('Episode: ', i_eps, 'R100: ', moving_avg_rewards[-1], 'n_steps: ', np.array(eps_steps[-100]).mean())

        writer.add_scalars('rewards', {'raw': rewards[-1], 'moving_average': moving_avg_rewards[-1]}, i_eps)
        writer.add_scalar('steps_of_each_trials', eps_steps[-1], i_eps)

    writer.close()
    print('Model saved!')
    if cfg.eval_eps > 0:
        print('start to evaluate agent over {} episodes'.format(cfg.eval_eps))
        agent.epsilon = 0
        agent.epsilon_end = 0
        eval_returns = evaluation(env, cfg.eval_eps, agent)
        print("Average evaluation return =", sum(eval_returns) / len(eval_returns))
        print('Finish evaluating.')


def evaluation(env, episodes, agent):
    rewards = []
    steps = []
    for i in range(1, 1 + episodes):
        state, info = env.reset()
        done = False
        step_cnt = 0
        eps_reward = 0
        while not done:
            step_cnt += 1
            state = np.array(state, dtype=np.float32)
            act, act_param, all_action_param = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, done, info = env.step(action)
            eps_reward += reward

        rewards.append(eps_reward)
        steps.append(step_cnt)

        if i % 100 == 0:
            print("R100 = ", np.mean(rewards[-100]))

    return rewards


if __name__ == '__main__':
    cfg = get_args()
    run(cfg)
