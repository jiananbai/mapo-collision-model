import numpy as np
import torch as th
import time
import os

from agents.agent import Agent
from components.rollout import Rollout
from components.replay_buffer import ReplayBuffer
from envs import REGISTRY as env_REGISTRY


class Runner:
    def __init__(self, args):
        self.args = args

        self.agent = Agent(args)
        self.env = env_REGISTRY[args.env](args)
        self.rollout = Rollout(args, self.env, self.agent)
        self.buffer = ReplayBuffer(args)

        self.run_time = None

    def run(self):
        args = self.args

        i_epoch, n_train_steps, n_test_steps = 0, 0, 0

        start_time = time.time()

        while i_epoch < args.max_epochs:
            # print('Epoch {}'.format(i_epoch))
            if i_epoch % args.test_interval == 0:
                self.eval(n_test_steps)
                n_test_steps += 1

            episodes = []
            for i_episode in range(args.n_episodes_per_epoch):
                episode, _ = self.rollout.generate_episode(i_epoch)
                episodes.append(episode)

            epoch_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in epoch_batch.keys():
                    epoch_batch[key] = np.concatenate((epoch_batch[key], episode[key]), axis=0)
            self.buffer.store(epoch_batch)

            for train_step in range(args.n_train_step_per_epoch):
                self.agent.train(self.buffer.sample(args.bs), n_train_steps)
                n_train_steps += 1

            self.run_time = np.array(time.time() - start_time)

            i_epoch += 1

    def eval(self, n_test_steps):
        args = self.args

        test_record = {
            'reward': np.zeros(args.n_test_episodes),
            'n_completed_tasks': np.zeros((args.n_test_episodes, args.n_agents)),
            'n_failed_tasks': np.zeros((args.n_test_episodes, args.n_agents)),
            'pilot_sel': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
            'power': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
            'priority_level': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
            'ncpdr': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
            'virtual_backlog': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
            'urgency_level': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
            'lsfc_dB_rec': np.zeros((args.n_test_episodes, args.episode_len, args.n_agents)),
        }

        for i_episode in range(args.n_test_episodes):
            _, episode_record = self.rollout.generate_episode(is_eval=True)
            for key in test_record.keys():
                test_record[key][i_episode] = episode_record[key]

        for key in test_record.keys():
            if key in ['reward']:
                test_record[key] = np.mean(test_record[key])
            elif key in ['n_completed_tasks', 'n_failed_tasks']:
                test_record[key] = np.sum(test_record[key], axis=0)
            else:
                test_record[key].reshape(-1, args.n_agents)

        if args.verbose:
            drop_rates = test_record['n_failed_tasks'] / test_record['n_completed_tasks'] * args.arrival_prob
            tputs = args.arrival_prob - drop_rates
            sum_tput = np.sum(tputs)

            print('Test step: {}. Reward: {}. Sum tput: {}. Max NCPDR: {}. Drop rates: {}.'
                  .format(n_test_steps,
                          np.round(test_record['reward'], 2),
                          np.round(sum_tput, 2),
                          np.round(np.max(drop_rates / args.drop_rate_thresh), 2),
                          np.round(drop_rates, 2)))

        if args.save:
            for key in test_record.keys():
                file = os.path.join(args.result_path, 'data', key)
                with open(file, 'a') as f:
                    np.savetxt(f, test_record[key].reshape(-1))

            th.save(self.agent.policy.drqn.state_dict(), os.path.join(args.result_path, 'model', 'drqn.pt'))
