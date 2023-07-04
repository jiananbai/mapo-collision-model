import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args

        self.buffer = {
            's': np.empty([args.max_buffer_size, args.episode_len, args.state_dim]),
            'o': np.empty([args.max_buffer_size, args.episode_len, args.n_agents, args.obs_dim]),
            'u_pilot': np.empty([args.max_buffer_size, args.episode_len, args.n_agents]),
            'u_pilot_onehot': np.empty([args.max_buffer_size, args.episode_len, args.n_agents, args.n_avail_pilots]),
            'r': np.empty([args.max_buffer_size, args.episode_len]),
            'eta': np.empty([args.max_buffer_size, args.episode_len, args.n_agents]),
            'mask': np.empty([args.max_buffer_size, args.episode_len, args.n_agents]),
            's_next': np.empty([args.max_buffer_size, args.episode_len, args.state_dim]),
            'o_next': np.empty([args.max_buffer_size, args.episode_len, args.n_agents, args.obs_dim]),
        }

        if args.learn_power:
            self.buffer['u_power'] = np.empty([args.max_buffer_size, args.episode_len, args.n_agents])
            self.buffer['lsfc_dB'] = np.empty([args.max_buffer_size, args.episode_len, args.n_agents])

        self.current_idx = 0
        self.current_size = 0

        self.lock = threading.Lock()

    def store(self, batch):
        args = self.args

        n_episodes = batch['s'].shape[0]

        with self.lock:
            idx = self._get_idx(inc=n_episodes)
            self.buffer['s'][idx] = batch['s']
            self.buffer['o'][idx] = batch['o']
            self.buffer['u_pilot'][idx] = batch['u_pilot']
            self.buffer['u_pilot_onehot'][idx] = batch['u_pilot_onehot']
            self.buffer['r'][idx] = batch['r']
            self.buffer['eta'][idx] = batch['eta']
            self.buffer['mask'][idx] = batch['mask']
            self.buffer['s_next'][idx] = batch['s_next']
            self.buffer['o_next'][idx] = batch['o_next']

            if args.learn_power:
                self.buffer['u_power'][idx] = batch['u_power']
                self.buffer['lsfc_dB'][idx] = batch['lsfc_dB']

    def _get_idx(self, inc=1):
        args = self.args

        if self.current_idx + inc <= args.max_buffer_size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        else:
            overflow = inc - (args.max_buffer_size - self.current_idx)
            idx_a = np.arange(self.current_idx, args.max_buffer_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow

        self.current_size = min(args.max_buffer_size, self.current_size + inc)

        if inc == 1:
            idx = np.array([idx[0]])

        return idx

    def sample(self, bs):
        args = self.args

        if self.current_size < bs:
            bs = self.current_size

        transitions = {}
        indices = np.random.randint(0, self.current_size, size=bs)
        for key in self.buffer.keys():
            transitions[key] = self.buffer[key][indices]

        return transitions
