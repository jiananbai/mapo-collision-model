import numpy as np
import torch as th


class Rollout:
    def __init__(self, args, env, agent):
        self.args = args

        self.env = env
        self.agent = agent

    @th.no_grad()
    def generate_episode(self, i_epoch=None, is_eval=False):
        args = self.args

        self.env.reset()

        i_time_slot = 0

        s, o, u_pilot, u_pilot_onehot, r, eta, mask, u_power = [], [], [], [], [], [], [], []

        episode_reward = 0

        last_action_pilot_onehot = np.zeros((args.n_agents, args.n_avail_pilots))
        last_action_power = np.zeros(args.n_agents)

        self.agent.policy.init_hidden(1)

        if is_eval:
            epsilon = 0
        else:
            epsilon = max(args.epsilon_start - i_epoch * args.epsilon_anneal_step, args.epsilon_final)

        while i_time_slot < args.episode_len + 1:
            obs = self.env.get_observations()
            state = self.env.get_state()
            backlogged_users = self.env.get_backlogged_users()
            agent_status = np.zeros(args.n_agents)
            agent_status[backlogged_users] = 1

            actions_pilot, actions_pilot_onehot, actions_power = [], [], []
            for agent_idx in range(args.n_agents):
                if args.learn_power:
                    last_action = np.hstack((last_action_pilot_onehot[agent_idx], last_action_power[agent_idx]))
                else:
                    last_action = last_action_pilot_onehot[agent_idx]
                action = self.agent.select(obs[agent_idx], last_action, agent_idx, epsilon)
                action_pilot = action[0] if agent_status[agent_idx] == 1 else 0

                actions_pilot.append(action_pilot)
                action_pilot_onehot = np.zeros(args.n_avail_pilots)
                if action_pilot > 0:
                    action_pilot_onehot[action_pilot - 1] = 1
                actions_pilot_onehot.append(action_pilot_onehot)

                last_action_pilot_onehot[agent_idx] = action_pilot_onehot

                if args.learn_power:
                    action_power = action[1] if agent_status[agent_idx] == 1 else 0
                    actions_power.append(action_power)

                    last_action_power[agent_idx] = action_power

            actions = (actions_pilot, actions_power)

            priority_levels, _ = self.env.cal_priority_levels_and_stats()

            reward = self.env.step(actions)

            s.append(state)
            o.append(obs)
            u_pilot.append(actions_pilot)
            u_pilot_onehot.append(actions_pilot_onehot)
            r.append(reward)
            eta.append(priority_levels)
            mask.append(agent_status)

            if args.learn_power:
                u_power.append(actions_power)

            episode_reward += reward

            i_time_slot += 1

        s_next = s[1:]
        o_next = o[1:]
        s = s[:-1]
        o = o[:-1]
        u_pilot = u_pilot[:-1]
        u_pilot_onehot = u_pilot_onehot[:-1]
        r = r[:-1]
        eta = eta[:-1]
        mask = mask[:-1]

        episode = {
            's': s.copy(),
            'o': o.copy(),
            'u_pilot': u_pilot.copy(),
            'u_pilot_onehot': u_pilot_onehot.copy(),
            'r': r.copy(),
            'eta': eta.copy(),
            'mask': mask.copy(),
            's_next': s_next.copy(),
            'o_next': o_next.copy(),
        }
        if args.learn_power:
            u_power = u_power[:-1]
            lsfc_dB = self.env.lsfc_dB_rec[:-1]

            episode['u_power'] = u_power.copy()
            episode['lsfc_dB'] = lsfc_dB.copy()

        for key in episode.keys():
            episode[key] = np.expand_dims(np.array(episode[key]), axis=0)

        episode_reward = episode_reward / args.episode_len

        if np.sum(self.env.n_completed_tasks == 0) > 0:
            self.env.n_completed_tasks += np.finfo(float).eps  # avoid dividing by zero

        episode_record = {
            'reward': episode_reward,
            'n_completed_tasks': self.env.n_completed_tasks,
            'n_failed_tasks': self.env.n_failed_tasks,
            'pilot_sel': self.env.pilot_sel_rec[:-1],
            'power': self.env.power_rec[:-1],
            'priority_level': self.env.priority_level_rec[:-1],
            'ncpdr': self.env.ncpdr_rec[:-1],
            'virtual_backlog': self.env.virtual_backlog_rec[:-1],
            'urgency_level': self.env.urgency_level_rec[:-1],
            'lsfc_dB_rec': self.env.lsfc_dB_rec[:-1],
        }

        return episode, episode_record
