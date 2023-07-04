import numpy as np

from utils.functions import fairness_func


class CollisionModel:
    def __init__(self, args):
        self.args = args

        self.i_time_slot = 0

        # record
        self.n_completed_tasks = np.zeros(args.n_users)
        self.n_failed_tasks = np.zeros(args.n_users)
        self.n_succ_tasks = np.zeros(args.n_users)
        self.total_delay = np.zeros(args.n_users)
        self.ncpdr_rec = np.zeros((args.episode_len + 1, args.n_users))  # normalized cumulated packet drop rate
        self.virtual_backlog_rec = np.zeros((args.episode_len + 1, args.n_users))
        self.urgency_level_rec = np.zeros((args.episode_len + 1, args.n_users))
        self.priority_level_rec = np.zeros((args.episode_len + 1, args.n_users))
        self.pilot_sel_rec = np.zeros((args.episode_len + 1, args.n_users))
        self.power_rec = np.zeros((args.episode_len + 1, args.n_users))
        self.lsfc_dB_rec = np.zeros((args.episode_len + 1, args.n_users))

        self.pilot_status = np.zeros(args.n_pilots)  # -1: collision. 0: idle. +1: success
        self.user_status = np.zeros(args.n_users)  # 0: idle. 1: backlogged
        self.virtual_backlogs = np.zeros(args.n_users)

        self.queues = [[] for _ in range(args.n_users)]
        self.new_task_indicator = np.zeros(args.n_users)
        self._update_queues([])

    def reset(self):
        self.__init__(self.args)

    def get_state(self):
        args = self.args

        state = []
        for i in range(args.n_users):
            n_rem_slots_onehot = np.zeros(args.delay_len)
            if len(self.queues[i]) > 0:
                n_rem_slots_onehot[self.queues[i][0] - 1] = 1
            state.append(n_rem_slots_onehot)

        state.append(self.new_task_indicator)

        ncpdr = self.n_failed_tasks / (args.episode_len * args.drop_rate_thresh)
        if args.approx in ['log-sum-exp']:
            state.append(ncpdr)
        elif args.approx in ['virtual-queue']:
            state.append(self.virtual_backlogs)

        state.append(self.pilot_status)

        state = np.concatenate(state)

        return state

    def get_observations(self):
        args = self.args

        ncpdr = self.n_failed_tasks / (args.episode_len * args.drop_rate_thresh)

        observations = []
        for i in range(args.n_users):
            obs = []

            n_rem_slots_onehot = np.zeros(args.delay_len)
            if len(self.queues[i]) > 0:
                n_rem_slots_onehot[self.queues[i][0] - 1] = 1
            obs.append(n_rem_slots_onehot)

            obs.append([self.new_task_indicator[i]])

            if args.approx in ['log-sum-exp']:
                obs.append([ncpdr[i]])
            elif args.approx in ['virtual-queue']:
                obs.append([self.virtual_backlogs[i]])

            if args.feedback:
                if args.pre_alloc:
                    obs.append([self.pilot_status[args.assigned_pilots[i] - 1]])
                else:
                    obs.append(self.pilot_status)

            observations.append(np.concatenate(obs))

        return observations

    def step(self, actions):
        args = self.args

        self.pilot_sel_rec[self.i_time_slot, :] = actions[0]

        if not args.pre_alloc:
            pilot_sel = np.asarray(actions[0])
        else:
            pilot_sel = np.zeros(args.n_users, dtype=np.int32)
            for i in range(args.n_users):
                pilot_sel[i] = args.assigned_pilots[i] if actions[0][i] > 0 else 0

        detected_users = self._pilot_transmission(pilot_sel)

        succ_users = detected_users  # a user can deliver a packet successfully if and only if it is detected

        priority_levels, stats = self.cal_priority_levels_and_stats()

        reward = np.sum(priority_levels[succ_users])

        self.priority_level_rec[self.i_time_slot, :] = priority_levels
        self.ncpdr_rec[self.i_time_slot, :] = stats['ncpdr']
        self.virtual_backlog_rec[self.i_time_slot, :] = stats['virtual_backlog']
        self.urgency_level_rec[self.i_time_slot, :] = stats['urgency_level']

        self.i_time_slot += 1

        self._update_queues(succ_users)

        return reward

    def cal_priority_levels_and_stats(self):
        args = self.args

        backlogged_users = self.get_backlogged_users()

        ncpdr = self.n_failed_tasks / (args.episode_len * args.drop_rate_thresh)
        urgency_levels = np.zeros(args.n_users)
        priority_levels = np.zeros(args.n_users)
        for i in backlogged_users:
            urgency_levels[i] = 1 - (self.queues[i][0] - 1) / args.max_delay[i]
            normalized_urgency_levels = urgency_levels[i] / (args.episode_len * args.drop_rate_thresh[i])
            if args.approx in ['log-sum-exp']:
                priority_levels[i] = fairness_func(ncpdr[i] + normalized_urgency_levels) - fairness_func(ncpdr[i])
            elif args.approx in ['virtual-queue']:
                priority_levels[i] = self.virtual_backlogs[i] * normalized_urgency_levels

        priority_levels = priority_levels / (np.sum(priority_levels) + np.finfo(float).eps)
        stats = {'ncpdr': ncpdr,
                 'virtual_backlog': self.virtual_backlogs,
                 'urgency_level': urgency_levels}

        return priority_levels, stats

    def _pilot_transmission(self, pilot_sel):
        args = self.args

        backlogged_users = self.get_backlogged_users()

        sel_mat = np.zeros((args.n_users, args.n_pilots))
        active_users = []
        for i in backlogged_users:
            if pilot_sel[i] > 0:
                sel_mat[i, pilot_sel[i] - 1] = 1
                active_users.append(i)

        n_users_on_pilots = np.sum(sel_mat, axis=0)
        active_pilot_indicator = np.zeros(args.n_pilots)
        active_pilot_indicator[n_users_on_pilots > 0] = 1

        detected_users = []
        collided_users = []
        for k in range(args.n_pilots):
            if n_users_on_pilots[k] == 1:
                detected_users.append(np.where(sel_mat[:, k] == 1)[0][0])
                self.pilot_status[k] = 1
            elif n_users_on_pilots[k] > 1:
                collided_users += list(np.where(sel_mat[:, k] == 1)[0])
                self.pilot_status[k] = -1
            else:
                self.pilot_status[k] = 0

        return detected_users

    def get_backlogged_users(self):
        return [i for i in range(self.args.n_users) if len(self.queues[i]) > 0]

    def _update_queues(self, succ_users):
        args = self.args

        self.n_succ_tasks[succ_users] += 1
        self.n_completed_tasks[succ_users] += 1

        for i in succ_users:
            self.total_delay[i] += args.max_delay[i] - self.queues[i][0]
            self.queues[i].pop(0)

        backlogged_users = self.get_backlogged_users()

        for i in range(args.n_users):
            z_indicator = (np.sum(self.virtual_backlogs) > args.V)
            self.virtual_backlogs[i] = np.max([1, self.virtual_backlogs[i] - args.z_max * z_indicator])
            if i in backlogged_users:
                if self.queues[i][0] == 1 and i not in succ_users:
                    self.n_completed_tasks[i] += 1
                    self.n_failed_tasks[i] += 1
                    self.virtual_backlogs[i] += 1 / args.drop_rate_thresh[i]

                self.queues[i] = [n_rem_slots - 1 for n_rem_slots in self.queues[i] if n_rem_slots > 1]

            if np.random.rand() < args.arrival_prob[i]:
                self.queues[i].append(args.max_delay[i])
                self.new_task_indicator[i] = 1
            else:
                self.new_task_indicator[i] = 0

        self.user_status = np.zeros(args.n_users)
        self.user_status[self.get_backlogged_users()] = 1
