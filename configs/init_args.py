import numpy as np


def init_args(args):

    if args.policy in ['mapo']:
        args.policy = args.policy + '_' + args.env

    if args.env in ['collision']:
        args.learn_power = False  # power control is not considered in collision model

    # system configuration and pilot pre-allocation
    if args.env in ['collision', 'o-gfra']:
        args.n_users = 12
        args.pilot_len = 6
        args.n_pilots = 6
        if args.pre_alloc:
            args.assigned_pilots = np.array([1, 2, 3, 4, 5, 6] * 2)

            args.pre_alloc_indmtx = np.zeros((args.n_users, args.n_pilots))
            for i in range(args.n_users):
                args.pre_alloc_indmtx[i, args.assigned_pilots[i] - 1] = 1

            args.contention_map = []
            for i in range(args.n_users):
                k = args.assigned_pilots[i]
                contention_users = []
                for j in range(args.n_users):
                    if j != i and args.assigned_pilots[j] == k:
                        contention_users.append(j)
                args.contention_map.append(contention_users)

    elif args.env in ['n-gfra']:
        args.n_users = 12
        args.pilot_len = 6
        args.n_pilots = 12
    else:
        raise NotImplementedError
    args.n_agents = args.n_users

    # traffic model
    args.drop_rate_thresh = np.array([0.05] * 4 + [0.2] * 8)
    args.arrival_prob = np.array([0.2] * 4 + [0.65] * 8)
    args.max_delay = np.array([2] * 4 + [5] * 8)
    args.rate_thresh = np.array([0.5] * 4 + [1.5] * 8)

    args.delay_len = np.max(args.max_delay)

    # observation/state dimension
    args.obs_dim = args.delay_len + 1 + 1  # delay + packet arrival + NCPDR or virtual backlog
    args.state_dim = (args.delay_len + 1 + 1) * args.n_agents + args.n_pilots
    if args.feedback:
        if args.pre_alloc:
            args.obs_dim += 1  # status of the pre-assigned pilot
        else:
            args.obs_dim += args.n_pilots  # status of all pilots
    if args.learn_power:
        args.obs_dim += 1  # LSFC of the user
        args.state_dim += args.n_users  # LSFCs of all users

    # action dimension
    if args.env in ['collision', 'o-gfra']:
        if args.pre_alloc:
            args.n_avail_pilots = 1
        else:
            args.n_avail_pilots = args.n_pilots
    elif args.env in ['n-gfra']:
        args.n_avail_pilots = 1

    args.on_off_pilot_action = True if args.n_avail_pilots == 1 else False

    # neural network input dimension
    args.drqn_input_dim = args.obs_dim + args.n_avail_pilots + args.n_users  # obs + last pilot action + user id
    if args.learn_power:
        args.drqn_input_dim += 1  # last power action

    # neural network output dimension
    args.drqn_pilot_output_dim = 1 if args.on_off_pilot_action else args.n_avail_pilots + 1

    if args.on_off_pilot_action:
        args.drqn_pilot_output_dim = 2 if args.policy in ['qmix'] else 1
    else:
        args.drqn_pilot_output_dim = args.n_avail_pilots + 1


    # epsilon greedy
    args.epsilon_anneal_step = (args.epsilon_start - args.epsilon_final) / args.epsilon_anneal_time

    return args