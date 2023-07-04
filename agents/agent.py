import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

from policies import REGISTRY as policy_REGISTRY


class Agent:
    def __init__(self, args):
        self.args = args

        self.policy = policy_REGISTRY[args.policy](args)

    def select(self, obs, last_action, agent_id, epsilon=0.05):
        args = self.args

        inputs = np.hstack((obs.copy(), last_action))  # add last action to input

        agent_id_onehot = np.zeros(args.n_agents)
        agent_id_onehot[agent_id] = 1
        inputs = np.hstack((inputs, agent_id_onehot))  # add onehot encoded agent id to input

        inputs = th.tensor(inputs, dtype=th.float32).unsqueeze(0)

        output, self.policy.hidden[:, agent_id, :] = self.policy.drqn(inputs, self.policy.hidden[:, agent_id, :])

        if args.policy in ['qmix']:  # use epsilon greedy for qmix
            if np.random.uniform() < epsilon:
                action_pilot = np.random.randint(args.n_avail_pilots + 1)
            else:
                action_pilot = th.argmax(output[0]).type(dtype=th.long).item()
        else:  # stochastic pilot selection for mapo
            if args.on_off_pilot_action:
                probs = th.sigmoid(output[0])
                action_pilot = Bernoulli(probs=probs).sample().type(dtype=th.long).item()
            else:
                probs = F.softmax(output[0], dim=1)
                action_pilot = Categorical(probs=probs).sample().type(dtype=th.long).item()

        if args.learn_power:
            action_power = th.sigmoid(output[1]).item()
            action = (action_pilot, action_power)
        else:
            action = (action_pilot, )

        return action

    def train(self, batch, i_step=None):
        self.policy.learn(batch, i_step)

