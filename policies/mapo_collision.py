import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from nets.drqn import DRQN


class MapoCollision:
    def __init__(self, args):
        self.args = args

        self.drqn = DRQN(args)

        self.parameters = list(self.drqn.parameters())

        self.optimizer = th.optim.RMSprop(self.parameters, lr=args.lr)

        self.hidden = None

    def learn(self, batch, train_step=None):
        args = self.args

        bs = batch["s"].shape[0]

        self.init_hidden(bs)

        for key in batch.keys():
            if key in ['u_pilot', 'u_pilot_one_hot']:
                batch[key] = th.tensor(batch[key], dtype=th.long)
            else:
                batch[key] = th.tensor(batch[key], dtype=th.float32)

        loss = self.get_batch_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, args.grad_norm_clip)
        self.optimizer.step()

    def get_batch_loss(self, batch):
        args = self.args

        bs = batch["s"].shape[0]

        mask = batch["mask"].unsqueeze(3).repeat(1, 1, 1, args.drqn_pilot_output_dim)

        pilot_probs = []

        for transition_idx in range(args.episode_len):
            inputs = self._get_inputs(batch, transition_idx)

            outputs, self.hidden = self.drqn(inputs, self.hidden)

            if args.on_off_pilot_action:
                pilot_prob = th.sigmoid(outputs[0]).view(bs, args.n_agents, -1)
            else:
                pilot_prob = F.softmax(outputs[0], dim=-1).view(bs, args.n_agents, -1)

            pilot_probs.append(pilot_prob)

        # without pre-allocation: bs x episode_len x n_agents x (n_pilots + 1)
        # with pre-allocation: bs x episode_len x n_agents x 1
        pilot_probs = th.stack(pilot_probs, dim=1)
        pilot_probs = pilot_probs * mask

        non_coll_probs = th.zeros(bs, args.episode_len, args.n_agents)

        if args.on_off_pilot_action:
            pilot_probs = pilot_probs.squeeze(-1)  # remove the last dimension
            for i in range(args.n_agents):
                pilot_idle_probs = th.ones(bs, args.episode_len)
                for j in args.contention_map[i]:
                    pilot_idle_probs *= (1 - pilot_probs[:, :, j])

                non_coll_probs[:, :, i] = pilot_probs[:, :, i] * pilot_idle_probs
        else:
            for i in range(args.n_agents):
                pilot_prob = pilot_probs[:, :, i, 1:]

                pilot_idle_probs = th.ones(bs, args.episode_len, args.n_pilots)
                for j in range(args.n_agents):
                    if i != j:
                        pilot_idle_probs *= (1 - pilot_probs[:, :, j, 1:])

                non_coll_probs[:, :, i] = th.sum(pilot_prob * pilot_idle_probs, dim=-1)

        succ_probs = non_coll_probs

        reward = th.sum(succ_probs * batch["eta"], dim=-1)
        loss = - th.mean(reward)

        return loss

    def _get_inputs(self, batch, transition_idx):
        args = self.args

        bs = batch["s"].shape[0]

        inputs = [batch["o"][:, transition_idx]]

        if transition_idx == 0:
            inputs.append(th.zeros_like(batch["u_pilot_onehot"][:, transition_idx]))
        else:
            inputs.append(batch["u_pilot_onehot"][:, transition_idx - 1])

        inputs.append(th.eye(args.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * args.n_agents, -1) for x in inputs], dim=-1)

        return inputs

    def init_hidden(self, n_dim):
        self.hidden = th.zeros((n_dim, self.args.n_users, self.args.drqn_hidden_dim))

