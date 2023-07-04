import numpy as np
import torch as th
import torch.nn as nn

from nets.drqn import DRQN
from nets.qmix_net import QMixNet


class QMix:
    def __init__(self, args):
        self.args = args

        self.drqn = DRQN(args)
        self.target_drqn = DRQN(args)

        self.mix_net = QMixNet(args)
        self.target_mix_net = QMixNet(args)

        self.target_drqn.load_state_dict(self.drqn.state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())

        self.parameters = list(self.drqn.parameters()) + list(self.mix_net.parameters())

        self.optimizer = th.optim.RMSprop(self.parameters, lr=args.lr)

        self.hidden = None
        self.target_hidden = None

    def learn(self, batch, train_step):
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

        if train_step > 0 and train_step % args.target_update_interval == 0:
            self.target_drqn.load_state_dict(self.drqn.state_dict())
            self.target_mix_net.load_state_dict(self.mix_net.state_dict())

    def get_batch_loss(self, batch):
        args = self.args

        bs = batch["s"].shape[0]

        q_evals, q_targets = self.get_q_values(batch)

        q_evals[batch["mask"] == 0] = 0
        q_targets[batch["mask"] == 0] = 0

        q_evals = th.gather(q_evals, dim=-1, index=batch["u_pilot"].unsqueeze(-1)).squeeze(-1)
        q_targets = q_targets.max(dim=-1)[0]

        q_total_eval = self.mix_net(q_evals, batch["s"])
        q_total_target = self.target_mix_net(q_targets, batch["s_next"])

        terminated = th.zeros_like(batch["r"])
        terminated[:, -1] = 1.0

        targets = batch['r'] + args.gamma * (1-terminated) * q_total_target

        td_error = q_total_eval - targets.detach()

        loss = (td_error ** 2).mean()

        return loss

    def get_q_values(self, batch):
        args = self.args

        bs = batch["s"].shape[0]

        q_evals, q_targets = [], []

        for transition_idx in range(args.episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)

            q_eval, self.hidden = self.drqn(inputs, self.hidden)
            q_eval = q_eval[0]
            q_target, self.target_hidden = self.target_drqn(inputs, self.target_hidden)
            q_target = q_target[0]

            q_eval = q_eval.view(bs, args.n_agents, -1)
            q_target = q_target.view(bs, args.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = th.stack(q_evals, dim=1)
        q_targets = th.stack(q_targets, dim=1)

        return q_evals, q_targets

    def _get_inputs(self, batch, transition_idx):
        args = self.args

        bs = batch["s"].shape[0]

        inputs = [batch["o"][:, transition_idx]]
        inputs_next = [batch["o_next"][:, transition_idx]]

        if transition_idx == 0:
            inputs.append(th.zeros_like(batch["u_pilot_onehot"][:, transition_idx]))
        else:
            inputs.append(batch["u_pilot_onehot"][:, transition_idx - 1])
        inputs_next.append(batch["u_pilot_onehot"][:, transition_idx])

        inputs.append(th.eye(args.n_agents).unsqueeze(0).expand(bs, -1, -1))
        inputs_next.append(th.eye(args.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(bs * args.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def init_hidden(self, n_dim):
        args = self.args

        self.hidden = th.zeros((n_dim, args.n_agents, args.drqn_hidden_dim))
        self.target_hidden = th.zeros((n_dim, args.n_agents, args.drqn_hidden_dim))
