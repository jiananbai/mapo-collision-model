import torch.nn as nn
import torch.nn.functional as F


class DRQN(nn.Module):
    def __init__(self, args):
        super(DRQN, self).__init__()
        self.args = args

        self.input_layer = nn.Linear(args.drqn_input_dim, args.drqn_hidden_dim)
        self.rnn_layer = nn.GRUCell(args.drqn_hidden_dim, args.drqn_hidden_dim)
        self.pilot_layer = nn.Linear(args.drqn_hidden_dim, args.drqn_pilot_output_dim)
        if args.learn_power:
            self.power_layer = nn.Linear(args.drqn_hidden_dim, 1)

    def forward(self, inputs, hidden_state):
        args = self.args

        x = F.relu(self.input_layer(inputs))
        h_in = hidden_state.reshape(-1, args.drqn_hidden_dim)
        h = self.rnn_layer(x, h_in)
        q_pilot = self.pilot_layer(h)
        if args.learn_power:
            q_power = self.power_layer(h)
            return (q_pilot, q_power), h
        else:
            return (q_pilot, ),  h

