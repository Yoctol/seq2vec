import torch
import torch.nn as nn
from torch.autograd import Variable

class BidirectionalRNNEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            n_cells,
            dropout,
            bias=True,
            **kwargs
            ):
        super(BidirectionalRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_cells = n_cells
        self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout,
                bidirectional=True,
                )

    def forward(self, inputs):
        """
        inputs: Variable
            Input data with type Variable(sequence_size, batch_size, embedding_size)
        """
        batch_size = inputs.size()[1]
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = Variable(inputs.data.new(*state_shape).zero_())
        c0 = Variable(inputs.data.new(*state_shape).zero_())

        _, (ht, _) = self.rnn(inputs, (h0, c0))

        return self.transform_state_bi(ht, batch_size)

    def transform_state_bi(self, state, batch_size):
        """Help function for transforming state of bidirectional LSTM
        state: Variable
            Last state of rnn
        """
        return state.transpose(0, 1).contiguous().view(batch_size, -1)
