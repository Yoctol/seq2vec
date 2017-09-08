import torch
import torch.nn as nn
from torch.autograd import Variable

class BidirectionalRNNEncoder(nn.module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            **kwargs
            ):
        super(BidirectionalRNNEncoder, self).__init__()
        self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True
                )

    def forward(self, inputs):
        pass
