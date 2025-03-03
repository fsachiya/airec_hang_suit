import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,trainInitialValue=True):
        super(BasicRNN, self).__init__()

        self.activation = torch.tanh
        self.hidden_size = hidden_size
        
        # https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
        self.h2h = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh' )
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):

        new_state = self.h2h(x, state)
        out_vec   = self.activation(self.h2o(new_state))

        return out_vec, new_state


if __name__ == '__main__':
    from torchinfo import summary
    rnn_model = BasicRNN(2, 20, 2)
    summary( rnn_model, input_size=(4,2) )
