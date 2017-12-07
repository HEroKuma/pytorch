import torch
from torch import nn
from torch.autograd import Variable

basic_rnn = nn.RNN(input_size=20, hidden_size=50, num_layers=2)
print(basic_rnn.weight_ih_l0)
print(basic_rnn.weight_hh_l0)
print(basic_rnn.bias_hh_l0)

toy_input = Variable(torch.randn(100, 32, 20))
h_0 = Variable(torch.randn(2, 32, 50))

toy_output, h_n = basic_rnn(toy_input, h_0)
print(toy_output.size())
print(h_n.size())

# LSTM
lstm = nn.LSTM(input_size=20, hidden_size=50, num_layers=2)
print(lstm.weight_ih_l0)
lstm_out, (h_n, c_n) = lstm(toy_input)
print(lstm_out.size())
print(h_n.size())
print(c_n.size())
