import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable


D_in = 3
D_out = 2
hidden_size = 4

batch_size = 1

h_0 = Variable(torch.Tensor(batch_size, hidden_size))
c_0 = Variable(torch.Tensor(batch_size, hidden_size))

x_data = Variable(torch.Tensor(torch.randn(batch_size, D_in)))
y_data = Variable(torch.Tensor(torch.randn(batch_size, D_out)))


# x_data = Variable(torch.Tensor(3))
# y_data = Variable(torch.Tensor(2))

print(x_data)

class OneLSTM(torch.nn.Module):
	def __init__(self):

		super(OneLSTM, self).__init__()
		self.lstm1 = torch.nn.LSTMCell(D_in, hidden_size)
		self.lstm2 = torch.nn.LSTMCell(hidden_size, hidden_size)
		self.linear1 = torch.nn.Linear(hidden_size, D_out)

	def forward(self, x, h_0, c_0):
		h_1, c_1 = self.lstm1(x, (h_0, c_0))
		h_2, c_2 = self.lstm2(h_1, (h_1, c_1))
		y_pred = self.linear1(h_2)

		return y_pred


model = OneLSTM()



output = model(x_data, h_0, c_0)
# print('output')
# print(output)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
	y_pred = model(x_data, h_0, c_0)

	loss = criterion(y_pred, y_data)

	print(epoch, loss.data[0])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

print(y_data)
print(model(x_data, h_0, c_0))