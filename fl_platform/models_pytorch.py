import torch.nn as nn

class LogisticRegressionModel(nn.Module):
  def __init__(self, input_dim, output_dim=1):
    super(LogisticRegressionModel, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.linear(x)

class SimpleNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
      super(SimpleNN, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
      out = self.fc1(x)
      out = self.relu(out)
      out = self.fc2(out)
      return out