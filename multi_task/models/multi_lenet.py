import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def __init__(self, input_shape): 
    super(FiLM, self).__init__()
    self.gammas = torch.tensor(torch.randn(input_shape), requires_grad=True)
    self.betas = torch.tensor(torch.randn(input_shape), requires_grad=True)

  def forward(self, x):
    assert x.dim() == 4 or x.dim() == 2
    if (x.dim() == 4):
        gammas = self.gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = self.betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    elif (x.dim() == 2):
        gammas = self.gammas.expand_as(x)
        betas = self.betas.expand_as(x)
    return (gammas * x) + betas

class MultiFilmLeNetR(nn.Module):
    def __init__(self):
        super(MultiFilmLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)
        self.film1_task1 = FiLM(input_shape = (1, 10))
        self.film1_task2 = FiLM(input_shape = (1, 10))
        self.film2_task1 = FiLM(input_shape = (1, 20))
        self.film2_task2 = FiLM(input_shape = (1, 20))
        self.film3_task1 = FiLM(input_shape = (1, 50))
        self.film3_task2 = FiLM(input_shape = (1, 50))

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1,channel_size,1,1)*0.5).to(DEVICE))
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask):
        x = self.conv1(x)
        x1 = self.film1_task1(x)
        x2 = self.film1_task2(x)
        x1 = F.relu(F.max_pool2d(x1, 2))
        x2 = F.relu(F.max_pool2d(x2, 2))
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x1 = self.film2_task1(x1)
        x2 = self.film2_task2(x2)
        if self.training:
            x1 = self.conv2_drop(x1)
            x2 = self.conv2_drop(x2)
        x1 = F.relu(F.max_pool2d(x1, 2))
        x2 = F.relu(F.max_pool2d(x2, 2))
        x1 = x1.view(-1, 320)
        x2 = x2.view(-1, 320)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x1 = self.film3_task1(x1)
        x2 = self.film3_task2(x2)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        return x1, x2, mask


class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1,channel_size,1,1)*0.5).to(DEVICE))
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if self.training:
            x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x, mask

class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            x = x*mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask
