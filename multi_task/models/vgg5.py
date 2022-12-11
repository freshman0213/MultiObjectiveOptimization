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
    self.gammas = nn.parameter.Parameter(torch.randn(input_shape))
    self.betas = nn.parameter.Parameter(torch.randn(input_shape))

  def forward(self, x):
    assert x.dim() == 4 or x.dim() == 2
    if (x.dim() == 4):
        gammas = self.gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = self.betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    elif (x.dim() == 2):
        gammas = self.gammas.expand_as(x)
        betas = self.betas.expand_as(x)
    return (gammas * x) + betas


class two_conv_pool(nn.Module):
    def __init__(self, in_channel, F1, F2):
        super(two_conv_pool, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(F2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        return x


class three_conv_pool(nn.Module):
    def __init__(self, in_channel, F1, F2, F3):
        super(three_conv_pool, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(F3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        return x



# adapted from 
# https://github.com/kkweon/mnist-competition/blob/master/vgg5.py?fbclid=IwAR3KCzA88m1Ki15YrVkgH0fGUfwoqxq0oWfUIGEIm2JDXeizeYE_7p6Dx1E
class MultiVgg5R(nn.Module):
    def __init__(self):
        super(MultiVgg5R, self).__init__()
        self.layer1 = two_conv_pool(3, 32, 32)
        self.layer2 = two_conv_pool(32, 64, 64)
        self.layer3 = three_conv_pool(64, 128, 128, 128)
        self.layer4 = three_conv_pool(128, 256, 256, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 1024)  # flatten
        return x


class two_conv_pool_film(nn.Module):
    def __init__(self, in_channel, F1, F2, num_tasks=2):
        super(two_conv_pool_film, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=3, padding='same')
        # self.bn1 = nn.BatchNorm2d(F1)
        # self.bn1 = nn.ModuleList([nn.BatchNorm2d(F1), nn.BatchNorm2d(F1)])
        self.films1 = nn.ModuleList([FiLM(input_shape = (1, F1)) for _ in range(num_tasks)])
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding='same')
        # remove bn for film
        # self.bn2 = nn.BatchNorm2d(F2, affine=False) 
        # self.bn2 = nn.ModuleList([nn.BatchNorm2d(F2, affine=False), nn.BatchNorm2d(F2, affine=False)]) 
        self.films2 = nn.ModuleList([FiLM(input_shape = (1, F2)) for _ in range(num_tasks)])
    def forward(self, x, task):
        x = F.relu(self.films1[task](self.conv1(x)))
        x = F.relu(self.films2[task](self.conv2(x)))
        x = F.max_pool2d(x, 2)
        return x


class three_conv_pool_film(nn.Module):
    def __init__(self, in_channel, F1, F2, F3, num_tasks=2):
        super(three_conv_pool_film, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=3, padding='same')
        # self.bn1 = nn.BatchNorm2d(F1)
        # self.bn1 = nn.ModuleList([nn.BatchNorm2d(F1), nn.BatchNorm2d(F1)])
        self.films1 = nn.ModuleList([FiLM(input_shape = (1, F1)) for _ in range(num_tasks)])
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding='same')
        # self.bn2 = nn.BatchNorm2d(F2)
        # self.bn2 = nn.ModuleList([nn.BatchNorm2d(F2), nn.BatchNorm2d(F2)])
        self.films2 = nn.ModuleList([FiLM(input_shape = (1, F2)) for _ in range(num_tasks)])
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=3, padding='same')
        # remove bn for film
        # self.bn3 = nn.BatchNorm2d(F3, affine=False)
        # self.bn3 = nn.ModuleList([nn.BatchNorm2d(F3, affine=False), nn.BatchNorm2d(F3, affine=False)])
        self.films3 = nn.ModuleList([FiLM(input_shape = (1, F3)) for _ in range(num_tasks)])

    def forward(self, x, task):
        x = F.relu(self.films1[task](self.conv1(x)))
        x = F.relu(self.films2[task](self.conv2(x)))
        x = F.relu(self.films3[task](self.conv3(x)))
        x = F.max_pool2d(x, 2)
        return x

class two_conv_pool_split(nn.Module):
    def __init__(self, in_channel, F1, F2, num_tasks=2):
        super(two_conv_pool_split, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=3, padding='same')
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(F1), nn.BatchNorm2d(F1)])
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding='same')
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(F2), nn.BatchNorm2d(F2)]) 

    def forward(self, x, task):
        x = F.relu(self.bn1[task](self.conv1(x)))
        x = F.relu(self.bn2[task](self.conv2(x)))
        x = F.max_pool2d(x, 2)
        return x


class three_conv_pool_split(nn.Module):
    def __init__(self, in_channel, F1, F2, F3, num_tasks=2):
        super(three_conv_pool_split, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, F1, kernel_size=3, padding='same')
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(F1), nn.BatchNorm2d(F1)])
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding='same')
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(F2), nn.BatchNorm2d(F2)])
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=3, padding='same')
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(F3), nn.BatchNorm2d(F3)])

    def forward(self, x, task):
        x = F.relu(self.bn1[task](self.conv1(x)))
        x = F.relu(self.bn2[task](self.conv2(x)))
        x = F.relu(self.bn3[task](self.conv3(x)))
        x = F.max_pool2d(x, 2)
        return x

class MultiVgg5R_film(nn.Module):
    def __init__(self):
        super(MultiVgg5R_film, self).__init__()
        # self.layer1 = two_conv_pool(3, 32, 32)
        # self.layer1 = two_conv_pool_film(3, 32, 32)
        self.layer1 = two_conv_pool_split(3, 32, 32)
        # self.layer2 = two_conv_pool_split(32, 64, 64)
        # self.layer2 = two_conv_pool_film(32, 64, 64)
        self.layer2 = two_conv_pool_split(32, 64, 64)
        # self.layer3 = three_conv_pool_split(64, 128, 128, 128)
        # self.layer3 = three_conv_pool_film(64, 128, 128, 128)
        self.layer3 = three_conv_pool_split(64, 128, 128, 128)
        # self.layer4 = three_conv_pool_split(128, 256, 256, 256)
        # self.layer4 = three_conv_pool_film(128, 256, 256, 256)
        self.layer4 = three_conv_pool_split(128, 256, 256, 256)

    def forward(self, x, task):
        # x = self.layer1(x)
        x = self.layer1(x, task)
        # x = self.layer2(x)
        x = self.layer2(x, task)
        # x = self.layer3(x)
        x = self.layer3(x, task)
        # x = self.layer4(x)
        x = self.layer4(x, task)
        x = x.view(-1, 1024)  # flatten
        return x

class MultiVgg5O(nn.Module):
    def __init__(self):
        super(MultiVgg5O, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




if __name__ == '__main__':
    model = MultiVgg5R()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('MultiVgg5R: ', pytorch_total_params)
    model = MultiVgg5R_film()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('MultiVgg5R_film: ', pytorch_total_params)
    model = MultiVgg5O()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('MultiVgg5O: ', pytorch_total_params)
