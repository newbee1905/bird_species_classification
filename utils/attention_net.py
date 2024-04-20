import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import ResNet, Bottleneck

import torch
from torch import nn
import torch.nn.functional as F

# This code is base of https://github.com/yangze0930/NTS-Net

class ResNet50(ResNet):
	def __init__(self):
		super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
		# self.load_state_dict(ResNet50_Weights.IMAGENET1K_V2.get_state_dict(check_hash=True))

	def forward(self, x):
		"""
		Outside of returning normal resnet50 output,
		also returning features
		"""
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		feature1 = x

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = nn.Dropout(p=0.5)(x)
		feature2 = x

		x = self.fc(x)

		return x, feature1, feature2

class ProposalNet(nn.Module):
	def __init__(self):
		super(ProposalNet, self).__init__()
		self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
		self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
		self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
		self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
		self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
		self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

	def forward(self, x):
		batch_size = x.size(0)
		d1 = F.relu(self.down1(x))
		d2 = F.relu(self.down2(d1))
		d3 = F.relu(self.down3(d2))
		t1 = self.tidy1(d1).view(batch_size, -1)
		t2 = self.tidy2(d2).view(batch_size, -1)
		t3 = self.tidy3(d3).view(batch_size, -1)
		return torch.cat((t1, t2, t3), dim=1)
