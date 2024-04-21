from . import Model as BasicModel
from utils.attention_net import ResNet50, ProposalNet
from utils.anchors import generate_default_anchor_maps, hard_nms

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch

import numpy as np
from config import CAT_NUM, PROPOSAL_NUM

from collections import defaultdict
from tempfile import TemporaryDirectory
import time
from tqdm import tqdm
import os
from os import path
from typing import Tuple

# This code is base of https://github.com/yangze0930/NTS-Net

class AttentionNet(nn.Module):
	def __init__(self, top_n=4):
		super(AttentionNet, self).__init__()

		self.base_model = ResNet50()
		num_ftrs = self.base_model.fc.in_features

		self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
		self.base_model.fc = nn.Linear(num_ftrs, 200)
		self.proposal_net = ProposalNet()
		
		self.top_n = top_n

		self.concat_net = nn.Linear(num_ftrs * (CAT_NUM + 1), 200)
		self.partcls_net = nn.Linear(num_ftrs, 200)

		_, edge_anchors, _ = generate_default_anchor_maps()
		self.pad_side = 224
		self.edge_anchors = (edge_anchors + 224).astype(np.int64)

	def forward(self, x):
		"""
		Forward pass of the AttentionNet model.

		:param
			- x: Input tensor
		:type
			- x: torch.Tensor

		:return
			- raw_logits: output of base model
			- concat_logits: output of concatnated features
			- part_logits: output logits from each features (partcls_net)
			- top_n_index: indices of top N features
			- top_n_props: top N features
		"""
		resnet_out, rpn_feature, feature = self.base_model(x)

		x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
		batch = x.size(0)

		# Reshaping rpn to shape: batch * nb_anchor
		rpn_score = self.proposal_net(rpn_feature.detach())
		all_cdds = [
			np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
			for x in rpn_score.data.cpu().numpy()
		]
		top_n_cdds = [hard_nms(x, topn=self.top_n, iou_thresh=0.25) for x in all_cdds]
		top_n_cdds = np.array(top_n_cdds)
		top_n_index = top_n_cdds[:, :, -1].astype(np.int64)
		top_n_index = torch.from_numpy(top_n_index).cuda()
		top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)

		part_imgs = torch.zeros([batch, self.top_n, 3, 224, 224]).cuda()
		for i in range(batch):
			for j in range(self.top_n):
				[y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int64)
				part_imgs[i:i + 1, j] = F.interpolate(
					x_pad[i:i + 1, :, y0:y1, x0:x1],
					size=(224, 224),
					mode='bilinear',
					align_corners=True,
				)
		part_imgs = part_imgs.view(batch * self.top_n, 3, 224, 224)

		_, _, part_features = self.base_model(part_imgs.detach())
		part_feature = part_features.view(batch, self.top_n, -1)
		part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
		part_feature = part_feature.view(batch, -1)

		# concat_logits have the shape: B*200
		concat_out = torch.cat([part_feature, feature], dim=1)
		concat_logits = self.concat_net(concat_out)
		raw_logits = resnet_out

		# part_logits have the shape: B*N*200
		part_logits = self.partcls_net(part_features).view(batch, self.top_n, -1)
		return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]

def list_loss(logits, targets):
	temp = F.log_softmax(logits, -1)
	loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
	return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
	loss = Variable(torch.zeros(1).cuda())
	batch_size = score.size(0)
	for i in range(proposal_num):
		targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
		pivot = score[:, i].unsqueeze(1)
		loss_p = (1 - pivot + score) * targets_p
		loss_p = torch.sum(F.relu(loss_p))
		loss += loss_p
	return loss / batch_size

class Model(BasicModel):
	def __init__(self, name, device):
		"""
		Initialize Model object.

		:param
			- name: Name of the model.
			- device: Device to use for training and inference.

		:type
			- name: str
			- device: torch.Device
		"""
		super().__init__(name, device)

	def create_model(self):
		"""
		Initialize Model architecture.
		"""
		self.model = AttentionNet(top_n=PROPOSAL_NUM)


		# Reusing trained transfer learning resnet50 model
		if path.exists(path.join(os.getcwd(), "model_transfer_cnn.pth")):
			num_ftrs = self.model.base_model.fc.in_features
			self.model.base_model.fc = nn.Linear(num_ftrs, 200)

			self.model.base_model.load_state_dict(torch.load("./model_transfer_cnn.pth"))

			params = list(self.model.base_model.parameters())
			for param in params:
				param.requires_grad = False
			for param in params[-int(0.3 * len(params)):]:
				param.requires_grad = True
		else:
			self.model.base_model.load_state_dict(ResNet50_Weights.IMAGENET1K_V2.get_state_dict(check_hash=True))

			params = list(self.model.base_model.parameters())
			for param in params:
				param.requires_grad = False
			params[-1] = True

			num_ftrs = self.model.base_model.fc.in_features
			self.model.base_model.fc = nn.Linear(num_ftrs, 200)

		self.model = self.model.to(self.device)

		# Configuration for training
		self.criterion = nn.CrossEntropyLoss()

		self.raw_parameters = list(self.model.base_model.parameters())
		self.part_parameters = list(self.model.proposal_net.parameters())
		self.concat_parameters = list(self.model.concat_net.parameters())
		self.partcls_parameters = list(self.model.partcls_net.parameters())

		self.raw_optimizer = optim.Adam(self.raw_parameters, lr=3e-3, weight_decay=1e-4)
		self.part_optimizer = optim.Adam(self.part_parameters, lr=3e-3, weight_decay=1e-4)
		self.concat_optimizer = optim.Adam(self.concat_parameters, lr=3e-3, weight_decay=1e-4)
		self.partcls_optimizer = optim.Adam(self.partcls_parameters, lr=3e-3, weight_decay=1e-4)

	
	def _train(self, phase, dl):
		"""
		Perform training or validating for one epoch

		:param
			- phase: Phase of operation ("train" or "val").
			- dl: DataLoader object for the phase.

		:type
			- phase: str
			- dl: torch.utils.data.Dataloader

		:return
			- epoch_loss: mean loss of that epoch
			- epoch_acc: mean accuarcy of that epoch
			- epoch_apc: mean accuarcy per class of that epoch

		:rtype: Tuple[float, float, float]
		"""
		if phase == 'train':
			self.model.train()
		else:
			self.model.eval()

		running_loss = 0.0
		running_corrects = 0
		class_corrects = defaultdict(int)
		class_counts = defaultdict(int)

		for i, data in enumerate(tqdm(dl)):
			X, y = data[0].to(self.device), data[1].to(self.device)
			batch_size = X.size(0)
			self.raw_optimizer.zero_grad()
			self.part_optimizer.zero_grad()
			self.concat_optimizer.zero_grad()
			self.partcls_optimizer.zero_grad()

			with torch.set_grad_enabled(phase == 'train'):
				raw_logits, concat_logits, part_logits, _, top_n_prob = self.model(X)

				part_loss = list_loss(
					part_logits.view(batch_size * PROPOSAL_NUM, -1),
					y.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)
				).view(batch_size, PROPOSAL_NUM)

				raw_loss = self.criterion(raw_logits, y)
				concat_loss = self.criterion(concat_logits, y)

				rank_loss = ranking_loss(top_n_prob, part_loss)
				partcls_loss = self.criterion(
					part_logits.view(batch_size * PROPOSAL_NUM, -1),
					y.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1),
				)

				total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

				if phase == 'train':
					total_loss.backward()
					self.raw_optimizer.step()
					self.part_optimizer.step()
					self.concat_optimizer.step()
					self.partcls_optimizer.step()
			
			y_hat = torch.argmax(concat_logits, dim=1)
			running_loss += concat_loss.item() * X.size(0)
			running_corrects += torch.sum(y_hat == y.data)

			for i in range(200):
				class_mask = (y == i)
				class_corrects[i] += torch.sum(y_hat[class_mask] == y[class_mask]).item()
				class_counts[i] += torch.sum(class_mask).item()


		ds_size = len(dl.dataset)
		epoch_loss = running_loss / ds_size
		epoch_acc = running_corrects.double() / ds_size

		class_acc = {i: class_corrects[i] / class_counts[i] if class_counts[i] != 0 else 0 for i in range(200)}

		epoch_apc = sum(class_acc.values()) / len(class_acc)

		print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, Apc: {epoch_apc:.4f}")
		return epoch_loss, epoch_acc, epoch_apc

	def _fit(self, train_dl, val_dl, num_epochs):
		"""
		Fit the model	to training data

		:param
			- train_dl: Dataloader for training data
			- val_dl: DataLoader for validating data.
			- num_epochs: DataLoader object for the phase.

		:type
			- train_dl: torch.utils.data.Dataloader
			- val_dl: torch.utils.data.Dataloader
			- num_epochs: int
		"""
		with TemporaryDirectory() as tempdir:
			since = time.time()

			best_model_params_path = path.join(tempdir, f'_model_{self.name}.pth')
			torch.save(self.model.state_dict(), best_model_params_path)

			best_acc = 0.0
			epoch = 0

			while epoch < num_epochs:
				print(f"Epoch {epoch + 1}/{num_epochs}")
				
				self._train("train", train_dl)
				val_loss, val_acc, _ = self._train("val", val_dl)

				if val_acc > best_acc:
					best_acc = val_acc 
					torch.save(self.model.state_dict(), best_model_params_path)

				epoch += 1

			self.model.load_state_dict(torch.load(best_model_params_path))

			time_elapsed = time.time() - since
			print(f'Complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
			print(f'Best val Acc: {best_acc:4f}')
		self.save()


	def train(self, train_dl, val_dl, num_epochs=20):
		"""
		Train the model

		:param
			- train_dl: Dataloader for training data
			- val_dl: DataLoader for validating data.
			- num_epochs: DataLoader object for the phase.

		:type
			- train_dl: torch.utils.data.Dataloader
			- val_dl: torch.utils.data.Dataloader
			- num_epochs: int
		"""
		self._fit(train_dl, val_dl, num_epochs)

	def evaluate(self, dl):
		"""
		Evaluate the model

		:param
			- dl: Dataloader for testing data

		:type
			- dl: torch.utils.data.Dataloader
		"""
		self.load()
		self._train("val", dl)
	
	def load(self):
		"""Load trained mode"""
		self.model.load_state_dict(torch.load(f"model_{self.name}.pth"))

	def save(self):
		"""Save trained mode"""
		torch.save(self.model.state_dict(), f"model_{self.name}.pth")
