from . import Model as BasicModel
from utils.models import EarlyStopper

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from config import BATCH_SIZE

from collections import defaultdict
from tempfile import TemporaryDirectory
import time
from tqdm import tqdm
from os import path
from typing import Tuple

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
		self.model = models.resnet50(weights='IMAGENET1K_V2')

		# Freezing layers except the final fully connected layer
		for param in self.model.parameters():
			param.requires_grad = False
		num_ftrs = self.model.fc.in_features
		self.model.fc = nn.Linear(num_ftrs, 200)

		self.model = self.model.to(self.device)

		# Configuration for training
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=3e-3, weight_decay=0.0001)
		self.early_stopper = EarlyStopper(patience=15, min_delta=10)
	
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
			- is_early_stop (float): check if stop after epoch for early stop

		:rtype: Tuple[float, float, bool]
		"""
		if phase == 'train':
			self.model.train()
		else:
			self.model.eval()

		is_early_stop = False
		running_loss = 0.0
		running_corrects = 0
		class_corrects = defaultdict(int)
		class_counts = defaultdict(int)

		for i, data in enumerate(tqdm(dl)):
			X, y = data[0].to(self.device), data[1].to(self.device)
			self.optimizer.zero_grad()

			with torch.set_grad_enabled(phase == 'train'):
				y_hat = self.model(X)
				loss = self.criterion(y_hat, y)
				y_hat = torch.argmax(y_hat, dim=1)

				if phase == 'train':
					loss.backward()
					self.optimizer.step()
			
			running_loss += loss.item() * X.size(0)
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

		if phase == 'val' and self.early_stopper.early_stop(epoch_loss):
			is_early_stop = True
		
		print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, Apc: {epoch_apc:.4f}")
		return epoch_loss, epoch_acc, epoch_apc, is_early_stop

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
				val_loss, val_acc, _, is_early_stop = self._train("val", val_dl)

				if val_acc > best_acc:
					best_acc = val_acc 
					torch.save(self.model.state_dict(), best_model_params_path)

				if is_early_stop:
					break
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
		print("Feature Extracting...")
		self._fit(train_dl, val_dl, num_epochs)

		print("Fine tuning...")

		# Enable Training for last 30% layers
		params = list(self.model.parameters())
		ft_layers = int(len(params) * 0.3)

		for param in params[-ft_layers:]:
			param.requires_grad = True

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
