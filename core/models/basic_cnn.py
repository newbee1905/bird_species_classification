from . import Model as BasicModel
from utils.models import EarlyStopper

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from config import BATCH_SIZE

from tempfile import TemporaryDirectory
import time
from tqdm import tqdm
from os import path

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		# INPUT_SIZE _ 448, 448
		# all stride = 1
		self.conv1 = nn.Conv2d(3, 32, 5)
		# output size = (448 - 5 + 1) / 2 = 222
		self.conv2 = nn.Conv2d(32, 64, 3)
		# output size = (222 - 3 + 1) / 2 = 110
		self.conv3 = nn.Conv2d(64, 128, 3)
		# output size = (110 - 3 + 1) / 2 = 54
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(128 * 54 * 54, 1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 200)
		self.prelu1 = nn.PReLU(32)
		self.prelu2 = nn.PReLU(64)
		self.prelu3 = nn.PReLU(128)

	def forward(self, x):
		x = self.pool(self.prelu1(self.conv1(x)))
		x = self.pool(self.prelu2(self.conv2(x)))
		x = self.pool(self.prelu3(self.conv3(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.softmax(x)


class Model(BasicModel):
	def __init__(self, name, device):
		super().__init__(name, device)

	def create_model(self):
		self.model = Net()
		self.model = self.model.to(self.device)
	
	def train(self, dl, num_epochs=100):
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
		early_stopper = EarlyStopper(patience=15, min_delta=10)

		since = time.time()

		with TemporaryDirectory() as tempdir:
			best_model_params_path = path.join(tempdir, f'_{self.name}.pt')

			torch.save(self.model.state_dict(), best_model_params_path)
			best_acc = 0.0

			epoch = 0
			while epoch < num_epochs:
				print(f'Epoch {epoch}/{num_epochs - 1}')
				print('-' * 10)

				# Each epoch has a training and validation phase
				for phase in ['train', 'val']:
					if epoch >= num_epochs: # early stopping
						break

					if phase == 'train':
						self.model.train()  # Set model to training mode
					else:
						self.model.eval()   # Set model to evaluate mode

					running_loss = 0.0
					running_corrects = 0

					for i, data in enumerate(tqdm(dl[phase])):
						data[1] = data[1].type(torch.LongTensor)
						X, y = data[0].to(self.device), data[1].to(self.device)
						optimizer.zero_grad()

						with torch.set_grad_enabled(phase == 'train'):
							y_hat = self.model(X)
							loss = criterion(y_hat, y)
							y_hat = torch.argmax(y_hat, dim=1)

							if phase == 'train':
								loss.backward()
								optimizer.step()
							elif early_stopper.early_stop(loss):
								epoch = num_epochs
								break
						
						running_loss += loss.item() * X.size(0)
						running_corrects += torch.sum(y_hat == y.data)
					if phase == 'train':
						scheduler.step()

					ds_size = len(dl['phase'].dataset)
					epoch_loss = running_loss / ds_size
					epoch_acc = running_corrects.double() / ds_size

					print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

					# deep copy the model
					if phase == 'val' and epoch_acc > best_acc:
						best_acc = epoch_acc
						torch.save(model.state_dict(), best_model_params_path)

				epoch += 1
			print()

		time_elapsed = time.time() - since
		print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
		print(f'Best val Acc: {best_acc:4f}')

		# load best model weights
		self.model.load_state_dict(torch.load(best_model_params_path))
	
	def evaluate(self, dl):
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(tqdm(dl)):
				data[1] = data[1].type(torch.LongTensor)
				X, y = data[0].to(self.device), data[1].to(self.device)
				optimizer.zero_grad()

				y_hat = model(X)
				loss = criterion(y_hat, y)
				y_hat = torch.argmax(y_hat, dim=1)

				running_loss += loss.item() * X.size(0)
				running_corrects += torch.sum(y_hat == y.data)

		ds_size = len(dl.dataset)
		epoch_loss = running_loss / ds_size
		epoch_acc = running_corrects.double() / ds_size
	
	def save(self):
		torch.save(self.model.state_dict(), f"{self.name}.pt")
