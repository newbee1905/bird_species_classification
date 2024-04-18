import cmd
import os

from core import data
from core.models import basic_cnn
import utils
from config import INPUT_SIZE, BATCH_SIZE

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
	"basic_cnn": basic_cnn.Model("basic_cnn", device),
}

class BirdSpeciesClassificationCLI(cmd.Cmd):
	prompt = 'bird-classification-species> '
	intro = 'Welcome to Bird Classification Species. Type "help" for available commands.'

	def __init__(self):
		super().__init__()
		self.train_dl = None
		self.val_dl = None
		self.test_dl = None

	def do_download(self, line):
		data.download_cub_200_2011()
	
	def do_load_data(self, line):
		self.train_df, self.val_df, self.test_df = data.get_dataframes_cub_200_2011()

		self.train_ds = data.CustomImageDataset(self.train_df, f"./{data.DATA_FOLDER}/images", utils.data.data_transforms["train"])
		self.val_ds = data.CustomImageDataset(self.val_df, f"./{data.DATA_FOLDER}/images", utils.data.data_transforms["val"])
		self.test_ds = data.CustomImageDataset(self.test_df, f"./{data.DATA_FOLDER}/images", utils.data.data_transforms["val"])
		
		self.train_dl = DataLoader(
			self.train_ds,
			batch_size=BATCH_SIZE,
			shuffle=True,
			num_workers=2,
		)
		self.val_dl = DataLoader(
			self.val_ds,
			batch_size=BATCH_SIZE,
			shuffle=True,
			num_workers=2,
		)
		self.test_dl = DataLoader(
			self.test_ds,
			batch_size=BATCH_SIZE,
			shuffle=True,
			num_workers=2,
		)

	def do_quit(self, line):
		"""Exit the CLI."""
		return True

	def do_exit(self, line):
		"""Exit the CLI."""
		return True
	
	def do_train(self, model):
		if model not in models:
			print("Please select valid model")	
			print(f"Here is the list of the models: {models.keys()}")	
		elif self.train_dl == None:
			print("Please load data first")	
		else:
			models[model].train({
				"train": self.train_dl,
				"val": self.val_dl,
			})

	def do_evaluate(self, model):
		if model not in models:
			print("Please select valid model")	
			print(f"Here is the list of the models: {models.keys()}")	
		elif self.test_dl == None:
			print("Please load data first")	
		else:
			models[model].evaluate(self.test_dl)

	def do_save(self, model):
		if model not in models:
			print("Please select valid model")	
			print(f"Here is the list of the models: {models.keys()}")	
		else:
			models[model].save()

	def postcmd(self, stop, line):
		# Add an empty line for better readability
		print()
		return stop

if __name__ == '__main__':
	BirdSpeciesClassificationCLI().cmdloop()
