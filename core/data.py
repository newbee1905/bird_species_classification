from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection

from torchvision.transforms import v2
from torchvision import models

from torchvision.datasets.utils import download_url
import tarfile
import os
from os import path

import pandas as pd

from config import DATA_FOLDER, DATA_URL, DATA_FILE, TGZ_MD5


def check_cub_200_2011():
	return path.exists(path.join(os.getcwd(), DATA_FOLDER))


def download_cub_200_2011():
	if check_cub_200_2011():
		print(f"CUB_200_211 data is already downloaded and extracted.")
		return

	print("Downloading...")
	if not path.exists(path.join(os.getcwd(), DATA_FILE)):
		download_url(DATA_URL, os.getcwd(), DATA_FILE, TGZ_MD5)
	else:
		print(f"{DATA_FILE} is already downloaded.")

	print("Extracting...")
	if not path.exists(path.join(os.getcwd(), DATA_FOLDER)):
		with tarfile.open(path.join(os.getcwd(), DATA_FILE), "r:gz") as tar:
			tar.extractall(path=os.getcwd())
	else:
		print(f"{DATA_FOLDER} is already extracted.")

def get_dataframes_cub_200_2011():
	images = pd.read_csv(
		os.path.join(os.getcwd(), DATA_FOLDER, 'images.txt'),
		sep=' ',
		names=['img_id', 'filepath'],
	)
	image_class_labels = pd.read_csv(
		os.path.join(os.getcwd(), DATA_FOLDER, 'image_class_labels.txt'),
		sep=' ', 
		names=['img_id', 'target'],
	)
	train_test_split = pd.read_csv(
		os.path.join(os.getcwd(), DATA_FOLDER, 'train_test_split.txt'),
		sep=' ',
		names=['img_id', 'is_training_img'],
	)

	data = images.merge(image_class_labels, on='img_id')
	data = data.merge(train_test_split, on='img_id')

	test_df = data[data.is_training_img == 0]
	data = data[data.is_training_img == 1]
	train_df, val_df = model_selection.train_test_split(data, test_size=0.2)

	return train_df, val_df, test_df

class CustomImageDataset(Dataset):
	def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
		self.img_labels = img_labels
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
	
	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = path.join(self.img_dir, self.img_labels.iloc[idx, 1])
		image = read_image(img_path)
		label = self.img_labels.iloc[idx, 2] - 1
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label
