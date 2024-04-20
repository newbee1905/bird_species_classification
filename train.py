import os
import argparse

from core import data
from core.data import CustomImageDataset
from core.models import transfer_cnn, attention_net
import utils
from config import INPUT_SIZE, BATCH_SIZE
from config import IMAGE_FOLDER

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = {
	"transfer_cnn": transfer_cnn.Model("transfer_cnn", device),
	"attention_net": attention_net.Model("attention_net", device),
}

if not data.check_cub_200_2011():
	print(f"Please run download.py to download the dataset first")
	os.exit()

train_df, val_df, test_df = data.get_dataframes_cub_200_2011()

train_ds = CustomImageDataset(train_df, IMAGE_FOLDER, utils.data.data_transforms["train"])
val_ds = CustomImageDataset(val_df, IMAGE_FOLDER, utils.data.data_transforms["val"])
test_ds = CustomImageDataset(test_df, IMAGE_FOLDER, utils.data.data_transforms["val"])

train_dl = DataLoader(
	train_ds,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=2,
)
val_dl = DataLoader(
	val_ds,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=2,
)
test_dl = DataLoader(
	test_ds,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=2,
)

parser = argparse.ArgumentParser(
	prog="bird-classification-training",
	description="Bird Classification Program",
)
parser.add_argument("model")
parser.add_argument("-n", "--num-epochs", default=10, type=int)
parser.add_argument("-e", "--evaluate-only", action="store_true")

args = parser.parse_args()

if args.model not in model_list:
	print(f"Please select from this list of models {list(model_list.keys())}")
	exit()

model_list[args.model].create_model()

if not args.evaluate_only:
	model_list[args.model].train(train_dl, val_dl, args.num_epochs)

model_list[args.model].evaluate(test_dl)
