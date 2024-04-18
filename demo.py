import cmd
import os
from tqdm import tqdm

from core import data
from core.models import basic_cnn
import utils
from config import INPUT_SIZE, BATCH_SIZE

import torch
from torchsummary import summary
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
	"basic_cnn": basic_cnn.Model("basic_cnn", device),
}

train_df, val_df, test_df = data.get_dataframes_cub_200_2011()

train_ds = data.CustomImageDataset(train_df, f"./{data.DATA_FOLDER}/images", utils.data.data_transforms["train"])
val_ds = data.CustomImageDataset(val_df, f"./{data.DATA_FOLDER}/images", utils.data.data_transforms["val"])
test_ds = data.CustomImageDataset(test_df, f"./{data.DATA_FOLDER}/images", utils.data.data_transforms["val"])

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

models["basic_cnn"].create_model()
print(summary(models["basic_cnn"].model, (3, ) + INPUT_SIZE))
models["basic_cnn"].train({
	"train": train_dl,
	"val": val_dl,
}, 5)

models["basic_cnn"].save()
