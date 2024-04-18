from config import INPUT_SIZE

from torchvision.transforms import v2 
import torch

def get_label(path):
  return " ".join(path.split("_")[:-2])

data_transforms = {
	'train': v2.Compose([
		v2.RandomResizedCrop(size=INPUT_SIZE, antialias=True),
		v2.RandomHorizontalFlip(p=0.5),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': v2.Compose([
		v2.CenterCrop(size=INPUT_SIZE),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

