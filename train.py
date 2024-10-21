from setup import setup
from download_dataset import download_dataset
from prepare_data import prepare_data
from get_dataloaders import get_dataloaders
from model.OpenPose import OpenPose
from engine import train
from torchvision import transforms
from utils import load_config
import torch

config_data = load_config()

ANN_FILE = config_data['ann_file']
ANN_SAVE_FILE = config_data['ann_save_file']
DATA_URL = config_data['data_url']
ANNOT_URL = config_data['annot_url']
SAVE_DIR = config_data['save_dir']
DATA_N = config_data['data_n']
ANNOT_N = config_data['annot_n']
BATCH_SIZE = config_data['batch_size']
TRAIN_SIZE = config_data['train_size']
RESIZE = (config_data['resize_x'], config_data['resize_y'])
OG_RESIZE = (config_data['og_resize_x'], config_data['og_resize_y'])
O = config_data['o']

STAGES = config_data['stages']
FEATURE_CHANNELS = config_data['feature_channels']
EPOCHS = config_data['epochs']
FACTOR = config_data['factor']
PATIENCE = config_data['patience']
L_R = config_data['l_r']
MODEL_NAME = config_data['name']

TRANSFORM = transforms.Compose([
    transforms.Resize(OG_RESIZE),
    transforms.ToTensor(),
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

setup()

download_dataset(DATA_URL, ANNOT_URL, SAVE_DIR, DATA_N, ANNOT_N)

prepare_data(ANN_FILE, ANN_SAVE_FILE, O, RESIZE)

train_dataloader, test_dataloader, keypoints, connections = get_dataloaders(ANN_SAVE_FILE, ANN_FILE, TRANSFORM, BATCH_SIZE, train_size=TRAIN_SIZE)

model = OpenPose(FEATURE_CHANNELS, STAGES, len(keypoints), len(connections)).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=L_R)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE)

criterion = torch.nn.MSELoss()

train(EPOCHS, model, optimizer, criterion, train_dataloader, test_dataloader, MODEL_NAME, DEVICE, scheduler)
