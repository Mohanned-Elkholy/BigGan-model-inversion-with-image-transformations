
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names)
import lpips
import cv2
import nltk
import time
from google.colab.patches import cv2_imshow
import torchvision
nltk.download('wordnet')
import logging
from google.colab import files

from src.helper_functions.utils import *
from src.transformations.utils import *
from src.classification.utils import *
from src.optimization_functions.utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--image_path',type=str, required = True)


opt = parser.parse_args()

# get the image
ground_truth_image = adjust_image(opt.image_path)
# get the latent variable
latent_var = produce_truncated(1)
# dictionary
dict_of_losses = {
'lpips_alexnet' : lpips.LPIPS(net='alex'),
'MSE': torch.nn.MSELoss(),
'L1': torch.nn.L1Loss(),
}

dict_of_sides={}

category = classify_image(ground_truth_image)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
get_image_transformations(ground_truth_image.to(device),300,dict_of_losses,category=category)

