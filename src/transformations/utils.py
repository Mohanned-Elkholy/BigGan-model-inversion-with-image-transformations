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

def get_avg_pixel(image):
    return torch.mean(image[:,[0],:,:]).item(),torch.mean(image[:,[1],:,:]).item(),torch.mean(image[:,[2],:,:]).item()

def translate_right(image,ratio=0.1):
    return torchvision.transforms.functional.affine(image,0,[256*ratio,0],1,shear=[0],fill=get_avg_pixel(image))

def translate_left(image,ratio=0.1):
    return torchvision.transforms.functional.affine(image,0,[-256*ratio,0],1,shear=[0],fill=get_avg_pixel(image))

def translate_up(image,ratio=0.1):
    return torchvision.transforms.functional.affine(image,0,[0,-256*ratio],1,shear=[0],fill=get_avg_pixel(image))

def translate_down(image,ratio=0.1):
    return torchvision.transforms.functional.affine(image,0,[0,256*ratio],1,shear=[0],fill=get_avg_pixel(image))

def zoom_down(image,ratio=0.1):
    """ this function zooms down the image """
    # apply the padding
    image =  torchvision.transforms.Pad(int(256*ratio),fill=0)(image)
    # resize the image
    image =  torchvision.transforms.Resize(256)(image)
    # replace the padding with the average pixels
    new_image = torch.zeros(image.shape)
    mean_colors = get_avg_pixel(image)
    new_image[:,[0],:,:]=mean_colors[0]
    new_image[:,[1],:,:]=mean_colors[1]
    new_image[:,[2],:,:]=mean_colors[2]
    p=int(256*ratio)
    margin = int(256*p/(256+2*p))
    new_image[:,:,margin:-margin,margin:-margin]=image[:,:,margin:-margin,margin:-margin]
    return new_image
    

