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
import torchvision
nltk.download('wordnet')
import logging
from google.colab import files

import torch
import torchvision
from torchvision import transforms

def classify_image(image):
    """ 
    This functions classifies the image into one of the 1000 imagenet categories using inceptionv3
    input: image
    output: one hot vector of the class's label
    """
    # get the model and turn into eval mode
    inception_v3 = torchvision.models.inception_v3(pretrained=True)
    inception_v3.eval()

    # set the transformation
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # preprocess the image
    torch_image = torch.tensor(image).type(torch.float32)
    input_tensor = preprocess(torch_image[0])
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        inception_v3.to('cuda')

    # get the output 
    with torch.no_grad():
        output = inception_v3(input_batch)

    # softmax the output to get the maximim
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # change the output to one hot vector
    classification,classification_prop = torch.argmax(probabilities),probabilities[torch.argmax(probabilities)]
    one_hot_vector = torch.zeros((1,1000))
    one_hot_vector[0][classification]=1

    # return the one hot vector
    return one_hot_vector
