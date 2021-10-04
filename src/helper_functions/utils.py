
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


def produce_truncated(truncation_value):
    """ this function produces a truncated output """
    print(3)
    output_vector = torch.normal(0,1,(1,128)).type(torch.FloatTensor)
    output_vector[output_vector>truncation_value]=truncation_value
    output_vector[output_vector<-truncation_value]=-truncation_value
    return Variable(output_vector,requires_grad=True)
    
def produce_image(category,plot=False,truncation=0.5,latent_input=None):
    """ this function produces an image from the biggan space given a latent input. If not, it produces a new image """
    model = BigGAN.from_pretrained('biggan-deep-256')
    class_vector = torch.from_numpy(one_hot_from_names([category], batch_size=1))
    if latent_input == None:
        latent_input = produce_truncated(truncation)
    output = model(latent_input, class_vector, 1.0)
    # renormalize it and switch channels
    if plot:
        image = np.transpose(output[0].detach().numpy(),(1,2,0))*0.5+0.5
        plt.imshow(image)
        plt.show()
        
    return output

def show_image(image):
    """ this function shows an image """
    plt.imshow(np.transpose(image[0].detach().numpy(),(1,2,0))*0.5+0.5)
    plt.show()

def show_both_images(image_real,image_fake):
    """ This function shows both images, real and fake """
    image_real_adjusted = np.transpose(image_real[0].detach().cpu().numpy(),(1,2,0))*0.5+0.5
    image_fake_adjusted = np.transpose(image_fake[0].detach().cpu().numpy(),(1,2,0))*0.5+0.5
    print("Left: Fake, Right: real")
    out_image = np.concatenate([image_fake_adjusted,image_real_adjusted],1)
    plt.imshow(out_image)
    plt.show()

def adjust_image(image_path=None,blur=False):
    """ This function adjusts an image for you. if no path specified, it uploads an image if you are using jupiter notebook and later adjust it """
    if image_path == None:
        uploaded = files.upload()
        for k, v in uploaded.items():
            open(k, 'wb').write(v)
        image_path = list(uploaded.keys())[0]

    # normalize the image
    image = cv2.imread(image_path)/127.5-1

    # resize the image
    image = cv2.resize(image,(256,256))
    if blur:
        image=cv2.blur(image, (30,30))

    # transpose the image and make it in one batch tensor (following torch format for the biggan)
    image = np.transpose(image,(2,0,1))
    image = image[[2,1,0],:,:]
    image = np.expand_dims(image,axis=0)
    return torch.tensor(image).type(torch.float32)

def list_of_tensors(tensor_1,tensor_2,n=200):
    # get the intermediate line of tensors between two tensors
    return [tensor_1+(tensor_2-tensor_1)*i/n for i in range(n)]

def save_list_of_images(tensors,category,path):
    """ this function takes list of tensor with their category and save their corrosponding biggan images in a given path 
    input
    tensors: list of tensors
    category: the category of the image
    path: path of the images to be saved in

    """
    model = BigGAN.from_pretrained('biggan-deep-256')
    for i,latent_var in enumerate(tensors):

        predicted_image = model(latent_var,category,1)
        if type(category) == type('str'):
            class_vector = torch.from_numpy(one_hot_from_names([category], batch_size=1))
        else:
            class_vector=category
        cv2.imwrite(f'{path}/{i}.png',np.transpose(predicted_image[0].detach().cpu().numpy(),(1,2,0))[:,:,[2,1,0]]*127.5+127.5)
    print(f'done saving in {path}')

