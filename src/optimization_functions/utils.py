import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names)
import lpips
import cv2
import copy
import nltk
import time
import torchvision
nltk.download('wordnet')
import logging
from google.colab import files
from tqdm.notebook import tqdm
from src.helper_functions.utils import *
from src.transformations.utils import *
from src.classification.utils import *
from src.optimization_functions.utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_optim_on_variable(latent_var,ground_truth_image,opt,epochs,criterion,model_path,category,save_images_location):
    """ 
    This function runs the optimization on the latent space of a specific image 
    
    inputs
    latent_var:  input latent variable (1,128) tensor
    ground_truth_image: the image to optimize on
    opt: the optimizer to choose
    epochs: number of epochs
    model_path: the model to run the inversion on, biggan by default
    category: the category of the image
    criterion: list of losses needed for optimization (by default they are added together)
    lr: learning rate

    outputs
    optimized latent variable , loss list
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set the latent variable
    latent_var = Variable(latent_var.clone(),requires_grad=True)

    # set the optimizer and initialize the loss list
    optimizer = opt([latent_var],0.01)
    loss_list=[]
    t=time.time()

    # set the model
    if model_path=='biggan':
        model = BigGAN.from_pretrained('biggan-deep-256').to(device)
        model.eval()
    else:
        model = torch.load(model_path).to(device)
    if type(category) == type('str'):
        class_vector = torch.from_numpy(one_hot_from_names([category], batch_size=1)).to(device)
    else:
        class_vector=category.to(device)

    # set the ground truth image and initialize the variable that will store the best latent variable
    ground_truth_image=ground_truth_image.to(device)
    u_latent_var=latent_var.clone()
    loss_min=np.Inf

    for epoch_num in tqdm(range(epochs)):
        # initialize the optimizer
        optimizer.zero_grad()

        # predict the image
        if model_path=='biggan':
            predicted_image = model(latent_var.to(device),class_vector,1)
        else:
            predicted_image = model(latent_var.to(device)) 

        save_image=False
        # predicted image
        pred_image = np.transpose(predicted_image[0].cpu().detach().numpy(),(1,2,0))[:,:,[2,1,0]]*127.5+127.5
        # real image
        real_image = np.transpose(ground_truth_image[0].cpu().detach().numpy(),(1,2,0))[:,:,[2,1,0]]*127.5+127.5
        # save the image
        cv2.imwrite(f'{save_images_location}/{epoch_num}__left:fake__right:real.png',np.concatenate([pred_image,real_image],axis=1))
        
        # get the loss
        if type(criterion)==list:
            loss = sum([torch.sum(criterion_loss(predicted_image.view((1,3,256,256)),ground_truth_image.view((1,3,256,256)))**2) for criterion_loss in criterion])
        else:
            loss = torch.sum(criterion(predicted_image.view((1,3,256,256)),ground_truth_image.view((1,3,256,256)))**2)

        # optimize
        loss.backward(retain_graph=True)
        optimizer.step()
                
        # update loss list
        loss_list.append((loss.item(),latent_var))
        if loss.item()<loss_min:
            loss_min=loss.item()
            u_latent_var=latent_var.clone()

        # truncate the latent variable
        latent_var=latent_var.detach().cpu()
        latent_var[latent_var<-1]=np.random.rand()*2-1
        latent_var[latent_var>1]=np.random.rand()*2-1
        latent_var = Variable(latent_var.clone(),requires_grad=True)
        optimizer = opt([latent_var],0.01)

    return u_latent_var,loss_list

def get_image_transformations(image,epochs,dict_of_losses,category):
    """ 
    This function applies all the transformations to the images and saves the images

    input: image
    output: dictionary that consists of: (optimized latent variable, losses in every direction) 
    """
    
    latent_input = produce_truncated(0.5).to(device)
    dict_of_sides = {'original':None,'left':None,'right':None,'up':None,'down':None}
    dict_of_sides['original'] = run_optim_on_variable(latent_input,image,torch.optim.Adam,epochs,[dict_of_losses['lpips_alexnet'].to(device),dict_of_losses['L1']],'biggan',category,'output_images/original')[0]
    print('done original')
    for direction,f in [('up',translate_up),('down',translate_down),('right',translate_right),('left',translate_left)]:
        ground_truth_image = f(copy.copy(image))
        dict_of_sides[direction] = run_optim_on_variable(latent_input,ground_truth_image,torch.optim.Adam,epochs,[dict_of_losses['lpips_alexnet'].to(device),dict_of_losses['L1']],'biggan',category,f'output_images/original_to_{direction}')
        print(f'done {direction}')
