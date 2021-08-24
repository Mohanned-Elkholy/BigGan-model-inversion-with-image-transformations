# BigGan-model-inversion-with-image-transformations
This the torch implementation of (https://arxiv.org/abs/1907.07171). Model inversion has been one of the hot topics recently in deep learning. The idea behind model inversion is to find the closest image generated by a one of the state of the art generators to a given image. In most of the cases, the BigGan generator (https://arxiv.org/abs/1809.11096) is used for this problem.
![image](https://user-images.githubusercontent.com/47930821/130579758-40f77ba6-c1ad-4e99-8380-c7d80c51f13a.png)


---

# Advanced data augmentation using transformations in the latent space
It has been known that applying certain image transformations (like sliding left or right, center crop,..etc) are usually discrete and the image as a whole lacks continuity in some filters (like zero-padding). Applying this transformations in the latent space, however, is different because generative models are trained to produce realistic images. Thus, more ralistic image augmentation will happen in the latent space. In this repo, multiple transformations (sliding left, right, upwards, downwards and zooming) are done to a given image in the latent space of the BigGan generator. This data augmentation technique can be crucial in improving the accuracy of image classifiers. 

#add GIF here

---

# Prerequisites
1- python3 

2- CPU or NVIDIA GPU + CUDA CuDNN (GPU is recommended for faster inversion)

---

# Install dependencies
In this repo, a pretrained biggan in a specified library
```python
pip install torch torchvision matplotlib lpips numpy nltk cv2 pytorch-pretrained-biggan
```
---

# Classifying the input image
In this repo, BigGan is used to apply the model inversion. Since it is a conditional GAN that is trained on imagenet, the category of the image should be known before inverting the image. Therefore a classifier is used to assign a class to the image. In this repo, a state of the art Inception-V3 net is used. You can read more about inception net in this paper (https://arxiv.org/abs/1512.00567v3)

![inception](https://user-images.githubusercontent.com/47930821/130579434-678b3445-f432-42b1-96b5-7d4d7b5868bb.png)


# Training
#provide image to work on
```python
python train.py --image_path '/image.rgb' --loss ['l1','lpips_alexnet'] --transforms ['zoom','slide_left','slide_right','slide_upward','slide_downward'] --num_epochs 2000 --num_samples 6 --learning_rate 0.007 
```
---

# How does the optimization work
In the very beginning, a random input is chosen. Later the model is frozen and multiple backward propagations happen. The loss function in the back propagation tries to make the produced image and the target image as clos as possible. Since the model is frozen, the gradient updates only changes the latent input until the produced image matches a close representation in the latent space to the target image. 

---

# Truncation trick
Truncation trick is first introduced in the BigGan paper (https://arxiv.org/abs/1809.11096). Since the random input is chosen from a normal distribution, most of the training happens for random input values that range from (-1 to 1). In the inversion task, the latent input is truncated after each gradient update: (values that are less than -1 or more than 1 are set to a random value). This ensures the realisticity of the image as well as help avoid local minima in the generator manifold.

---

# Loss functions
In this repo, two loss functions are added together

# Pixel-wise loss function
There are multiple pixel-wise loss function that can be chosen for this task, but after multiple trials, L1 loss worked best. The reason behind this is because the gradient in the L1 loss doesn't depend on the value of the loss itself. Thus, the training becomes more consistent and avoids asymptotic behaviours.


<img src="https://render.githubusercontent.com/render/math?math=\sum_{n=1}^{N}\left|y_{true}-y_{predicted}\right|">

# Feature-wise loss function (LPIPS loss)
This loss function cares more about the features of the produced image by applying a norm difference between the internal convolutional features of a pretrained when applied on both the real and the fake image. For the sake of the speed, alexnet is applied since VGG net is much more complicated. You can find more about lpips loss in this paper (https://arxiv.org/abs/1801.03924)

![lpips](https://user-images.githubusercontent.com/47930821/130575694-50b818d2-f0ff-4b09-b662-341becfa18a7.jpg)



