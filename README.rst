Abstract
========

In this project, new fruit images were generated from random noise data
with GAN. Discriminator model with four convolutional layers for
downsample and one dense layer for classification was used to determine
fake and real images. Also, generator model with first dense layer used;
then, reshaped random noise data into something image-like data and
three Conv2DTranspose layers used to increase number of pixels to pixel
of target images and last Convolutional layer used to convert number of
image color channels to number of target color channels. Final model
after each ten epochs were saved and images displayed (up to 200
epochs). We could distinguish fruit generated images after 50 epochs;
but, images after 200 epochs were very similar to real images and we
could distinguish types of fruit.

Contains
========

Src contains source code to train GAN architecture

Models: contains generator.py and dictriminator.py to define Generator
and Discriminator models

Jupyter: you can find .ipynb format

Report: contains .doc file which explain which explains all parts of
code

Requirements
============

To train the GAN architecture, you need a fruits-360 dataset which
contains Training folder, you can download from
https://www.kaggle.com/moltean/fruits .
