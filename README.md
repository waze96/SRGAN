# SRGAN
Super Resolution Generative Adversial Network

We are

    Ramon Vallés - 205419
    Eduard Masip - 207322
    Ferran Enguix - 195659
    
This repository includes:

    The project code in .ipynb format.
    The project code in .py format.
    Two scripts in .py format to create new .mat files and slice images, both commented.
    The .mat file with 10K of images from DIV2K Dataset (TRAINING).
    The .mat file with 100 of images from DIV2K Dataset (TEST).
    The .mat file with few images of pixel-art to test the results in other kind of images (TEST).
    
The idea of this project is obtain images with more resolution applying SRGAN.
We use the architecture of this paper, with some changes: https://arxiv.org/abs/1609.04802.
This SRGAN is trained and programed to increment by 2 the resolution, we use images of 64x64 to obtain 128x128 images.
But it can be used to obtain 128x128 to 256x256, we can't test the results.

To upscale the resolution more than 2 is necessary change a little bit the architecture!!

    
