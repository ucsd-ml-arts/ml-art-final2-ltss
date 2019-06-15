# Final Project

Lucas Tindall, ltindall@ucsd.edu
Samuel Sunarjo, ssunarjo@eng.ucsd.edu 

## Abstract Proposal

In this project we build a system for real time translation of human faces to cartoon stylized faces. We trained CycleGANs on different cartoon styles to perform the transformations. The end product is a real time application which displays a camera feed where real human faces are replaced by the selected cartoon style. 


## Project Report

[Report Link](ECE_188_Final_Report.pdf)

## Model/Data


- [CycleGAN](pytorch-CycleGAN-and-pix2pix) - Folder containing the Pytorch implementation of CycleGAN written by [junyanz](https://github.com/junyanz)
- [CartoonGAN](pytorch-CartoonGAN) - Folder containing the Pytorch implementation of CartoonGAN written by [znxlwm](https://github.com/znxlwm)  
- [model checkpoints](pytorch-CycleGAN-and-pix2pix/checkpoints) - Model files used in the generation scripts below. 

The datasets we used for training can be found in the following repositories
- [CaVI Dataset](https://github.com/lsaiml/CaVINet): includes the real human faces used in all models and the hand drawn caricatures
- [Anime Face Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset): includes anime faces
- [Simpsons Faces Dataset](https://www.kaggle.com/kostastokis/simpsons-faces): includes faces from The Simpsons 
- [Cartoon Set Dataset](https://google.github.io/cartoonset/): includes cartoon avatar faces 

## Code

- [camera_gen.py](pytorch-CycleGAN-and-pix2pix/camera_gen.py) - Script to overlay transformations on a web cam video feed. 
```bash
cd pytorch-CycleGAN-and-pix2pix
python camera_gen.py --dataroot . --load_size 129 --crop_size 128 --no_dropout --gpu_ids -1
```

- [generate.py](pytorch-CycleGAN-and-pix2pix/generate.py) - Script to generate transformations on a static image. 
```bash
cd pytorch-CycleGAN-and-pix2pix
python generate.py  --dataroot . --load_size 140 --crop_size 128 --no_dropout --gpu_ids -1
```

## Results

[WebCam video](output.avi) - An example video(.avi) of the webcam feed overlayed with the cartoon style transformations.  

## Technical Notes

Runs on the datahub. 

## Reference

[Pytorch implementation of CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 

[Pytorch implementation of CartoonGAN](https://github.com/znxlwm/pytorch-CartoonGAN)

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) 
