import os
import torch
from PIL import Image, ImageEnhance
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from data.base_dataset import BaseDataset, get_transform
from models import networks
import cv2
import numpy as np
#from util.visualizer import save_images
#from util import html

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#model = create_model(opt)      # create a model given opt.model and other options
#print(model.model_names)
#model.setup(opt)               # regular setup: load and print networks; create schedulers
# create a website
#web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
#webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
# test with eval mode. This only affects layers like batchnorm and dropout.
# For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
# For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    
transform = get_transform(opt, grayscale=False)

B_img = Image.open('jobs.jpg')#.convert('RGB')

enhancer = ImageEnhance.Brightness(B_img)
B_img = enhancer.enhance(1.3)

B = transform(B_img).unsqueeze(0)

#netG_B = netG_B.module

##########
netG_B_simpsons = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

netG_B_simpsons.load_state_dict(torch.load('checkpoints/1_simpsons2photo/latest_net_G_B.pth'))
netG_B_simpsons = netG_B_simpsons.to('cpu')
##########
netG_B_anime = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

netG_B_anime.load_state_dict(torch.load('checkpoints/1_anime2photo/latest_net_G_B.pth'))
netG_B_anime = netG_B_anime.to('cpu')
##########
netG_B_caricature = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

netG_B_caricature.load_state_dict(torch.load('checkpoints/1_caricature2photo/latest_net_G_B.pth'))
netG_B_caricature = netG_B_caricature.to('cpu')
##########
netG_B_cartoon = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

netG_B_cartoon.load_state_dict(torch.load('checkpoints/2_cartoon2photo/latest_net_G_B.pth'))
netG_B_cartoon = netG_B_cartoon.to('cpu')

with torch.no_grad(): 
    fake_A_simpsons = netG_B_simpsons(B)
    fake_A_anime = netG_B_anime(B)
    fake_A_caricature = netG_B_caricature(B)
    fake_A_cartoon = netG_B_cartoon(B)

img_simpsons = fake_A_simpsons[0].detach().numpy()
img_simpsons = np.moveaxis(img_simpsons, 0, -1)

img_anime = fake_A_anime[0].detach().numpy()
img_anime = np.moveaxis(img_anime, 0, -1)

img_caricature = fake_A_caricature[0].detach().numpy()
img_caricature = np.moveaxis(img_caricature, 0, -1)

img_cartoon = fake_A_cartoon[0].detach().numpy()
img_cartoon = np.moveaxis(img_cartoon, 0, -1)

#b,g,r = cv2.split(img)           # get b, g, r
#rgb_img1 = cv2.merge([r,g,b])




#img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#cv2.imshow('image',img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

B = B[0].detach().numpy()
B = np.moveaxis(B, 0, -1)

#B = (255 * (B+1)/2).astype(np.uint8)

print(B.shape)
print(img_simpsons.shape)
print(img_anime.shape)
print(img_caricature.shape)
print(img_cartoon.shape)
#final = np.hstack((B,img_simpsons, img_anime, img_caricature, img_cartoon))
f1 = B
f2 = np.hstack((img_simpsons, img_anime))
f3 = np.hstack((img_caricature, img_cartoon))
from matplotlib import pyplot as plt

plt.figure(figsize=(5,5))
plt.imshow(f1)
plt.axis('off')
plt.show()

plt.figure(figsize=(20,15))
plt.imshow(f2)
plt.axis('off')
plt.show()


plt.figure(figsize=(20,15))
plt.imshow(f3)
plt.axis('off')
plt.show()
    
