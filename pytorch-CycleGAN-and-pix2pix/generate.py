import os
import torch
from PIL import Image
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

B_img = Image.open('datasets/caricature2photo/trainB/Chuck_Norris_r_17.jpg').convert('RGB')

B = transform(B_img).unsqueeze(0)

netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
netG_B = netG_B.module
netG_B.load_state_dict(torch.load('checkpoints/1_caricature2photo/latest_net_G_B.pth'))

print(B.shape)
netG_B = netG_B.to('cpu')

with torch.no_grad(): 
    fake_A = netG_B(B)

img = fake_A[0].detach().numpy()
img = np.moveaxis(img, 0, -1)

#b,g,r = cv2.split(img)           # get b, g, r
#rgb_img1 = cv2.merge([r,g,b])


img = (255 * (img+1)/2).astype(np.uint8)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

from matplotlib import pyplot as plt

plt.figure()
plt.imshow(img)
plt.show()
    
