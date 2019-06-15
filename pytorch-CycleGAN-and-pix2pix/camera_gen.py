import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import sys
import matplotlib.pyplot as plt
import numpy as np

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

from PIL import ImageFilter, ImageEnhance

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0


#############
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


netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
#netG_B = netG_B.module

style="Simpsons"

netG_B.load_state_dict(torch.load('checkpoints/1_simpsons2photo/latest_net_G_B.pth'))

netG_B = netG_B.to('cpu')

##########

plt.figure()

font = cv2.FONT_HERSHEY_SIMPLEX

print("\n\n\n\n\n\n############################################\n")
print("Usage: \n")
print("Press 'a' to switch to anime filter")
print("Press 's' to switch to The Simpsons filter")
print("Press 'h' to switch to hand drawn filter") 
print("Press 'c' to switch to cartoon filter")

sys.stdout.flush()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    extra = 10
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        
        #print("x = %d, y = %d, w = %d, h = %d" % (x,y,w,h))
        #print(frame.shape)
        #print("[%d : %d][%d : %d]" % (y-extra,y+h+extra,x-extra,x+w+extra))
        
        x_start = x-extra if x-extra >= 0 else 0
        y_start = y-extra if y-extra >= 0 else 0 
        
        x_end = x+w+extra if x+w+extra <= frame.shape[1] else frame.shape[1]
        y_end = y+h+extra if y+h+extra <= frame.shape[0] else frame.shape[0]
        
        just_face = np.array(frame[y_start:y_end,x_start:x_end,:])
        
        face_pil = Image.fromarray(just_face)
        face_pil = face_pil.filter(ImageFilter.SHARPEN)
        #bright = ImageEnhance.Brightness(face_pil)
        #face_pil = bright.enhance(1)
        
        B = transform(face_pil).unsqueeze(0)
        #B_img = Image.open('Chuck_Norris_test.jpg').convert('RGB')
        #B = transform(B_img).unsqueeze(0)
        
        with torch.no_grad(): 
            fake_A = netG_B(B)

        new_img = fake_A[0].detach().numpy()
        new_img = np.moveaxis(new_img, 0, -1)
        
        new_img = (255 * (new_img+1)/2).astype(np.uint8)
        
        
        new_img = Image.fromarray(new_img).resize((x_end-x_start, y_end-y_start))
        

        #print(just_face.shape)
        #sys.stdout.flush()
        #plt.imshow(new_img)
        #plt.show()
        
        
        #new_img = cv2.cvtColor(np.array(new_img),cv2.COLOR_BGR2RGB)
        frame[y_start:y_end,x_start:x_end,:] = new_img
        
        #cv2.rectangle(frame, (x-extra, y-extra), (x+w+extra, y+h+extra), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.putText(frame,style,(10,470), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('Video', frame)
    
    for i in range(4): 
        out.write(frame)


    keypress = cv2.waitKey(5) & 0xFF
    if keypress == ord('a'): 
        netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        #netG_B = netG_B.module
        netG_B.load_state_dict(torch.load('checkpoints/1_anime2photo/latest_net_G_B.pth'))

        netG_B = netG_B.to('cpu')
        
        style="Anime"
        
    if keypress == ord('s'): 
        netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        #netG_B = netG_B.module
        netG_B.load_state_dict(torch.load('checkpoints/1_simpsons2photo/latest_net_G_B.pth'))

        netG_B = netG_B.to('cpu')
        style= "Simpsons"
        
    if keypress == ord('h'): 
        netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        #netG_B = netG_B.module
        netG_B.load_state_dict(torch.load('checkpoints/1_caricature2photo/latest_net_G_B.pth'))

        netG_B = netG_B.to('cpu')
        style= "Hand drawn"
        
    if keypress == ord('c'): 
        netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        #netG_B = netG_B.module
        netG_B.load_state_dict(torch.load('checkpoints/2_cartoon2photo/latest_net_G_B.pth'))

        netG_B = netG_B.to('cpu')
        style= "Cartoon"
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    #cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()
