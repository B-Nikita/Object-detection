# -*- coding: utf-8 -*-
"""detector.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TarseJzFlnn5hPVuk0aTMaXSK7g4vEeA
"""
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

from util import *

"""##### def arg_parse()"""

def arg_parse():
  '''
  Parse arguments to the detect module
  '''
  parser = argparse.ArgumentParser(description='YOLO v3 Detection module')
  #images : used to specify the input image or directory of images
  parser.add_argument('--images',dest='images',help='Image/ Directory containing images to perform detection upon',
                      default = 'imgs', type=str)
  
  #det : directory to save detections to
  parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
  parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
  parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
  parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
  
  #Alternative configuration file
  parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "yolov3.cfg", type = str)
  
  #weightfile
  parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
  
  #reso : Input image's resolution, can be used for speed - accuracy tradeoff
  parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
  return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

#Load the class file
num_classes = 80 #for COCO
classes = load_classes('data/coco.names')

#set up neural network
print('Loading network...')
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print('Network Successfully loaded')

model.net_info['height'] = args.reso
inp_dim = int(model.net_info['height'])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU available, Put a model on GPU
if CUDA:
  model.cuda()

#Set the model in evaluation mode
model.eval()

#Read the image from the disk, or froma directory
#The paths of the image/images are stored in a list called imlist
read_dir = time.time()    #It's checkpoint used to measure time
#Detection phase
try:
  imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
  imlist=[]
  imlist.append(osp.join(osp.realpath('.'),images))
except FileNotFoundError:
  print('No file or directory with the name {}'.format(images))
  exit()

#if the directory to save the detections, defined by det flag,doesn't exist, create it
if not os.path.exists(args.det):
  os.makedirs(args.det)

load_batch=time.time()
loaded_ims=[cv2.imread(x) for x in imlist]    #use OpenCV to load the images

#PyTorch variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

#List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

#Create the batches
leftover = 0
if (len(im_dim_list) % batch_size):
  leftover = 1

if batch_size != 1:
  num_batches = len(imlist) // batch_size + leftover            
  im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,len(im_batches))]))  for i in range(num_batches)]  

write = 0

if CUDA:
  im_dim_list = im_dim_list.cuda()

#The detection loop
'''
We iterate over the batches, generate the prediction, and concatenate
the prediction tensors (of shape, D x 8, the output of write_results fnction)
of all the image we have to perform detection upon.
'''

start_det_loop = time.time()
for i, batch in enumerate(im_batches):
  #Load the image
  start =time.time()
  if CUDA:
    batch= batch.cuda()
  with torch.no_grad():
    prediction = model(Variable(batch), CUDA)

  prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
  end = time.time()

  if type(prediction) == int:
    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
      im_id = i*batch_size + im_num
      print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
      print("{0:20s} {1:s}".format("Objects Detected:", ""))
      print("----------------------------------------------------------")
    continue

  prediction[:,0] += i*batch_size   #transform the atribute from index in batch to index in imlist
  if not write:   #If we haven't initialised output
    output = prediction
    write = 1

  else:
    output = torch.cat((output,prediction))

  for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
    im_id = i*batch_size + im_num
    objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
    print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
    print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
    print("----------------------------------------------------------")

  '''
  This torch.cuda.synchronize() line makes sure that CUDA kernel is synchronized with the CPU.Otherwise
  CUDA kernel returns the control to CPU as soon as the GPU job is queued and well
  before the GPU job is completed(Asynchronous calling).This might lead to a
  misleading time if end=time.time() gets printed before the GPU job is actually
  over
  '''
  if CUDA:
    torch.cuda.synchronize()       
try:
  output
except NameError:
  print ("No detections were made")
  exit()

'''
Before we draw the bounding boxes, the predictions contained in
our output tensor conform to the input size of the network, 
and not the original sizes of the images. So, before we can 
draw the bounding boxes, let us transform the corner attributes
of each bounding box, to the original dimensions of images.
'''
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor= torch.min(416/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

'''
Let us now clip any bounding boxes that may have boundaries outside 
the image to the edges of our image.
'''
for i in range(output.shape[0]):
  output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
  output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

output_recast = time.time()
class_load =time.time()
colors = pkl.load(open('pallete','rb'))

draw = time.time()

"""##### def write(x, results)

This function to draw the rectangle (bounding boxes) with a color of a random choice from colors.It also creates a filled rectangle on the top left corner of the bounding box, and writes the class of the object detected acroaa the filled rectangle.
"""

def write(x, results):
  c1 = tuple(x[1:3].int())
  c2 = tuple(x[3:5].int())
  img = results[int(x[0])]
  cls = int(x[-1])
  color = random.choice(colors)
  label = "{0}".format(classes[cls])
  cv2.rectangle(img, c1, c2,color, 1)
  t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
  c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
  cv2.rectangle(img, c1, c2,color, -1)
  cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
  return img

#Let's draw the bounding boxes on image

#modifies the images inside loaded_ims inplace. 
list(map(lambda x: write(x, loaded_ims), output))

'''
Each image is saved by prefixing the "det_" in front of the image name. We create a list of addresses,
to which we will save the our detection images to.
'''
det_names = pd.Series(imlist).apply(lambda x: "{}/det1_{}".format(args.det,x.split("/")[-1]))

#write the images with detections to the address in det_names.
list(map(cv2.imwrite, det_names, loaded_ims))
print('det_names: ',det_names)

'''
At the end of our detector we will print a summary containing which part of the code took how long to
execute. This is useful when we have to compare how different hyperparameters effect the speed of the
detector. Hyperparameters such as batch size, objectness confidence and NMS threshold, 
(passed with bs, confidence, nms_thresh flags respectively) can be set while executing the script 
detection.py on the command line.
'''

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()