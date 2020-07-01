# YOLO(v3) object detector in Pytorch

The aim behind implementing the algorithm from scratch is understanding each and every key aspect about
how computations take place for machine to detect the object.

PyTorch has been used to implement an object detector based on YOLO v3, 
one of the faster object detection algorithms.

-------------------------------------------------------------------------------
**YOLO** stands for **You Only Look Once**

It's an object detector that uses features learned by a deep convolutional neural network to detect an object.
YOLO can only detect objects belonging to the classes present in the dataset used to train the network. 
We will be using the official weight file for our detector. These weights have been obtained by training the 
network on COCO dataset, and therefore we can detect 80 object categories.

### darknet.py
==============
Darknet is the name of the underlying architecture of YOLO. This file will contain the code that creates 
the YOLO network. We will supplement it with a file called `util.py` which will contain the code for various 
helper functions. Save both of these files in your detector folder.

Parsing the config file, and store every block as dictionary,
Creating the building blocks for neural network layers
Implementation of class Darknet includng the forward pass of the network

### yolov3.cfg
==============
Configuration file describes the layout of the network, block by block.

### util.py
===============
Consisting helper function like transform the output of the network into detection predictions,writting results,
Confidence thresholding,performing non-maximum supression(NMS),calculation IoU(Intersection over Union),Writting
 the prediction back to disk
 
### detector.py
====================
In this file, the input and the output pipelines of detector is 
specified. This involves the reading images off the disk, making a prediction, using the prediction to 
draw bounding boxes on images, and then saving them to the disk.
some command line flags to allow some experimentation with various hyperparamters of the network are introduced.


Thanks to Ayoosh Kathuriya for awesome blog.Checkout his blog on paperspace.

Link : https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

You can reach out to him on https://github.com/ayooshkathuria
