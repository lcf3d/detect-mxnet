import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz
from gluoncv.data import VOCDetection
import cv2

classes = ['rbc', 'wbc', 'zfq','bz']

start = time.time() 


#load model
net=gluon.SymbolBlock.imports('./model2/model-symbol.json', ['data'], './model2/model-0000.params', ctx=mx.cpu())

#load image
im_name='C:/Users/DELL/Desktop/train_code/VOCtemplate/VOC2018/JPEGImages/272.jpg'

imgs = mx.img.imread(im_name)
          
x, image = gcv.data.transforms.presets.rcnn.transform_test(imgs, 600, 800)
print(x.shape)


#detect result
cid, score, bbox = net(x)
#net.export('model')####
time_s = "time %.2f sec" % (time.time() - start)
print(time_s)

ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes, thresh=0.5)
#cv2.imwrite('C:/Users/DELL/Desktop/rbc/valid_result.jpg', imgs)
plt.show()       
