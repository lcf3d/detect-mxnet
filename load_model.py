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

classes = ['rbc', 'wbc', 'zfq','hc']

start = time.time() 
#load model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('./model/ssd_512_mobilenet1.0_voc_0140_0.8212.params')   

#load image
im_name='./img/1825.jpg'

imgs = mx.img.imread(im_name)
          
x, image = gcv.data.transforms.presets.ssd.transform_test(imgs, 512)


net.hybridize()####

#detect result
cid, score, bbox = net(x)
net.export('./model/best_8212', epoch=0)####
time_s = "time %.2f sec" % (time.time() - start)
print(time_s)

ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes, thresh=0.5)
#cv2.imwrite('C:/Users/DELL/Desktop/rbc/valid_result.jpg', imgs)
plt.show()       




