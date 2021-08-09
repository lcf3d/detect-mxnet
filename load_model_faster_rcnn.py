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
#net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_custom', classes=classes, pretrained_base=False)
net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc',  pretrained=True)
net.reset_class(classes)
net.load_parameters('./faster_rcnn_resnet50_v1b_voc.params')#, ignore_extra=True)   

#load image
#im_name='C:/Users/DELL/Desktop/train_code/VOCtemplate/VOC2018/JPEGImages/123.jpg'
im_name = 'C:\\Users\\DELL\\Desktop\\data-new\\image-lc\\motic\\5000.jpg'

imgs = mx.img.imread(im_name)
          
x, image = gcv.data.transforms.presets.rcnn.transform_test(imgs, 600, 800)
print(x.shape)


net.hybridize()####

#detect result
cid, score, bbox = net(x)
#net.export('model')####

for k_id in range(0, 1200):
    if(cid[0][k_id] == -1):
        break

print(k_id)

for k_id in range(0, 1200):
    if(score[0][k_id] < 0.5):
        break

print(k_id)

#result exclude   
detect_result = nd.concat(cid, score, bbox*2, dim = 2).reshape((k_id,6)).asnumpy()
invalid = []
for i in range(0, k_id):
    if (detect_result[i][1] < 0.5):
        invalid.append(i)       
valid_result = np.delete(detect_result[0:k_id], invalid, axis=0)
txt ="./" + 'mask.txt' 
np.savetxt(txt, valid_result, fmt="%.2f", delimiter=' ') #float
time_s = "time %.2f sec" % (time.time() - start)
print(time_s)

#ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=net.classes, thresh=0.5)
#cv2.imwrite('C:/Users/DELL/Desktop/rbc/valid_result.jpg', imgs)
#plt.show()       




