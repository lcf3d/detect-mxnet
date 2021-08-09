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
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='object_detect.')
    parser.add_argument("--image_path", type=str, default='./img', help="image path")
    parser.add_argument("--model_path", type=str, default='./model', help="model path")
    return parser.parse_args()

#加载检测模型
def load_model(model_path):
    json_path = model_path + "/" + "best_8212-symbol.json"
    params_path =model_path + "/" + "best_8212-0000.params"
    net = gluon.nn.SymbolBlock(outputs=mx.sym.load(json_path), inputs=mx.sym.var('data'))
    net.load_parameters(params_path)
    return net

#保存待检测图片绝对路径
def load_image(image_path):
    list_files = sorted(os.listdir(image_path))
    image_files = []
    for img in list_files:
        if img.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
            image_files.append(image_path + "/" + img)
    return image_files


def detect(image_file, net):
    imgs = mx.img.imread(image_file)
    #transforms
    x, image = gcv.data.transforms.presets.ssd.transform_test(imgs, 512)
    #detect result
    cid, score, bbox = net(x)
    bbox = bbox/1.067
	#result exclude   
    detect_result = nd.concat(cid, score, bbox, dim = 2).reshape((100,6)).asnumpy()
    invalid = []
    for i in range(0, 100):
        if (detect_result[i][0] == -1) or (detect_result[i][1] < 0.5):
            invalid.append(i)       
    valid_result = np.delete(detect_result, invalid, axis=0)
    return valid_result

#对图片进行目标检测
def object_detect(image_files, net):
    colors = [(0,0,255), (255,255,255), (0,255,0), (0,255,255)]
    stat_result = np.zeros(shape=(1,4))
    for image_file in image_files:
        valid_result = detect(image_file, net)

        if(valid_result.shape[0]):
            for index in range(0, valid_result.shape[0]):
                if(valid_result[index][0] == 0):
                    stat_result[0][0] = stat_result[0][0] + 1
                    #print(stat_result[0])
                elif(valid_result[index][0] == 1):
                    stat_result[0][1] = stat_result[0][1] + 1
                elif(valid_result[index][0] == 2):
                    stat_result[0][2] = stat_result[0][2] + 1
                elif(valid_result[index][0] == 3):
                    stat_result[0][3] = stat_result[0][3] + 1

	    #opencv save img
        img_cv = cv2.imread(image_file)
        if(valid_result.shape[0]):
            for i in range(0, len(valid_result)):
                cv2.rectangle(img_cv, (valid_result[i][2], valid_result[i][3]), (valid_result[i][4], valid_result[i][5]), colors[int(valid_result[i][0])], 2)
       
        img_cv_file = '.' + image_file.split(".")[1]  + '_D' + '.png'
        cv2.imwrite(img_cv_file, img_cv)
    return stat_result

def ObjectDetect(model_path, image_path):
    net = load_model(model_path)
   
    image_files = load_image(image_path)
    stat_result = object_detect(image_files, net)
    #save txt
    if(stat_result.shape[0]):
        txt = image_path + "/" + 'mask.txt' 
        np.savetxt(txt, stat_result, fmt="%.2f", delimiter=',') #float
    return stat_result


if __name__ == "__main__":
    args = parse_args()

    ObjectDetect(args.model_path, args.image_path)
    
    print("all detect!")
