import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import scipy.misc
from model_0 import *
from makedataset import Dataset
import utils_train
from Test_SSIM import *
from EdgeLoss import edgeloss
from TVLoss import tvloss

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def GFLap(data):
    x = cv2.GaussianBlur(data, (3,3),0)
    x = cv2.Laplacian(np.clip(x*255,0,255).astype('uint8'),cv2.CV_8U,ksize =3)
    Lap = cv2.convertScaleAbs(x)
    return Lap/255.0


img = cv2.imread('2015_00719.jpg')/255.0
cv2.imwrite('output2.jpg',GFLap(img)*255.0)

