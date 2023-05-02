# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:37:45 2020

@author: Administrator
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

def Laplacian(x):
    weight=torch.tensor([
    [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
    [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
    [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
    ]).cuda()

    
    frame= nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, dilation=1, groups=1)
        
    return frame
		
    
def edge(x, imitation):
        
    def inference_mse_loss(frame_hr, frame_sr):
        content_base_loss = torch.mean(torch.sqrt((frame_hr - frame_sr) ** 2+(1e-3)**2))
        return torch.mean(content_base_loss)

    x_edge = Laplacian(x)
    imitation_edge = Laplacian(imitation)
    edge_loss = inference_mse_loss(x_edge, imitation_edge)

    return  edge_loss

class edgeloss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image):
        
        loss = edge(out_image,gt_image)
        
        return loss

