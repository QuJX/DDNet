# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



class Main(nn.Module):
	def __init__(self):
		super(Main,self).__init__()

		self.left =  LeftED(4,32)
		self.right = RightED(3,32)
         
	def forward(self,x,xgl):
        
		x_fout, x_eout, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3 = self.left(x,xgl)
		x_out = self.right(x, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3)

		return x_fout, x_eout, x_out

class LeftED(nn.Module):
	def __init__(self,inchannel,channel):
		super(LeftED,self).__init__()

		self.e1 = ResUnit(channel)
		self.e2 = ResUnit(channel*2)
		self.e3 = ResUnit(channel*4)



		self.ed1 = ResUnit(channel*2)
		self.ed2 = ResUnit(channel*1)
		self.ed3 = ResUnit(int(channel*0.5))

		self.fd1 = ResUnit(channel*2)
		self.fd2 = ResUnit(channel*1)
		self.fd3 = ResUnit(int(channel*0.5))

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

		self.conv_in = nn.Conv2d(inchannel,channel,kernel_size=3,stride=1,padding=1,bias=False)    
        
		self.conv_eout = nn.Conv2d(int(0.5*channel),3,kernel_size=1,stride=1,padding=0,bias=False)                
		self.conv_fout = nn.Conv2d(int(0.5*channel),1,kernel_size=1,stride=1,padding=0,bias=False)          

		self.conv_e1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_e2te3 = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False) 

		self.conv_e1_a = nn.Conv2d(channel,int(0.5*channel),kernel_size=1,stride=1,padding=0,bias=False)  
		self.conv_e2_a = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)          
		self.conv_e3_a = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)  

        
		self.conv_fd1td2 = nn.Conv2d(2*channel,1*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_fd2td3 = nn.Conv2d(channel,int(0.5*channel),kernel_size=1,stride=1,padding=0,bias=False)  


         
		self.conv_ed1td2 = nn.Conv2d(2*channel,1*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_ed2td3 = nn.Conv2d(channel,int(0.5*channel),kernel_size=1,stride=1,padding=0,bias=False) 

        
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')
    
	def forward(self,x,xgl):

		x_in = self.conv_in(torch.cat((x,xgl),1))

		e1 = self.e1(x_in)    
		e2 = self.e2(self.conv_e1te2(self.maxpool(e1)))    
		e3 = self.e3(self.conv_e2te3(self.maxpool(e2)))

		e1_a = self.conv_e1_a(e1)
		e2_a = self.conv_e2_a(e2)
		e3_a = self.conv_e3_a(e3)

        
		fd1 = self.fd1(e3_a)
		fd2 = self.fd2(self.conv_fd1td2(self._upsample(fd1,e2)) + e2_a)   
		fd3 = self.fd3(self.conv_fd2td3(self._upsample(fd2,e1)) + e1_a)  


		ed1 = self.ed1(e3_a + fd1)     
		ed2 = self.ed2(self.conv_ed1td2(self._upsample(ed1,e2)) + fd2 + e2_a)   
		ed3 = self.ed3(self.conv_ed2td3(self._upsample(ed2,e1)) + fd3 + e1_a)  


		x_fout = self.conv_fout(fd3) 
		x_eout = self.conv_eout(ed3)
        
		return x_fout, x_eout, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3

  
    
class RightED(nn.Module):
	def __init__(self,inchannel,channel):
		super(RightED,self).__init__()
        
		self.ee1 = ResUnit(int(0.5*channel))
		self.ee2 = ResUnit(channel)
		self.ee3 = ResUnit(channel*2)

		self.fe1 = ResUnit(int(0.5*channel))
		self.fe2 = ResUnit(channel)
		self.fe3 = ResUnit(channel*2)   

		self.d1 = ResUnit(channel*4)
		self.d2 = ResUnit(channel*2)
		self.d3 = ResUnit(channel)

		self.d4 = ResUnit(channel)
        
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)  
            

		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)  

		self.conv_fe0te1 = nn.Conv2d(int(0.5*channel),channel,kernel_size=1,stride=1,padding=0,bias=False)         
		self.conv_fe1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   

		self.conv_ee0te1 = nn.Conv2d(int(0.5*channel),channel,kernel_size=1,stride=1,padding=0,bias=False)         
		self.conv_ee1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)          

		self.conv_e0te1 = nn.Conv2d(int(1*channel),channel,kernel_size=3,stride=1,padding=1,bias=False) 
		self.conv_e1te2 = nn.Conv2d(int(2*channel),2*channel,kernel_size=3,stride=1,padding=1,bias=False)         
		self.conv_e2te3 = nn.Conv2d(int(4*channel),4*channel,kernel_size=3,stride=1,padding=1,bias=False) 
        
		self.conv_d1td2 = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_d2td3 = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  


		self.act1 = nn.PReLU(channel)
		self.norm1 = nn.GroupNorm(num_channels=channel,num_groups=1)

		self.act2 = nn.PReLU(channel*2)
		self.norm2 = nn.GroupNorm(num_channels=channel*2,num_groups=1)
        
		self.act3 = nn.PReLU(channel*4)
		self.norm3 = nn.GroupNorm(num_channels=channel*4,num_groups=1)
        
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')
    
	def forward(self,x, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3):


		fe1 = self.fe1(fd3)
		fe2 = self.fe2(self.conv_fe0te1(self.maxpool(fe1)) + fd2)
		fe3 = self.fe3(self.conv_fe1te2(self.maxpool(fe2)) + fd1)

		ee1 = self.ee1(ed3 + fe1)
		ee2 = self.ee2(self.conv_ee0te1(self.maxpool(ee1)) + fe2 + ed2)
		ee3 = self.ee3(self.conv_ee1te2(self.maxpool(ee2)) + fe3 + ed1)

		fde1 = self.act1(self.norm1(self.conv_e0te1(torch.cat((ee1 , fe1),1)))) 
		fde2 = self.act2(self.norm2(self.conv_e1te2(torch.cat((ee2 , fe2),1)))) 
		fde3 = self.act3(self.norm3(self.conv_e2te3(torch.cat((ee3 , fe3),1))))  

		d1 = self.d1(fde3 + e3)
		d2 = self.d2(self.conv_d1td2(self._upsample(d1,e2)) + fde2 + e2)
		d3 = self.d3(self.conv_d2td3(self._upsample(d2,e1)) + fde1 + e1)

        
		x_out = self.conv_out(self.d4(d3))

		return x_out + x
    

    
class ResUnit1(nn.Module):    # Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(ResUnit1,self).__init__()

		self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1(x)))
		x_2 = self.act(self.norm(self.conv_2(x_1)))
		x_3 = self.act(self.norm(self.conv_3(x_2))+x_1)

		return	x_3
    

# =============================================================================
# class ResUnit1(nn.Module):
# 	def __init__(self,channel):                                
# 		super(ResUnit1,self).__init__()
# 
# 		self.conv_cam_1 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
# 		self.conv_sam_1 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
# 		self.conv_scl_1 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
#         
# 		self.conv_cam_m3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
# 		self.conv_sam_m3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
# 
# 		self.conv_cam_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
# 		self.conv_sam_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
# 		self.conv_scl_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
#         
# 
# 		self.conv_11 = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
# 		self.conv_12 = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
# 
# 		self.act = nn.PReLU(channel)
#         
# 		self.scl = StandardConvolutionalLayers(channel)
# 		self.cam = ChannelAttention(channel)
# 		self.sam = SpatialAttention() 
# # =============================================================================
# # 		self.cam = ChannelAttentionModule(channel)
# # 		self.sam = SpatialAttentionModule(channel) 
# # =============================================================================
# 	def forward(self,x):
#         
# 		x_cam_1 = self.conv_cam_1(x)
# 		x_sam_1 = self.conv_sam_1(x)
# 		x_scl_1 = self.conv_scl_1(x)        
# 
# 		x_cam_2 = self.conv_cam_m3(x_cam_1) + self.cam(x_cam_1)
# 		x_sam_2 = self.conv_sam_m3(x_sam_1) + self.sam(x_sam_1)
# 		x_scl_2 = self.scl(x_scl_1)
#         
# 		x_cam_3 = self.conv_cam_3(x_cam_2)
# 		x_sam_3 = self.conv_sam_3(x_sam_2)
# 		x_scl_3 = self.conv_scl_3(x_scl_2) 
#         
# 		x_1 = self.conv_11(torch.cat((x_cam_3,x_sam_3),1))
# 		x_2 = self.conv_12(torch.cat((x_1,x_scl_3),1))
# 
# 		x_out = self.act(x_2+x)     
#         
# 		return	x_out
#     
# =============================================================================

class ResUnit(nn.Module):
	def __init__(self,channel):                                
		super(ResUnit,self).__init__()

		self.conv_cam_1 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_sam_1 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_scl_1 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
		self.conv_cam_m3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_sam_m3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_cam_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_sam_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_scl_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        

		self.conv_11 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_12 = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_13 = nn.Conv2d(channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
		self.act = nn.PReLU(channel)
        
		self.scl = StandardConvolutionalLayers(channel)
		self.cam = ChannelAttention(channel)
		self.sam = SpatialAttention() 
		#self.cam = ChannelAttentionModule(channel)
		#self.sam = ChannelAttentionModule(channel)#SpatialAttentionModule(channel) 
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
         
         
	def forward(self,x):
        
		#x_cam_1 = self.conv_cam_1(x)
		x_sam_1 = self.conv_sam_1(x)
		x_scl_1 = self.conv_scl_1(x)        

		#x_cam_2 = self.conv_cam_m3(x_cam_1) * self.cam(x_cam_1)
		x_sam_2 = self.conv_sam_m3(x_sam_1) * self.sam(x_sam_1) 
		x_scl_2 = self.scl(x_scl_1)
        
		#x_cam_3 = self.conv_cam_3(x_cam_2)
		x_sam_3 = self.scl(x_sam_2)
		#x_sam_3 = self.conv_sam_3(x_sam_2)
		#x_scl_3 = self.conv_scl_3(x_scl_2)
		x_scl_3 = self.scl(x_scl_2)
        
		#x_1 = self.conv_11(x_cam_3+x_sam_3)
		x_2 = self.conv_12(torch.cat((x_sam_3,x_scl_3),1))

		x_out = self.conv_13(x_2) + x
        
		return	x_out
    
class StandardConvolutionalLayers(nn.Module):    # StandardConvolutional
	def __init__(self,channel):                                
		super(StandardConvolutionalLayers,self).__init__()

		self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1(x)))
		#x_2 = self.act(self.norm(self.conv_2(x_1)))

		return	x_1


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 3), padding=(0, 1))
        self.key = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out
        
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        #平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        #MLP  除以16是降维系数
        self.fc1   = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #结果相加
        out = avg_out + max_out
        return self.sigmoid(out)

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        #声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  #平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) #最大池化
        #拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) #7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)