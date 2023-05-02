# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:00:46 2020

@author: Administrator
"""

import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata

class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset, self).__init__()
		self.trainrgb = trainrgb
		self.trainsyn = trainsyn
		self.train_haze	 = 'train_ImageEdge.h5'
		
		if self.trainrgb:
			if self.trainsyn:
				h5f = h5py.File(self.train_haze, 'r')
			else:
				h5f = h5py.File(self.train_real_rgb, 'r')				 
		else:
			if self.trainsyn:				 
				h5f = h5py.File(self.train_syn_gray, 'r')
			else:
				h5f = h5py.File(self.train_real_gray, 'r')			  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		if self.trainrgb:
			if self.trainsyn:
				h5f = h5py.File(self.train_haze, 'r')
			else:
				h5f = h5py.File(self.train_real_rgb, 'r')				 
		else:
			if self.trainsyn:				 
				h5f = h5py.File(self.train_syn_gray, 'r')
			else:
				h5f = h5py.File(self.train_real_gray, 'r')			  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)


def data_augmentation(image, mode):
	r"""Performs dat augmentation of the input image

	Args:
		image: a cv2 (OpenCV) image
		mode: int. Choice of transformation to apply to the image
			0 - no transformation
			1 - flip up and down
			2 - rotate counterwise 90 degree
			3 - rotate 90 degree and flip up and down
			4 - rotate 180 degree
			5 - rotate 180 degree and flip
			6 - rotate 270 degree
			7 - rotate 270 degree and flip
	"""
	out = np.transpose(image, (1, 2, 0))
	if mode == 0:
		# original
		out = out
	elif mode == 1:
		# flip up and down
		out = np.flipud(out)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(out)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(out)
		out = np.flipud(out)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(out, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(out, k=2)
		out = np.flipud(out)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(out, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(out, k=3)
		out = np.flipud(out)
	else:
		raise Exception('Invalid choice of image transformation')
	return np.transpose(out, (2, 0, 1))

def img_to_patches(img,win,stride,Syn=True):
	
	chl,raw,col = img.shape
	chl = int(chl)
	num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
	num_col = np.ceil((col-win)/stride+1).astype(np.uint8) 
	count = 0
	total_process = int(num_col)*int(num_raw)
	img_patches = np.zeros([chl,win,win,total_process])
	if Syn:
		for i in range(num_raw):
			for j in range(num_col):			   
				if stride * i + win <= raw and stride * j + win <=col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, stride*j : stride*j + win]				 
				elif stride * i + win > raw and stride * j + win<=col:
					img_patches[:,:,:,count] = img[:,raw-win : raw,stride * j : stride * j + win]		   
				elif stride * i + win <= raw and stride*j + win>col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, col-win : col]
				else:
					img_patches[:,:,:,count] = img[:,raw-win : raw,col-win : col]				
				count +=1		   
		
	return img_patches


def readfiles(filepath):
	'''Get dataset images names'''
	files = os.listdir(filepath)
	return files

def normalize(data):

	return np.float32(data/255.0)

def samesize(img,size):
	
	img = cv2.resize(img,size)
	return img

def concatenate2imgs(img,depth):	
	c,w,h = img.shape
	conimg = np.zeros((c+c,w,h))
	conimg[0:c,:,:] = img
	conimg[c:2*c,:,:] = depth
	
	return conimg

def Edge_TrainSynRGB(img_filepath, depth_filepath, patch_size, stride):
	'''synthetic ImageEdge images'''
	train_haze = 'train_ImageEdge.h5'
	img_files = readfiles(img_filepath)
	count = 0
	scales = [1.0]#[0.6,0.8,1.0]	  
		
	with h5py.File(train_haze, 'w') as h5f:
		for i in range(len(img_files)):
			filename = img_files[i][:-4]
			oimg = cv2.imread(img_filepath + '/' + filename + '.png')

			odepth = cv2.imread(depth_filepath + '/' + filename + '.png')


			for sca in scales:
				#img = cv2.resize(oimg, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)
				#depth = cv2.resize(odepth, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)

				img = oimg.transpose(2, 0, 1)
				depth = odepth.transpose(2, 0, 1)
				#depth = depth.transpose((1,0))
                
				print(img.shape,depth.shape)
              
				img = normalize(img)
				depth = normalize(depth)               
				img_depth = concatenate2imgs(img,depth)
				img_patches = img_to_patches(img_depth, win=patch_size, stride=stride)
				print("\tfile: %s scale %.1f # samples: %d" %(img_files[i], sca,img_patches.shape[3]))	  
				for nx in range(img_patches.shape[3]):
					data = data_augmentation(img_patches[:, :, :, nx].copy(), np.random.randint(0, 7))
					h5f.create_dataset(str(count), data=data)
					count += 1
			i += 1
		print(data.shape)
	h5f.close()
	

def TrainSynRGB(img_filepath, patch_size, stride):
	'''synthetic Haze images'''
	train_haze = 'train_haze.h5'
	img_files = readfiles(img_filepath)
	count = 0
	scales = [0.6,0.8,1.0]	  
		
	with h5py.File(train_haze, 'w') as h5f:
		for i in range(len(img_files)):
			filename = img_files[i]
			o_img = cv2.imread(img_filepath + '/' + filename)
			o_img = cv2.resize(o_img,(360,360))

			#img= samesize(img,(360,360))
			for sca in scales:
				img = o_img

				img = cv2.resize(o_img, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)
				img = img.transpose(2, 0, 1)

				img = normalize(img)
				img_patches = img_to_patches(img, win=patch_size, stride=stride)
				print("\tfile: %s scale %.1f # samples: %d" %(img_files[i], sca,img_patches.shape[3]))	  
				for nx in range(img_patches.shape[3]):
					data = data_augmentation(img_patches[:, :, :, nx].copy(), np.random.randint(0, 7))
					h5f.create_dataset(str(count), data=data)
					count += 1
			i += 1
		print(data.shape)
	h5f.close()


def TrainSynRGB_NA(img_filepath, patch_size, stride):
	'''synthetic Haze images'''
	train_haze = 'train_haze.h5'
	img_files = readfiles(img_filepath)
	count = 0
	scales = [1.0]#[0.6,0.8,1.0]	  
		
	with h5py.File(train_haze, 'w') as h5f:
		for i in range(len(img_files)):
			filename = img_files[i]
			oooimg = cv2.imread(img_filepath + '/' + filename)
			img = cv2.resize(oooimg,(256,256))

			for sca in scales:
				img = cv2.resize(img, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)
				img = img.transpose(2, 0, 1)
				img = normalize(img) 
				print("\tfile: %s scale %.1f" %(img_files[i], sca))	  
				data = data_augmentation(img.copy(), np.random.randint(0, 7))
				h5f.create_dataset(str(count), data=data)
				count += 1
			i += 1
		print(data.shape)
	h5f.close()	
