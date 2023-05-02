"""
Construction of the training and validation databases

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from makedataset import *
import argparse

if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(description=\
								  "Building the training patch database")
	
	parser.add_argument("--rgb", action='store_true',default = True,\
						help='prepare RGB database instead of grayscale')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=128, \
					 help="Patch size")
	parser.add_argument("--stride", "--s", type=int, default=128, \
					 help="Size of stride")
	
	args = parser.parse_args()

	if args.rgb:
		Edge_TrainSynRGB('./dataset/High','./dataset/Low',args.patch_size,args.stride)
#Train_100