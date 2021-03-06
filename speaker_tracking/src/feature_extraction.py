import numpy as np
import argparse
import imutils
import pandas as pd 
#import matplotlib.pyplot as plt
from matplotlib.path import Path
import dlib
import cv2
from dlib import *
import cv2
import math


def normalize(img):
    return  cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def process_features(features, base_vals,flag):

    if(flag ==0):
        bv0 = np.sqrt(np.mean(np.array(features[0,:])**2))
        bv1 = np.sqrt(np.mean(np.array(features[1,:])**2))
        bv2 = np.sqrt(np.mean(np.array(features[2,:])**2))
        bv3 = np.sqrt(np.mean(np.array(features[3,:])**2))

        return [bv0, bv1, bv2, bv3]
    
    elif(flag ==1):
        pix_energy = np.sqrt(np.mean(np.array(features[0,:])**2))
        height_energy = np.sqrt(np.mean(np.array(features[1,:])**2))
        light_energy = np.sqrt(np.mean(np.array(features[2,:])**2))
        depth_energy = np.sqrt(np.mean(np.array(features[3,:])**2))

        pix = int(pix_energy > 	base_vals[0]*1.1)
        height = int(height_energy > base_vals[1]*1.1)
        light = int(light_energy < base_vals[2]*1.2)
        depth = int((depth_energy/base_vals[3]) > 1.002)

        return (np.bitwise_and(np.bitwise_and(np.bitwise_and(pix, height), depth), lightness))
    
# def process_npix(total_num_pix, base_e,flag):
#     if(flag ==0):
#         base_e = np.sqrt(np.mean(np.array(total_num_pix)**2))
#         return base_e
    
#     elif(flag ==1):
#         curr_energy = np.sqrt(np.mean(np.array(total_num_pix)**2))
#         return int(curr_energy > base_e*1.3), curr_energy

# def process_heights(total_heights, base_h,flag):
#     if(flag ==0):
#         base_h = np.sqrt(np.mean(np.array(total_heights)**2))
#         return base_h
    
#     elif(flag ==1):
#         curr_energy = np.sqrt(np.mean(np.array(total_heights)**2))
#         return int(curr_energy > base_h*1.3), curr_energy

# def process_lightness(total_lightness, base_l,flag):
#     if(flag ==0):
#         base_l = np.sqrt(np.mean(np.array(total_lightness)**2))
#         return base_l
    
#     elif(flag ==1):
#         curr_energy = np.sqrt(np.mean(np.array(total_lightness)**2))
#         return int(curr_energy < base_l*1.1), curr_energy

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def mouth(shape, frame):
	idx = [60, 61, 62, 63, 64, 65, 66, 67]
	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	nx = frame.shape[0]
	ny = frame.shape[1]

	x, y = np.meshgrid(np.arange(ny), np.arange(nx))
	x, y = x.flatten(), y.flatten()

	points = np.vstack((x,y)).T

	path = Path(verts)
	grid = path.contains_points(points)
	grid = grid.reshape((nx,ny))

	return grid

def lower_face(shape, frame):
	idx = [2,3,4,5,6,7,8,9,10,11,12,13,14,64, 55,56,57,58,59, 48]

	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	nx = frame.shape[0]
	ny = frame.shape[1]

	x, y = np.meshgrid(np.arange(ny), np.arange(nx))
	x, y = x.flatten(), y.flatten()

	points = np.vstack((x,y)).T

	path = Path(verts)
	grid = path.contains_points(points)
	grid = grid.reshape((nx,ny))

	return grid

def mouth_height(shape, frame):
	idx = [62, 66, 61, 67, 63, 65]
	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	h1 = verts[1][1] - verts[0][1]
	h2 = verts[3][1] - verts[2][1]
	h3 = verts[5][1] - verts[4][1]


	return np.mean([h1, h2, h3])

def full_height(shape, frame):
	idx = [8, 27]
	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	h1 = verts[1][1] - verts[0][1]

	return h1


def mouth_width(shape, frame):
	idx = [60, 64]
	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	width = verts[1][0] - verts[0][0]

	return abs(width)


def get_yaw_pitch(shape, frame, scale):
	idx = list(range(0,16))
	idx.extend([17,18,19,20,21,22,23,24,25,26])

	h = frame.shape[0]//2
	w = frame.shape[1]//2

	x,y = (np.mean(shape[idx], axis = 0))
	distance = frame[int(y),int(x)]/scale

	displacement = (w-x, h-y)

	yaw = np.arctan(displacement[0]/distance)
	pitch = np.arctan(displacement[1]/distance)

	return math.degrees(yaw), math.degrees(pitch)
