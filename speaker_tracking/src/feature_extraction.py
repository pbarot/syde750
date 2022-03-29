import numpy as np
import argparse
import imutils
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.path import Path
import dlib
from dlib import *
import cv2


def normalize(signal):
    return  (signal-np.min(signal))/(np.max(signal)-np.min(signal))


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

        pix = int(pix_energy > base_vals[0]*1.2)
        height = int(height_energy > base_vals[1]*1.1)
        light = int(light_energy < base_vals[2]*1.1)
        depth = int((depth_energy/base_vals[3]) > 1.04)

        return (np.bitwise_and(np.bitwise_and(pix, height), depth))
        #return pix, height, light, depth# depth_energy/base_vals[3]
        #return np.bitwise_and([height, light, depth], [1,1,1])


def process_npix(total_num_pix, base_e,flag):
    if(flag ==0):
        base_e = np.sqrt(np.mean(np.array(total_num_pix)**2))
        return base_e
    
    elif(flag ==1):
        curr_energy = np.sqrt(np.mean(np.array(total_num_pix)**2))
        return int(curr_energy > base_e*1.3), curr_energy

def process_heights(total_heights, base_h,flag):
    if(flag ==0):
        base_h = np.sqrt(np.mean(np.array(total_heights)**2))
        return base_h
    
    elif(flag ==1):
        curr_energy = np.sqrt(np.mean(np.array(total_heights)**2))
        return int(curr_energy > base_h*1.3), curr_energy

def process_lightness(total_lightness, base_l,flag):
    if(flag ==0):
        base_l = np.sqrt(np.mean(np.array(total_lightness)**2))
        return base_l
    
    elif(flag ==1):
        curr_energy = np.sqrt(np.mean(np.array(total_lightness)**2))
        return int(curr_energy < base_l*1.1), curr_energy

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
	idx = [5,6,7,8,9,10,11,55,56,57,58,59]

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


def get_yaw_pitch(shape, frame):
	idx = 30
	h = frame.shape[0]//2
	w = frame.shape[1]//2

	displacement = (w-shape[idx][0], h-shape[idx][1])

	yaw = displacement[0]*80/640
	pitch = displacement[1]*50/480



	return yaw, pitch