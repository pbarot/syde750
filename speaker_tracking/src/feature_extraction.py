import numpy as np
import argparse
import imutils
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.path import Path
import dlib
from dlib import *
import cv2

def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

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

def mouth_height(shape, frame):
	idx = [51, 57]
	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	height = verts[1][1] - verts[0][1]

	return height


def mouth_width(shape, frame):
	idx = [60, 64]
	verts =shape[idx]
	verts = [(x,y) for x,y in verts]

	width = verts[1][0] - verts[0][0]

	return abs(width)