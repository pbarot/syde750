#importing libraries

import numpy as np
import dlib
import cv2
from scipy import ndimage    
from openni import openni2
from openni import _openni2 as c_api


#import depth stream......
openni2.initialize()
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

while True:
#get depth stream running.....
    depframe = depth_stream.read_frame()
    frame_data = depframe.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (1, 480, 640)
    img = img.astype(float)#/ 1024
    
    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    cv2.imshow("Depth Image", img)
    ans=np.max(img[319:320,239:240,:])
    print(ans)
    if cv2.waitKey(30) == 27:
        break
cv2.destroyAllWindows()
