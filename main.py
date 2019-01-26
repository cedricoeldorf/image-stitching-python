################
# main file
import cv2
from part1 import *
import matplotlib.pyplot as plt

#########################
## These parameters are relatively large, as the code reduces them to adjust to image
type = 'euc'
error_threshold = 0.0025
distance_thresh = 0.01
sample_size = 10

#######################
## READ IMAGES
## PLease insert path to left and right image in that order
LEFT_IMAGE = './b.jpg'
RIGHT_IMAGE = './a.jpg'


output_image = ImStitch(LEFT_IMAGE, RIGHT_IMAGE, type, error_threshold, distance_thresh, sample_size)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.ion()
plt.show()
