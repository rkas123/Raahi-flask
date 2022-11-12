import requests
from PIL import Image
from numpy import asarray
import numpy as np
import cv2
import matplotlib.pyplot as plt



def showImage(image,text):
  img = image.copy()
  img = cv2.putText(img, text=text, org=(10, 25), fontFace= 0, fontScale=1, color= 255,thickness=3)
  cv2.imshow(text,img)
  cv2.waitKey(0)


def canny(image):
  gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #convert the image to grayscale
  blur = cv2.GaussianBlur(gray,(11,11),0)       #apply gaussian blur to remove noise
  canny = cv2.Canny(blur,50,150)                #canny edge detector to find edges (minThreshold,maxThreshold)
  # plt.imshow(canny)
  showImage(canny,'Canny')
  return canny

def segmentation(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    # poly = np.array([[(0,height),(width,height),(int(width/2),int(height/3))]])  #create a triangular polygon which will be segmented
    poly = np.array([[(0,height),(width,height),(int(width*6/10),int(height/3)),(int(width*4/10),int(height/3))]])  #create a trapezium polygon which will be segmented
    cv2.fillPoly(mask,poly,255)
    showImage(mask,'Region of Attention')
    masked_image = cv2.bitwise_and(image,mask)
    showImage(masked_image,'Segmented Image')
    lines = cv2.HoughLinesP(masked_image,2,np.pi/180, 100,np.array([]),minLineLength = 40,maxLineGap = 10)
    # parametric equation in hough space
    # 2nd and 3rd arguments define the size of blocks in hough space
    # A line in hough space is a point but a set of lines passing through a point in a sinosoidal curve
    # edges are the ones with large amount of contribution in the bins
    # threshold for a line(minimum number of points of the line present in the image)
    # min size of hough line to be considered as a hough line
    # max distance between two line segments to be different lines
    return lines

def display_lines(image,lines):
  line_image = np.zeros_like(image)
  if lines is not None:
    for line in lines:
      x1,y1,x2,y2 = line.reshape(4)
      cv2.line(line_image,(x1,y1),(x2,y2), (255,0,0),5)

  return line_image

image = cv2.imread('jpgimage.jpg')
print(type(image))
canny_image = canny(image)
houghLines = segmentation(canny_image)
line_image = display_lines(image,houghLines)
overlap_image = cv2.addWeighted(image, 0.8, line_image, 1,1)
cv2.imshow('image',overlap_image)
cv2.waitKey(0)
showImage(overlap_image,'Final Output')
cv2.destroyAllWindows()