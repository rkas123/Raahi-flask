import requests
from PIL import Image
from numpy import asarray
import numpy as np
import cv2
import matplotlib.pyplot as plt

BASE = "http://192.168.43.35:5000/"
CANNY = 'canny'
SEGMENTED = 'segmented'
HOUGH = 'hough'

im = open('jpgimage.jpg','rb').read()
print(type(im))

# http://192.168.1.5:5000/casnny

response = requests.post(BASE + '/raahi/' + CANNY, files = {'image' : im})
# print("STATUS CODE")
# print(response.status_code)
res = response.json()
print(type(res))
num = np.array(res, dtype= np.uint8)
print(num.shape)
print(type(num))
cv2.imshow('text',num)
cv2.waitKey(0)
