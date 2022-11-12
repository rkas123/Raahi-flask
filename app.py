from flask import Flask, request,render_template
from flask_restful import Api,Resource
import cv2
import numpy as np
from PIL import Image

CANNY = 'canny'
SEGMENTED = 'segmented'
HOUGH = 'hough'

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return 'Use POST verb to send an image'

class Raahi(Resource):
    def canny(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #convert the image to grayscale
        blur = cv2.GaussianBlur(gray,(11,11),0)       #apply gaussian blur to remove noise
        canny = cv2.Canny(blur,50,150)                #canny edge detector to find edges (minThreshold,maxThreshold)
        print('canny')
        return canny

    def segmentation(self,image):
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros_like(image)
        # poly = np.array([[(0,height),(width,height),(int(width/2),int(height/3))]])  #create a triangular polygon which will be segmented
        poly = np.array([[(0,height),(width,height),(int(width*6/10),int(height/3)),(int(width*4/10),int(height/3))]])  #create a trapezium polygon which will be segmented
        cv2.fillPoly(mask,poly,255)
        masked_image = cv2.bitwise_and(image,mask)
        lines = cv2.HoughLinesP(masked_image,2,np.pi/180, 100,np.array([]),minLineLength = 40,maxLineGap = 10)
        # parametric equation in hough space
        # 2nd and 3rd arguments define the size of blocks in hough space
        # A line in hough space is a point but a set of lines passing through a point in a sinosoidal curve
        # edges are the ones with large amount of contribution in the bins
        # threshold for a line(minimum number of points of the line present in the image)
        # min size of hough line to be considered as a hough line
        # max distance between two line segments to be different lines
        return lines, masked_image   

    def display_lines(self,image,lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image,(x1,y1),(x2,y2), (255,0,0),5)
        return line_image

    def get(self):
        print(request.form)
        return 'donzoes'
    
    def post(self,routine):
        print(routine)
        try:
            file = request.files['image']
            img = Image.open(file.stream)
            num = np.array(img)
        except: 
            return 'Failed to get image', 400
        
        canny_image = self.canny(num)
        if routine == CANNY:
            return canny_image.tolist(), 200
        houghLines, masked_image = self.segmentation(canny_image)

        if routine == SEGMENTED:
            return masked_image.tolist(), 200

        line_image = self.display_lines(num,houghLines)
        overlap_image = cv2.addWeighted(num, 0.8, line_image, 1,1)
        return overlap_image.tolist(), 200

api.add_resource(Raahi,"/raahi/<string:routine>")

if __name__ == "__main__":
    app.run(debug=False, host = '0.0.0.0')