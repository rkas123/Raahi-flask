from flask import Flask, request,render_template, send_file
from flask_restful import Api,Resource
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import time
import torch

CANNY = 'canny'
SEGMENTED = 'segmented'
HOUGH = 'hough'
YOLO = 'yolo'
DEPTH = 'depth_estimation'

app = Flask(__name__)
api = Api(app)

#hyperparameters
confthres=0.5
nmsthres=0.1

# declaring globally
labelsPath=""
cfgpath=""
wpath=""
Lables=''
CFG=''
Weights=''
nets=''
Colors=''

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def get_predection(image,net,LABELS,COLORS):
    H = image.shape[0]
    W = image.shape[1]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO algo took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:

        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image

### Depth Estimation
#img is numpy array
def depthEstimator(img):
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    im = Image.fromarray(depth_map)
    im.save('depth_map.jpg')

    return depth_map


@app.route('/')
def index():
    return 'Use POST verb to send an image'

class Raahi(Resource):
    def canny(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #convert the image to grayscale
        blur = cv2.GaussianBlur(gray,(11,11),0)       #apply gaussian blur to remove noise
        canny = cv2.Canny(blur,50,150)                #canny edge detector to find edges (minThreshold,maxThreshold)
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
    
    def encode_image(self,image):
        cv2.imwrite('file.jpeg',image)
        img = Image.open('file.jpeg','r')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        my_encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
        return my_encoded_img

    def get(self):
        return 'donzoes'
    
    def post(self,routine):
        start = time.time() 
        try:
            file = request.files['image']
            image = Image.open(file)
            num = np.array(image)

        except: 
            return 'Failed to get image', 400
        
        if routine == DEPTH:
            res = depthEstimator(num)
            return self.encode_image(res), 200
        if routine == YOLO:
            res = get_predection(num,nets,Lables,Colors)
            # cv2.imshow("Image", res)
            # cv2.waitKey()
            end = time.time()
            print("[INFO] YOLO API took {:.6f} seconds".format(end - start))
            return self.encode_image(res), 200

        canny_image = self.canny(num)
        if routine == CANNY:
            end = time.time()
            print("[INFO] Canny API took {:.6f} seconds".format(end - start))
            return self.encode_image(canny_image), 200

        houghLines, masked_image = self.segmentation(canny_image)

        if routine == SEGMENTED:
            # cv2.imshow('resp',masked_image)
            # cv2.waitKey(0)
            end = time.time()
            print("[INFO] Segmented API took {:.6f} seconds".format(end - start))
            return self.encode_image(masked_image), 200

        line_image = self.display_lines(num,houghLines)
        overlap_image = cv2.addWeighted(num, 0.8, line_image, 1,1)
        # cv2.imshow('resp',overlap_image)
        # cv2.waitKey(0)
        end = time.time()
        print("[INFO] Hough API took {:.6f} seconds".format(end - start))
        return self.encode_image(overlap_image), 200

api.add_resource(Raahi,"/raahi/<string:routine>")

if __name__ == "__main__":
    #initializing once
    labelsPath="yolov3/coco.names"
    cfgpath="yolov3/yolo.cfg"
    wpath="yolov3/yolo.weights"
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)

    # Load a MiDas model for depth estimation
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # res=get_predection(image,nets,Lables,Colors)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    # cv2.imshow("Image", res)
    # cv2.waitKey()

    app.run(debug=True, host = '0.0.0.0')