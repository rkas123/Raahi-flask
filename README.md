# Raahi - Assistive tool for visually impaired

Raahi uses Machine learning techinques like -

- Object Localization
- Depth Estimation
- Lane Detection
- Natural Language Processing

to assist visually impaired people in navigation.

We made this as our Bachelors of Technology Project(BTP).

### Made By

- Aditya Kumar
- Harmeet Singh
- Rupesh Kumar
- Sumit Yadav

## Setup

Follow the steps to setup the backend of Raahi.

1. Inside yolov3 folder, add weight for you YOLO model. You may use the pretrained YOLO model or your own custom model as well.
2. If you have a custom model, you can also change the coco.names file to the classes of your custom model.
3. Go to terminal and type the command

   `python app.py`

   This will start the Flask server.

4. Note down the IP address of the backend.
5. When you open your [Raahi Flutter application](https://github.com/rkas123/BTP-final), enter the IP address of the backend.

You are good to go now!

### NOTE

- Make sure the device running the backend and the android device running Raahi are both connected to the same Wifi.
- This is needed because we in the prototype phase, we didn't buy a hosting service to host the backend Flask server.
