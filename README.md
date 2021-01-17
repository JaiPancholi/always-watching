<!-- # TODO IMPROVE CONFIDENCE OF FACE EXTRACTION
# Remove so many start_camera functions
# MAKE INTO ARG PARSE TYPE FORMAT
    TO TRAIN RECOGNISER
    TO TUNE PID
    TO START CAMERA
    TO RECORD VIDEOS OF PROCESS
# TUNE PID BETTER

# Write Up
## Building The Pi
## Setting up the Pi
    - remote networking
    - git / rsync
    - testing the hardware
## Detecting Faces
    - Open CV vs. Deep Learning
    - confidence levels with open cv
    - Tuning the PIDs
    - Multi processing
## Recognising Faces
    - Open CV vs. (face_recognition + SVM) vs. TensorFlow vs. Deep Learning
    - On-the-fly face enrollment
    - draw.io process flow
    - Showing a specific light to certain faces -->


# pi_face_tracker
This is repository contains the software for a Raspberry Pi face tracking system. This system uses a Raspberry Pi as a mobile computer, Raspberry Pi Camera as an image capture method, a Pan Tilt hat as a method of aiming the camera, and a RGBW Neopixel LED strip as a light source / feedback mechanism. 

<!-- In the first stage of this project, I aim to be able to:
    - identify and extract a face from a live video stream
    - pan and tilt the camera such that a single face is always at the center of the screen
    - be able to recognise certain faces
    - be able to feedback personalised messages to different individuals


in the stage 1.5:
    - train a new face using the camera

In the second stage of this project, I will connect a Google Home that will allow me to
    - send personalised messages from the pi
    - send instructions to the Pi from the Google Home.

In the third stage of this project, I aim to be able to either:
    - build a 3d map of a room
        or
    - build a robot that is able to crawl (like a spider) around a room and follow me

In the fourth stage of this project:
    - i will take commands from images -->

## Setup
```sh
scp -r !(venv) face_tracker pi@192.168.0.55:Desktop/face_tracker
rsync -r face_tracker pi@192.168.0.55:Desktop/ --exclude=venv
```

<!-- ## References
https://towardsdatascience.com/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134
    - Low FPS on basic RPI hardware with Tensorflow 

https://www.pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/
    - OpenCV Face Detection and PID tuning
        - THIS IS IN PLACE

https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
    - OpenCV Face Detection and face_recognition Face Recognition
        - STILL TOO SLOW PROBABLY

https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
    - Background reading to above
https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/
    - Background reading to above
https://www.pyimagesearch.com/2020/01/06/raspberry-pi-and-movidius-ncs-face-recognition/
    - Background reading to above

https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/
    - Image Enrollment
        - THIS IS IN PLACE

https://circuitdigest.com/microcontroller-projects/raspberry-pi-and-opencv-based-face-recognition-system
    - OpenCV Face Recognition
        - THIS IS IN PLACE
https://www.instructables.com/id/Real-time-Face-Recognition-an-End-to-end-Project/
 -->

<!-- ## Steps
1. Setup Raspberry Pi to function with the Pan Tilt Module and Light
2. Raspberry Pi Generic Face Tracking
3. Raspberry Pi Specific Face Tracking

## Features
# Aim the camera in a certain location
# Detect a face
# Aim the camera at a face / multiple faces
# Identify a known face
# Train a new face


# Timelapse
raspistill -t 3600000 -tl 15000 -o image%04d.jpg

# reduce quality with
raspistill -w 640 -h 480 -t 3600000 -tl 1000 -o image%04d.jpg # specify dimenions
raspistill -n -q 10 # specify % https://www.raspberrypi.org/forums/viewtopic.php?t=54859

# stich together with
ffmpeg -r 24 -pattern_type glob -i '*.jpg' -s hd1080 -vcodec libx264 timelapse.mp4
 -->


<!-- Building Works
# 1 every second for an hour
raspistill -vf -hf -w 640 -h 480 -t 3600000 -tl 1000 -o ./timelamp/building/image%04d.jpg # specify dimenions

# 1 every second for 50 seconds
raspistill -vf -hf -w 640 -h 480 -t 50000 -tl 1000 -o ./timelamp/building/image%04d.jpg # specify dimenions

# 1 every 5 minutes for 6 hours
raspistill -vf -hf -w 640 -h 480 -t 21600000 -tl 300000 -o ./timelamp/building/image%04d.jpg # specify dimenions

# 1 every 5 minutes for 6 hours - HD
raspistill -vf -hf -w 1920 -h 1080 -t 21600000 -tl 300000 -o ./timelamp/building/image%04d.jpg # specify dimenions

ffmpeg -r 24 -pattern_type glob -i './timelamp/building_0/*.jpg' -s hd1080 -vcodec libx264 timelapse.mp4

ffmpeg -r 3 -pattern_type glob -i './timelamp/building_0/*.jpg' -s hd1080 -vcodec libx264 timelapse.mp4
 -->

ffmpeg -r 3 -pattern_type glob -i './timelamp/building_*/*.jpg' -s hd1080 -vcodec libx264 timelapse.mp4