# pi_face_tracker
This is repository contains the software for a Raspberry Pi face tracking system. This system uses a Raspberry Pi as a mobile computer, Raspberry Pi Camera as an image capture method, a Pan Tilt hat as a method of aiming the camera, and a RGBW Neopixel LED strip as a light source / feedback mechanism. 

In the first stage of this project, I aim to be able to:
    - identify and extract a face from a live video stream
    - pan and tilt the camera such that a single face is always at the center of the screen
    - be able to recognise certain faces
    - be able to feedback personalised messages to different individuals


In the second stage of this project, I will connect a Google Home that will allow me to
    - send personalised messages from the pi
    - send instructions to the Pi from the Google Home.

In the third stage of this project, I aim to be able to either:
    - build a 3d map of a room
        or
    - build a robot that is able to crawl (like a spider) around a room and follow me

In the fourth stage of this project:
    - i will take commands from images

## Setup
```sh
scp -r !(venv) face_tracker pi@192.168.0.55:Desktop/face_tracker
rsync -r face_tracker pi@192.168.0.55:Desktop/ --exclude=venv

```

## References
https://towardsdatascience.com/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134
    - Low FPS on basic RPI hardware
https://www.pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/

https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/
https://www.pyimagesearch.com/2020/01/06/raspberry-pi-and-movidius-ncs-face-recognition/

## Steps
1. Setup Raspberry Pi to function with the Pan Tilt Module and Light
2. Raspberry Pi Generic Face Tracking
3. Raspberry Pi Specific Face Tracking

## Features
# Aim the camera in a certain location
# Detect a face
# Aim the camera at a face / multiple faces
# Identify a known face
# Train a new face