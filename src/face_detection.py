from pantilthat import PanTilt
import pantilthat as pth

from multiprocessing import Manager, Process
from imutils.video import VideoStream

import argparse
import signal
import time
import cv2

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import HaarFaceDetector, PID
from src.face_recognition import PiFaceRecognition
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
import json

import matplotlib.pyplot as plt
import pandas as pd

pth.light_mode(pth.WS2812)
pth.light_type(pth.GRBW)

class PiFaceDetector:
	"""
	Aim a Raspberry Pi camera at one or more faces.
	
	If single face:
			-> aim at center of face
	If multiple faces:
			-> aim at average of center of faces
			-> move from one face to the next, staying on each face for n seconds
	"""
	def __init__(self, rpi=True):
		self.rpi = rpi

		# initialise constants
		# pan/tilt constants
		self.pan_min_degree = -90
		self.pan_max_degree = 90
		self.tilt_min_degree = -90
		self.tilt_max_degree = 90

		# camera constants
		self.resolution = (912, 400)
		self.frame_center_X = self.resolution[0] / 2
		self.frame_center_Y = self.resolution[1] / 2

		# pid constants
		self.pan_p = 0.04
		self.pan_i = 0.04
		self.pan_d = 0.00
		# self.pan_p = 0.025
		# self.pan_i = 0.04
		# self.pan_d = 0.00
		self.tilt_p = 0.025
		self.tilt_i = 0.04
		self.tilt_d = 0.00

		self.aimer = PanTilt(
			servo1_min=745,
			servo1_max=2200,
			servo2_min=580,
			servo2_max=1910
		)

	def start_camera(self, face_position_X, face_position_Y):
		"""
		1. Begin video stream
		2. Extract faces from frames
		3. Display frames with bounding boxes
		4. Update global variables with:
			-> pixel coordinates of the center of the frame
			-> pixel coordinates of the center of the faces
		"""
		# signal trap to handle keyboard interrupt
		signal.signal(signal.SIGINT, signal_handler)

		# start the video stream and wait for the camera to warm up
		# vs = VideoStream(usePiCamera=self.rpi, resolution=self.resolution).start()
		print('Starting Camera')
		if self.rpi:
			vs = VideoStream(usePiCamera=self.rpi, resolution=self.resolution).start()
		else:
			vs = VideoStream(src=1, resolution=self.resolution).start()
		time.sleep(2.0)

		# initialize the object center finder
		face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))

		# initialise the recogniser
		# fr = PiFaceRecognition()

		# start recording
		filename = os.path.join(DATA_DIRECTORY, 'recordings', '{}.avi'.format(time.time()))
		cv2.VideoWriter_fourcc(*'MJPG')
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out = cv2.VideoWriter(filename, fourcc, 20.0, self.resolution)

		while True:
			# grab the frame from the threaded video stream and flip it
			# vertically (since our camera was upside down)
			frame = vs.read()
			frame = cv2.flip(frame, 0)

			# (H, W) = frame.shape[:2]
			# print('H', H)

			# find the object's location
			object_locations = face_detector.extract_faces(frame)
			people = [] # for setting colour

			# get first face for now
			if object_locations:
				# print('{} faces found.'.format(len(object_locations)))
				(face_position_X.value, face_position_Y.value) = object_locations[0][0]
				# print(object_locations[0][0])
				# extract the bounding box and draw it
				for pos, rect, neighbour, weight in object_locations:
					(x, y, w, h) = rect
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
					
					# recogniser part
					gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray

					person, confidence = PiFaceRecognition.infer_lbph_face_recogniser(gray_frame[y:y+h,x:x+w])
					people.append(person)
					cv2.putText(frame, person, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
					cv2.putText(frame, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  

			else:
				print('No faces found.')
				face_position_X.value, face_position_Y.value = (self.frame_center_X, self.frame_center_Y)

			if len(people) > 1:
				# set to orange
				pth.set_all(255, 127, 80, 50)
				pth.show()
			elif 'jai' in people:
				# set to green
				pth.set_all(173, 255, 47, 50)
				pth.show()
			elif 'alleeya' in people:
				# set to purple
				pth.set_all(221, 160, 221, 50)
				pth.show()
			else:
				pth.clear()
				pth.show()

			# display the frame to the screen
			cv2.imshow("Pan-Tilt Face Tracking", frame)
			out.write(frame)
			cv2.waitKey(1)


	def pid_process(self, servo_angle, p, i, d, obj_coord, center_coord, tuning_time_data, tuning_error_data, tuning_angle_data):
		"""
		Calculates and updates servo angle
		"""
		# signal trap to handle keyboard interrupt
		signal.signal(signal.SIGINT, signal_handler)

		# create a PID and initialize it
		pid = PID(p, i, d)
		pid.initialize()
		
		while True:
			# calculate the error
			error = center_coord - obj_coord.value
			print(center_coord, obj_coord.value, error, pid.update(error))

			# update the value
			servo_angle.value = pid.update(error)

			# store tuning data
			tuning_time_data.append(pid.prevTime)
			tuning_error_data.append(servo_angle.value)
			tuning_angle_data.append(error)
		
			# For showing lights 
			# print('error: ', error, 'new_angle: ', servo_angle.value)
			# if servo_angle.value <= 10:
			# 	pth.set_all(255, 255, 0)
			# 	pth.show()
			# else:
			# 	pth.clear()
			# 	pth.show()
		
	
	def pan_pid_process(self, pan_angle, obj_coord_X, tuning_time_data, tuning_error_data, tuning_angle_data):
		"""
		Pan Angle is updated
		"""
		self.pid_process(
			pan_angle,
			self.pan_p,
			self.pan_i,
			self.pan_d,
			obj_coord_X,
			self.frame_center_X,
			tuning_time_data, 
			tuning_error_data, 
			tuning_angle_data
		)

	def tilt_pid_process(self, tilt_angle, obj_coord_Y, tuning_time_data, tuning_error_data, tuning_angle_data):
		self.pid_process(
			tilt_angle,
			self.tilt_p,
			self.tilt_i,
			self.tilt_d,
			obj_coord_Y,
			self.frame_center_Y,
			tuning_time_data, 
			tuning_error_data, 
			tuning_angle_data
		)

	def _angle_in_range(self, angle):
		# determine the input value is in the supplied range
		return (angle >= -90 and angle <= 90 and angle != 0)

	def set_servos(self, pan_angle, tilt_angle):
		signal.signal(signal.SIGINT, signal_handler)
		
		while True:
			# the pan and tilt angles are reversed
			pan_angle_this = -1 * pan_angle.value
			tilt_angle_this = -1 * tilt_angle.value

			# if the pan angle is within the range, pan
			if self._angle_in_range(pan_angle_this):
				# print('Pan in range', pan_angle_this)
				self.aimer.pan(pan_angle_this)

			# if the tilt angle is within the range, tilt
			if self._angle_in_range(tilt_angle_this):
				# print('Tilt in range', tilt_angle_this)
				self.aimer.tilt(tilt_angle_this)


	def _save_tuning_process(self, p, i, d, tuning_time_data, tuning_error_data, tuning_angle_data, directory):
		data_to_save = {
			'p': p,
			'i': i,
			'd': d,
			't': [],
			'error': [],
			'angle': [],
		}
		while True:
			for t, error, angle in zip(tuning_time_data, tuning_error_data, tuning_angle_data):
				data_to_save['t'].append(t)
				data_to_save['error'].append(error)
				data_to_save['angle'].append(angle)

			data_filepath = os.path.join(directory, f'{p}_{i}_{d}.json')
			image_filepath = os.path.join(directory, f'{p}_{i}_{d}.png')

			with open(data_filepath, 'w') as fp:
				json.dump(data_to_save, fp)

			df = pd.DataFrame(data_to_save)
			df = df.drop_duplicates()

			plt.plot(df['t'], df['error'], marker='')
			plt.title(f'{p}_{i}_{d}.png')
			plt.savefig(image_filepath)

	
	def save_pan_tuning_process(self, tuning_time_data, tuning_error_data, tuning_angle_data):
		self._save_tuning_process(
			self.pan_p,
			self.pan_i,
			self.pan_d,
			tuning_time_data, 
			tuning_error_data, 
			tuning_angle_data,
			os.path.join(DATA_DIRECTORY, 'pan_error')
		)
		
	def save_tilt_tuning_process(self, tuning_time_data, tuning_error_data, tuning_angle_data):
		self._save_tuning_process(
			self.tilt_p,
			self.tilt_i,
			self.tilt_d,
			tuning_time_data, 
			tuning_error_data, 
			tuning_angle_data,
			os.path.join(DATA_DIRECTORY, 'tilt_error')
		)


# function to handle keyboard interrupt
def signal_handler(sig, frame):
    # print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")
    
    # clear lights
    pth.clear()
    pth.show()

    # exit
    sys.exit()


if __name__ == '__main__':
    pi_face_detector = PiFaceDetector(rpi=True)
    # pi_face_detector = PiFaceDetector(rpi=False)

    start_pan = 0
    start_tilt = 30
    pth.pan(start_pan)
    pth.tilt(start_tilt)

    with Manager() as manager:
        print('Start Manager')
        # set integer values for the object's (x, y)-coordinates
        obj_coord_X = manager.Value("i", pi_face_detector.frame_center_X)
        obj_coord_Y = manager.Value("i", pi_face_detector.frame_center_Y)

        # pan and tilt values will be managed by independed PIDs
        pan_angle = manager.Value("i", start_pan)
        tilt_angle = manager.Value("i", start_tilt)

        # initialise tuning data variable holder to draw graphs
        tuning_time_data = manager.list()
        tuning_error_data = manager.list()
        tuning_angle_data = manager.list()

        print('Define process')
        process_start_camera = Process(target=pi_face_detector.start_camera,
            args=(obj_coord_X, obj_coord_Y))

        process_panning = Process(target=pi_face_detector.pan_pid_process,
            args=(pan_angle, obj_coord_X, tuning_time_data, tuning_error_data, tuning_angle_data))

        process_tilting = Process(target=pi_face_detector.tilt_pid_process,
            args=(tilt_angle, obj_coord_Y, tuning_time_data, tuning_error_data, tuning_angle_data))

        process_set_servos = Process(target=pi_face_detector.set_servos, args=(pan_angle, tilt_angle))

        # store data
        process_save_pan_tuning_process = Process(target=pi_face_detector.save_pan_tuning_process,
            args=(tuning_time_data, tuning_error_data, tuning_angle_data))
                
        process_save_tilt_tuning_process = Process(target=pi_face_detector.save_tilt_tuning_process,
            args=(tuning_time_data, tuning_error_data, tuning_angle_data))
    
        
        # start all 4 processes
        print('Start process.')
        process_start_camera.start()
        process_panning.start()
        #process_tilting.start()
        process_set_servos.start()
        process_save_pan_tuning_process.start()
        #process_save_tilt_tuning_process.start()

        # join all 4 processes
        print('Join process.')
        process_start_camera.join()
        process_panning.join()
        #process_tilting.join()
        process_set_servos.join()
        process_save_pan_tuning_process.join()
        #process_save_tilt_tuning_process.join()