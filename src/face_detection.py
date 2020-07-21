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
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# TODO return to (pan, tilt) = (0, 0)

class PiFaceDetector:
	"""
	Aim a Raspberry Pi camera at one or more faces.
	
	If single face:
			-> aim at center of face
	If multiple faces:
			-> aim at average of center of faces
			-> move from one face to the next, staying on each face for n seconds
	"""
	def __init__(self):
		# initialise constants
		# pan/tilt constants
		self.pan_min_degree = -90
		self.pan_max_degree = 90
		self.tilt_min_degree = -90
		self.tilt_max_degree = 90

		# camera constants
		self.resolution = (900, 400)
		self.frame_center_X = self.resolution[0] / 2
		self.frame_center_Y = self.resolution[1] / 2

		# pid constants
		self.pan_p = 0.04
		self.pan_i = 0.015
		self.pan_d = 0.01
		self.tilt_p = 0.06
		self.tilt_i = 0.10
		self.tilt_d = 0.01

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
		# start the video stream and wait for the camera to warm up
		vs = VideoStream(usePiCamera=True, resolution=self.resolution).start()
		time.sleep(2.0)

		# initialize the object center finder
		face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))

		while True:
			# grab the frame from the threaded video stream and flip it
			# vertically (since our camera was upside down)
			frame = vs.read()
			frame = cv2.flip(frame, 0)

			# (H, W) = frame.shape[:2]
			# print('H', H)

			# find the object's location
			object_locations = face_detector.extract_faces(frame)
			# get first face for now
			if object_locations:
				print('{} faces found.'.format(len(object_locations)))
				(face_position_X.value, face_position_Y.value) = object_locations[0][0]
				# ((objX.value, objY.value), rect) = objectLoc
				
				# extract the bounding box and draw it
				for pos, rect in object_locations:
					(x, y, w, h) = rect
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			else:
				print('No faces found.')
				face_position_X.value, face_position_Y.value = (self.frame_center_X, self.frame_center_Y)
				# face_position_X.value, face_position_Y.value = (0, 0)

			# display the frame to the screen
			cv2.imshow("Pan-Tilt Face Tracking", frame)
			cv2.waitKey(1)


	def pid_process(self, servo_angle, p, i, d, obj_coord, center_coord):
		"""
		Calculates and updates servo angle
		"""
		# signal trap to handle keyboard interrupt
		signal.signal(signal.SIGINT, signal_handler)

		# create a PID and initialize it
		p = PID(p, i, d)
		p.initialize()
		
		while True:
			# calculate the error
			error = center_coord - obj_coord.value
			# update the value
			servo_angle.value = p.update(error)
			# print('error: ', error, 'new_angle: ', servo_angle.value)
			# if servo_angle.value <= 10:
			# 	pth.set_all(255, 255, 0)
			# 	pth.show()
			# else:
			# 	pth.clear()
			# 	pth.show()
		
	
	def pan_pid_process(self, pan_angle, obj_coord_X):
		"""
		Pan Angle is updated
		"""
		self.pid_process(
			pan_angle,
			self.pan_p,
			self.pan_i,
			self.pan_d,
			obj_coord_X,
			self.frame_center_X
		)

	def tilt_pid_process(self, tilt_angle, obj_coord_Y):
		self.pid_process(
			tilt_angle,
			self.tilt_p,
			self.tilt_i,
			self.tilt_d,
			obj_coord_Y,
			self.frame_center_Y
		)

	def _angle_in_range(self, angle):
		# determine the input value is in the supplied range
		return (angle >= -90 and angle <= 90 and angle != 0)

	def set_servos(self, pan_angle, tilt_angle):
		signal.signal(signal.SIGINT, signal_handler)
		
		while True:
			# time.sleep(0.2)

			# the pan and tilt angles are reversed
			pan_angle_this = -1 * pan_angle.value
			tilt_angle_this = -1 * tilt_angle.value

			# # if pan_angle_this != 0:
			# if pan_angle_this >= -90 and pan_angle_this <= 90 and pan_angle_this!=0:
			# 	print('pan_angle_this', pan_angle_this, type(pan_angle_this))
			# 	# pth.pan(pan_angle_this)
			# 	self.aimer.pan(pan_angle_this)

			# if the pan angle is within the range, pan
			if self._angle_in_range(pan_angle_this):
				print('Pan in range', pan_angle_this)
				self.aimer.pan(pan_angle_this)
			else:
				print('Pan not in range', pan_angle_this)

			# if the tilt angle is within the range, tilt
			if self._angle_in_range(tilt_angle_this):
				print('Tilt in range', tilt_angle_this)
				self.aimer.tilt(tilt_angle_this)
			else:
				print('Tilt not in range', tilt_angle_this)


# function to handle keyboard interrupt
def signal_handler(sig, frame):
	# print a status message
	print("[INFO] You pressed `ctrl + c`! Exiting...")

	pth.clear()
	pth.show()

	# exit
	sys.exit()


if __name__ == '__main__':
	pi_face_detector = PiFaceDetector() 
	pth.pan(0)
	pth.tilt(-40)

	with Manager() as manager:
		# set integer values for the object's (x, y)-coordinates
		obj_coord_X = manager.Value("i", pi_face_detector.frame_center_X)
		obj_coord_Y = manager.Value("i", pi_face_detector.frame_center_Y)

		# pan and tilt values will be managed by independed PIDs
		pan_angle = manager.Value("i", 0)
		tilt_angle = manager.Value("i", 0)

		process_start_camera = Process(target=pi_face_detector.start_camera,
			args=(obj_coord_X, obj_coord_Y))

		process_panning = Process(target=pi_face_detector.pan_pid_process,
			args=(pan_angle, obj_coord_X))

		process_tilting = Process(target=pi_face_detector.tilt_pid_process,
			args=(tilt_angle, obj_coord_Y))

		process_set_servos = Process(target=pi_face_detector.set_servos, args=(pan_angle, tilt_angle))

		# start all 4 processes
		process_start_camera.start()
		process_panning.start()
		#process_tilting.start()
		process_set_servos.start()

		# join all 4 processes
		process_start_camera.join()
		process_panning.join()
		#process_tilting.join()
		process_set_servos.join()