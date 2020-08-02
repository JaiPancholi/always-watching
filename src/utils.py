import cv2
class HaarFaceDetector:
	def __init__(self, haar_path):
		# load OpenCV's Haar cascade face detector
		self.detector = cv2.CascadeClassifier(haar_path)

	def extract_faces(self, frame):
		"""
		Identify face locations from a frame and return coordinates.

		:param frame: Image - what type is this?
		"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect all faces in the input frame
		objects = self.detector.detectMultiScale3(gray, scaleFactor=1.05,
			minNeighbors=9, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE,
			outputRejectLevels=True
		)

		# print(objects)

		# return centers of faces and boxes
		if len(objects) == 0:
			return None
		else:
			center_and_rects = []
			for rect, neighbour, weight in zip(objects[0], objects[1], objects[2]):
				center_and_rects.append((self._get_center_of_rectangle(rect), rect, neighbour, weight))
			return center_and_rects
	
	@staticmethod
	def _get_center_of_rectangle(rect):
		(x, y, w, h) = rect
		mid_X = int(x + (w / 2.0))
		mid_Y = int(y + (h / 2.0))
		return mid_X, mid_Y


# Taken from https://www.pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/
import time
class PID:
	def __init__(self, kP=1, kI=0, kD=0):
		# initialize gains
		self.kP = kP
		self.kI = kI
		self.kD = kD

	def initialize(self):
		# intialize the current and previous time
		self.currTime = time.time()
		self.prevTime = self.currTime

		# initialize the previous error
		self.prevError = 0

		# initialize the term result variables
		self.cP = 0
		self.cI = 0
		self.cD = 0

	def update(self, error, sleep=0.2):
		# pause for a bit
		time.sleep(sleep)

		# grab the current time and calculate delta time
		self.currTime = time.time()
		deltaTime = self.currTime - self.prevTime

		# delta error
		deltaError = error - self.prevError

		# proportional term
		self.cP = error

		# integral term
		self.cI += error * deltaTime

		# derivative term and prevent divide by zero
		self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0

		# save previous time and error for the next update
		self.prevTime = self.currTime
		self.prevError = error

		# sum the terms and return
		return sum([
			self.kP * self.cP,
			self.kI * self.cI,
			self.kD * self.cD])

from pantilthat import PanTilt
import math
def test_hardware():
	"""
	Test full range of motion as aswell as lights
	"""
	# start in the middle
	# u shape to the top right
	# straght down
	# figure of eithe back to middle

	aimer = PanTilt(
		servo1_min=745,
		servo1_max=2200,
		servo2_min=580,
		servo2_max=1910
	)

	for t in range(0, 91):
		# print(t, math.sin(math.radians(t) + math.radians(270)), math.sin(math.radians(t) + math.radians(270)) * 90)
		x = t
		y = math.sin(math.radians(t) + math.radians(270)) * 90
		aimer.pan(x)
		aimer.tilt(y)
		time.sleep(0.01)

if __name__ == '__main__':
	# # Test on Laptop
	# from imutils.video import VideoStream
	# import os
	# MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
	# # DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')	

	# vs = VideoStream(src=1).start()

	# face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))
	# while True:
	# 	frame = vs.read()
	# 	objects = face_detector.extract_faces(frame)
		
	# 	# print(object_locations)
	# 	if objects:
	# 		for centre, (x,y,w,h), neigbour, weights,  in objects:
	# 			print(neigbour, weights)
	# 			neigbour = str(neigbour[0])
	# 			weights = str(weights[0])
	# 			cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
	# 			cv2.putText(frame, neigbour, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	# 			cv2.putText(frame, weights, (x+5, y+h+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


	# 	cv2.imshow("Pan-Tilt Face Tracking", frame)
	# 	cv2.waitKey(1)

	test_hardware()