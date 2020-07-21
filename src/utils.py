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
		rects = self.detector.detectMultiScale(gray, scaleFactor=1.05,
			minNeighbors=9, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)

		# return centers of faces and boxes
		if len(rects) == 0:
			return None
		else:
			center_and_rects = []
			for rect in rects:
				center_and_rects.append((self._get_center_of_rectangle(rect), rect))
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
