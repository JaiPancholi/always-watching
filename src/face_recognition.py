from imutils.video import VideoStream

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
from utils import HaarFaceDetector

import time
import cv2

from PIL import Image
import numpy as np
import json

class PiFaceRecognition:
	"""
	Classifies an extracted face against a database of known faces.
	Adds a new face to the database to allow classification of new faces.
	"""
	def __init__(self):
		self.resolution = (900, 400)

	def _set_or_get_person_directory(self, person):
		"""
		Create if training folder does not exists.
		"""
		image_directory = os.path.join(DATA_DIRECTORY, 'faces', person)
		if not os.path.exists(image_directory):
			print('Directory does not exist. Creating directroy for ', person)
			os.makedirs(image_directory)
		return image_directory

	def enroll_images(self, person, rpi=True):
		"""
		Loads a camera and takes a picture at every 'K' keep press.
		"""
		image_directory = self._set_or_get_person_directory(person)

		# initialize the object center finder
		self.haar_face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))

		# start camera, giving time to warm up
		print('Starting video stream...')
		if rpi:
			vs = VideoStream(usePiCamera=rpi).start()
		else:
			vs = VideoStream(src=1).start()

		time.sleep(2.0)
		total = 0

		while True:
			frame = vs.read()
			if rpi:
				frame = cv2.flip(frame, -1) # flip video image vertically

			# get key
			key = cv2.waitKey(1) & 0xFF

			# Extact and draw faces
			rects = self.haar_face_detector.extract_faces(frame)
			if rects:
				for centre, (x,y,w,h) in rects:
					cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)

			cv2.imshow('Frame', frame)
		
			# if the `k` key was pressed, write the *original* frame to disk
			# so we can later process it and use it for face recognition
			if key == ord("k"):
				if rects:
					for centre, (x,y,w,h) in rects:
						frame = frame[y:y+h,x:x+w]
						image_path = os.path.join(image_directory, f'{total}.jpg')
						cv2.imwrite(image_path, frame)
						total += 1

			# if the `q` key was pressed, break from the loop
			elif key == ord("q"):
				break

		# print the total faces saved and do a bit of cleanup
		print('{} face images stored'.format(total))
		cv2.destroyAllWindows()
		vs.stop()

	def train_lbph_face_recogniser(self):
		"""
		...
		"""
		people = {}
		faces = []
		labels = []
		faces_directory = os.path.join(DATA_DIRECTORY, 'faces')
		for i, person_folder in enumerate(os.listdir(faces_directory)):
			if person_folder == '.DS_Store':
				continue

			# store person to index lookup
			people[person_folder] = i
			
			# load images into array
			person_directory = os.path.join(faces_directory, person_folder)
			for image_name in os.listdir(person_directory):
				image_path = os.path.join(person_directory, image_name)
				img = Image.open(image_path).convert('L') # convert it to grayscale
				img_array = np.array(img, 'uint8')
				faces.append(img_array)
				labels.append(people[person_folder])

		# initialise and train model
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizer.train(faces, np.array(labels))
		
		# Save the model into trainer/trainer.yml
		model_path = os.path.join(MODELS_DIRECTORY, 'custom-opencv.yml')
		recognizer.write(model_path) # recognizer.save() worked on Mac, but not on Pi
		
		# save label lookup
		model_path = os.path.join(MODELS_DIRECTORY, 'custom-opencv-lookup.json')
		with open(model_path, 'w') as fp:
			json.dump(people, fp)

	def infer_lbph_face_recogniser(self, face):
		# load model
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		model_path = os.path.join(MODELS_DIRECTORY, 'custom-opencv.yml')
		recognizer.read(model_path)

		# load label lookup
		model_path = os.path.join(MODELS_DIRECTORY, 'custom-opencv-lookup.json')
		with open(model_path, 'r') as fp:
			people = json.load(fp)

		# reverse lookup
		index_to_person = {idx: person for person, idx in people.items()}

		# predict
		face_index, confidence = recognizer.predict(face)

		return index_to_person[face_index], confidence

	def start_camera(self, rpi=True):
		# initialize the object center finder
		self.haar_face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))

		# start camera, giving time to warm up
		print('Starting video stream...')
		if rpi:
			vs = VideoStream(usePiCamera=rpi).start()
		else:
			vs = VideoStream(src=1).start()
		print('Starting')
		time.sleep(2.0)

		font = cv2.FONT_HERSHEY_SIMPLEX

		while True:
			frame = vs.read()

			if rpi:
				frame = cv2.flip(frame, -1) # flip video image vertically

			# get key
			key = cv2.waitKey(1) & 0xFF

			# Extact and draw faces
			rects = self.haar_face_detector.extract_faces(frame)
			if rects:
				for centre, (x,y,w,h) in rects:
					cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)

					# convert to gray
					gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

					person, confidence = self.infer_lbph_face_recogniser(gray_frame[y:y+h,x:x+w])
					cv2.putText(frame, person, (x+5, y-5), font, 1, (255,255,255), 2)
					cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1)  

			# show image
			cv2.imshow('Frame', frame)

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# print the total faces saved and do a bit of cleanup
		print('{} face images stored'.format(total))
		cv2.destroyAllWindows()
		vs.stop()


	def add_face(self):
		"""
		1. prompt user with name of face to add
		2. give user ability to pan-tilt camera
		3. give user ability to capture image from camera
			extract face
			give user ability to 'accept' or 'deny' face extraction
		4. give user ability to capture new face or stop capturing and close gracefully.
		"""
		name = input('Enter the name of the person to train: ')
		print()
		print('You have typed ', name)




    # cv2.imshow('camera',img) 
    # k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    # if k == 27:
    #     break

	# def train_embedding(self):
	#     pass

	# def infer_embedding(self):
	#     pass

	# def train_svm(self):
	#     pass

# Take Pictures of Face (face enrollment)
	# Simple

# Build Recogniser
	# V1
		# Get Embedding Vectors of Face
		# Add Vectors To a Database

	# V2
		# LBPHFaceRecognizer_create
		# https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

# Start Camera Stream
# Extract Face From Frame
# Run recogniser
# Draw name on face with probability

# Things to note:
# Do face recognition every n frames
# train on laptop, infer on rpi

if __name__ == '__main__':
	fr = PiFaceRecognition()
	# fr.enroll_images('jai', rpi=False)
	# fr.train_lbph_face_recogniser()
	# fr.start_camera(rpi=False)
	fr.add_face()