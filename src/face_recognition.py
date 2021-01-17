from imutils.video import VideoStream

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
from utils import HaarFaceDetector
from pantilthat import PanTilt
import keyboard

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

		self.font = cv2.FONT_HERSHEY_SIMPLEX
		
		# initialize the object center finder
		self.haar_face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))

		# start camera, giving time to warm up
		print('Starting video stream...')
		if rpi:
			self.vs = VideoStream(usePiCamera=rpi, resolution=self.resolution).start()
		else:
			self.vs = VideoStream(src=1, resolution=self.resolution).start()
		time.sleep(2.0)

	def _set_or_get_person_directory(self, person):
		"""
		Create if training folder does not exists.
		"""
		image_directory = os.path.join(DATA_DIRECTORY, 'faces', person)
		if not os.path.exists(image_directory):
			print('Directory does not exist. Creating directroy for ', person)
			os.makedirs(image_directory)
		return image_directory

	# def enroll_images(self, person, rpi=True, infer_current_faces=False):
	def start_camera(self, purpose=None, person=None, rpi=True):
		"""
		Loads a camera and takes a picture at every 'K' keep press.
		"""
		if purpose not in ('enroll', 'infer'):
			raise ValueError('purpose keyword one of ("enroll", "infer")')
		if purpose == 'enroll' and person is None:
			raise ValueError('Pass value for keyword "purpose" if purpose is "enroll"')

		image_directory = self._set_or_get_person_directory(person)

		total = 0

		while True:
			frame = self.vs.read()
			if rpi:
				frame = cv2.flip(frame, -1) # flip video image vertically

			# Extact and draw faces
			rects = self.haar_face_detector.extract_faces(frame)
			if rects:
				for centre, (x,y,w,h), _, _ in rects:
					cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
					if purpose == 'infer':
						gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						person, confidence = self.infer_lbph_face_recogniser(gray_frame[y:y+h,x:x+w])
						cv2.putText(frame, person, (x+5, y-5), self.font, 1, (255,255,255), 2)
						cv2.putText(frame, str(confidence), (x+5, y+h-5), self.font, 1, (255,255,0), 1)  

			key = cv2.waitKey(1)
			cv2.imshow('Frame', frame)

			if purpose == 'enroll':
				# if the `k` key was pressed, write the *original* frame to disk
				# so we can later process it and use it for face recognition
				if key == ord("k"):
					if rects:
						for centre, (x,y,w,h), _, _ in rects:
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
		self.vs.stop()

	@staticmethod
	def train_lbph_face_recogniser():
		"""
		...
		"""
		print('Training recogniser.')
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
				if image_name == '.DS_Store':
					continue
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

		print('Recogniser trained and saved.')


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

	# def train_embedding(self):
	#     pass

	# def infer_embedding(self):
	#     pass

	# def train_svm(self):
	#     pass


import argparse
def parse_args():
	parser = argparse.ArgumentParser(description='Train an LBPH recogniser.')
	
	parser.add_argument('purpose', help='an integer for the accumulator')
	parser.add_argument('--person', help='sum the integers (default: find the max)')

	args = parser.parse_args()
	print(args)
	return args


# Build Recogniser
	# V1
		# Get Embedding Vectors of Face
		# Add Vectors To a Database

	# V2
		# LBPHFaceRecognizer_create
		# https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

if __name__ == '__main__':
	# fr = PiFaceRecognition()
	# # fr.enroll_images('jai', rpi=True)
	# # fr.enroll_images('alleeya', rpi=True)
	# # fr.train_lbph_face_recogniser()
	# fr.enroll_images('noone', infer_current_faces=True)
	args = parse_args()
	rpi = True
	if args.purpose == 'infer':
		# fr = PiFaceRecognition().start_camera(purpose=args.purpose, rpi=rpi)
		pass
	elif args.purpose == 'enroll':
		person = args.person
		# fr = PiFaceRecognition().start_camera(purpose=args.purpose, person=person, rpi=rpi)
		pass
	elif args.purpose == 'train':
		PiFaceRecognition.train_lbph_face_recogniser()

	# python src/face_recognition.py enroll --person jai
	# python src/face_recognition.py infer
	# python src/face_recognition.py train