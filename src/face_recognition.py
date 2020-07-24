from imutils.video import VideoStream

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
from utils import HaarFaceDetector

class PiFaceRecognition:
	"""
	Classifies an extracted face against a database of known faces.
	Adds a new face to the database to allow classification of new faces.
	"""
	def __init__(self):
		self.resolution = (900, 400)


	def enroll_images(self, rpi=True):
		"""
		Loads a camera and takes a picture at every 'K' keep press.
		"""
		# initialize the object center finder
		self.haar_face_detector = HaarFaceDetector(os.path.join(MODELS_DIRECTORY, 'haarcascade_frontalface_default.xml'))

		# initialize the video stream, allow the camera sensor to warm up,
		# and initialize the total number of example faces written to disk
		# thus far
		print("[INFO] starting video stream...")

		# vs = VideoStream(src=0).start()
		vs = VideoStream(usePiCamera=rpi).start()
		time.sleep(2.0)
		total = 0

		while True:
			frame = vs.read()
			# orig = frame.copy()
			# frame = imutils.resize(frame, width=400)

			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
		
			# if the `k` key was pressed, write the *original* frame to disk
			# so we can later process it and use it for face recognition
			if key == ord("k"):
				# p = os.path.sep.join([args["output"], "{}.png".format(
					# str(total).zfill(5))])
				# cv2.imwrite(p, orig)
				total += 1
			# if the `q` key was pressed, break from the loop
			elif key == ord("q"):
				break

		# print the total faces saved and do a bit of cleanup
		print("[INFO] {} face images stored".format(total))
		print("[INFO] cleaning up...")
		cv2.destroyAllWindows()
		vs.stop()

	def train_lbph_face_recogniser(self):
		pass

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
	fr.enroll_images(rpi=False)