from imutils.video import VideoStream
from pantilthat import PanTilt
from time import sleep
import cv2

def point_camera(pan, tilt):
	aimer = PanTilt(
		servo1_min=745,
		servo1_max=2200,
		servo2_min=580,
		servo2_max=1910
	)
	aimer.pan(pan)
	aimer.tilt(tilt)

# start camera
# vs = VideoStream(src=1).start() # computer
vs = VideoStream(usePiCamera=True).start()
sleep(2.0)
while True:
	frame = vs.read()
	cv2.imshow("Pan-Tilt Face Tracking", frame)
	# print(frame)

	sleep(3)
	point_camera(0, 0)
	cv2.waitKey(1)

	# capture key
	# break