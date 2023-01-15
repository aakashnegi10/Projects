# USAGE
# python drowsy.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python drowsy.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
# python drowsy.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav --webcam 0

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play the alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal eye landmarks
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of vertical mouth landmarks
	A = dist.euclidean(mouth[2], mouth[10])
	B = dist.euclidean(mouth[4], mouth[8])
	# compute the euclidean distances between the horizontal mouth landmarks
	C = dist.euclidean(mouth[0], mouth[6])
	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)
	# return the mouth aspect ratio
	return mar

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path To Facial Landmark Predictor(.DAT)")
ap.add_argument("-a", "--alarm", type=str, default="", help="Path To Alarm(.WAV) File")
ap.add_argument("-w", "--webcam", type=int, default=0, help="Index Of Webcam On The System")
args = vars(ap.parse_args())

# threshold for the eye aspect ratio to indicate a blink
EYE_AR_THRESH = 0.3
# threshold for the mouth aspect ratio to indicate a yawn
MOUTH_AR_THRESH = 0.65
# number of consecutive frames the eye must be below and the mouth must be above the threshold to set off the alarm
AR_CONSEC_FRAMES = 48
# initialize the frame counter
COUNTER = 0
# a boolean used to indicate if the alarm is going off
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] Loading Facial Landmark Predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye and the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] Starting VideoStream Thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=760)
	# convert the frame from the threaded video file stream to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for face region
		shape = predictor(gray, rect)
		# convert the facial landmarks coordinates to a NumPy array
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		# extract the mouth coordinates
		mouth = shape[mStart:mEnd]
		# compute the eye aspect ratio for both eyes
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#compute the mouth aspect ratio for mouth
		mar = mouth_aspect_ratio(mouth)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		# compute the convex hull for both eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# compute the convex hull for mouth
		mouthHull = cv2.convexHull(mouth)
		# visualize both eyes and mouth
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		# check to see if the EAR is below the threashold or MAR is above the threashold
		if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
			# if so, increment the blink frame counter
			COUNTER += 1
			if COUNTER >= AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True
					if args["alarm"] != "":
						# start a thread to have the alarm sound played in the background
						t = Thread(target=sound_alarm, args=(args["alarm"],))
						t.deamon = True
						t.start()
				# draw an alert on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
		else:
			# otherwise, reset the counter and alarm
			COUNTER = 0
			ALARM_ON = False
		# draw the computed EAR and MAR on the frame
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (600, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (600, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()
exit()