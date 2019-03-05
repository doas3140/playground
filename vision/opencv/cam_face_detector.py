# usage: python cam_face_detector.py

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

args = {}
args["prototxt"] = "./utils/face_model/deploy.prototxt.txt"
args["model"] = "./utils/face_model/res10_300x300_ssd_iter_140000.caffemodel"
args["confidence"] = 0.5

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	# grab the frame from the threaded video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability)
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()