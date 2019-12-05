import tensorflow as tf
import numpy as np
import time
import cv2
import threading
import socket
import os

#os.system("fuser -k 21567/tcp")
#os.system("fuser -k 21567/tcp")

label_map = ["Person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
            "boat", "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "", "backpack", "umbrella", "", "", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "", "dining table", "", "", "toilet",
            "", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
            "oven", "toaster", "sink", "refrigerator", "", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

list_of_near_obj = []
tag = ""
lock = threading.Lock()


def socket_thread_function():
	global tag, list_of_near_obj
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
		soc.bind(('', 21567))
		print("Started server on port 21567 and address " + socket.gethostname())
		soc.listen()
		while True:
			(clientsocket, address) = soc.accept()
			print("Connected by " + str(address))
			while True:
				try:            
					if tag != "":
						with lock:
							clientsocket.send(bytes(tag + "\n", "utf-8"))
						time.sleep(1.6)
				except:
					break

class TFLITE_DETECT:
	def __init__(self, model_path, labels):
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()
		self.labels = labels
		
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
	def detect_image(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.input_shape = self.input_details[0]['shape']
		image = cv2.resize(image, (self.input_shape[1], self.input_shape[2]))
		input_data = np.expand_dims(image, axis=0)

		self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
		self.interpreter.invoke()
		
		detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
		detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
		detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
		num_boxes = self.interpreter.get_tensor(self.output_details[3]['index'])
		
		return detection_boxes, detection_classes, detection_scores, num_boxes, image
		
class StereoVision:
	def __init__(self, minDisparity=0, numDisparities=160, blockSize=5, window_size=3, 
				disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=0, speckleRange=2,
				preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, lmbda=80000, sigma=1.2, 
				visual_multiplier=1.0):
				
		self.left_matcher = cv2.StereoSGBM_create(
								minDisparity=minDisparity,
								numDisparities=numDisparities,             
								blockSize=blockSize,
								P1=8 * 3 * window_size ** 2,    
								P2=32 * 3 * window_size ** 2,
								disp12MaxDiff=disp12MaxDiff,
								uniquenessRatio=uniquenessRatio,
								speckleWindowSize=speckleWindowSize,
								speckleRange=speckleRange,
								preFilterCap=preFilterCap,
								mode=mode
							)
		
		self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
		self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
		self.wls_filter.setLambda(lmbda)
		self.wls_filter.setSigmaColor(sigma)
		
	def generate_depth_map(self, leftFrame, rightFrame):
		leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
		rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
		displ = self.left_matcher.compute(leftFrame, rightFrame)  # .astype(np.float32)/16
		dispr = self.right_matcher.compute(rightFrame, leftFrame)  # .astype(np.float32)/16
		displ = np.int16(displ)
		dispr = np.int16(dispr)
		filteredImg = self.wls_filter.filter(displ, leftFrame, None, dispr)

		filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
		filteredImg = np.uint8(filteredImg)
		return filteredImg
		
class Main:
	def __init__(self, minDisparity=0, numDisparities=160, blockSize=5, window_size=3, 
				disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=0, speckleRange=2,
				preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, lmbda=80000, sigma=1.2, 
				visual_multiplier=1.0, model_path="/home/pi/Desktop/detect.tflite", labels=label_map):
		self.tflite_instance = TFLITE_DETECT(model_path, labels)
		self.stereo_instance = StereoVision(minDisparity, numDisparities, blockSize, window_size, 
											disp12MaxDiff, uniquenessRatio, speckleWindowSize, 
											speckleRange, preFilterCap, mode, lmbda, sigma, 
											visual_multiplier)
		self.left_camera = cv2.VideoCapture(2)
		self.right_camera = cv2.VideoCapture(0)
		self.counter = 0
		
	def main_recording(self):
		global list_of_near_obj, tag
		frame_rate_calc = 1
		freq = cv2.getTickFrequency()
		
		while True:
			if not (self.left_camera.grab() and self.right_camera.grab()):
				print("No more frames")
				break

			_, leftFrame = self.left_camera.retrieve()
			_, rightFrame = self.right_camera.retrieve()
			
			starttime = cv2.getTickCount()
			
			det_box, det_class, det_score, num_box, image = self.tflite_instance.detect_image(leftFrame)
			filter_img = self.stereo_instance.generate_depth_map(leftFrame, rightFrame)
			
			for i in range(int(num_box[0])):
				if det_score[0, i] > 0.6:
					class_id = det_class[0, i]
					x = det_box[0, i, [1, 3]] * self.tflite_instance.input_shape[1]
					y = det_box[0, i, [0, 2]] * self.tflite_instance.input_shape[2]
					image = cv2.rectangle(image, (x[0], y[0]), (x[1], y[1]), (255, 150, 0), 2)
					image = cv2.putText(image, str(label_map[int(class_id)]), (x[0], y[0]), 
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
					center = (int((x[0] + x[1]) / 2), int((y[0] + y[1]) / 2))
					image = cv2.circle(image, center, 1, (255, 0, 0), 10)
					print(filter_img[center[1]:center[1] + 1, center[0]:center[0] + 1][0][0])
					if filter_img[center[1]:center[1] + 1, center[0]:center[0] + 1][0][0] >= 10 :
						list_of_near_obj.append(label_map[int(class_id)])
						
			with lock:
				tag = ""
				for i in list_of_near_obj:
					tag += i + ", "
				list_of_near_obj = []
				if tag != "":
					tag += " are near!"



						
			end_time = cv2.getTickCount()
			time_take = (end_time - starttime)/freq
			frame_rate_calc = 1/time_take
			cv2.putText(image, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)	

			cv2.imshow("Frame", image)
			cv2.imshow("leftFrame", filter_img)    
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
				
		cv2.destroyAllWindows()
		
				
thread_socket = threading.Thread(target=socket_thread_function)
thread_socket.daemon = True
thread_socket.start()

main_instance = Main()
main_thread = threading.Thread(target=main_instance.main_recording)
main_thread.daemon = True
main_thread.start()			

while True:
	time.sleep(1)
          	

            
            
			
			


