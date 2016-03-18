import cv2
import numpy as np

def draw_pyramid(img, points):
	#TODO check if points look like a house (distances, line presence, shape...)
	#TODO figure out which points are which
	#TODO calculate where to put the "top" of the pyramid
	# - by fixing perspective? - http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspectiv
	# - - needs 4 points
	# - - sample code for perspective transformation:
	'''
	points_img = [mc[tl],mc[tr],mc[br],mc[bl]]
	points_new = [[0,0],[w_new,0],[w_new,h_new],[0,h_new]]
	p_img = np.array(points_img, np.float32)
	p_new = np.array(points_new, np.float32)
	persp = cv2.getPerspectiveTransform(p_img, p_new)
	trans = cv2.warpPerspective(img_copy, persp, (w_new, h_new))
	'''
	#TODO draw lines
	pass

def house(img):
	width, height = img.shape[:2]
	img_copy = img.copy()


	img_sharpen = cv2.GaussianBlur(img, (0,0),3)
	img_ = cv2.addWeighted(img, 1.5, img_sharpen, -0.5, 0)
	img_gs = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
	res, thresh = cv2.threshold(img_gs,150,255,0)#TODO play with threshold value

	edges = cv2.Canny(thresh, 50, 200, 5)
	image, contours, hierarchy_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if hierarchy_ is None:
		return None
	hierarchy = hierarchy_[0]
	cv2.imshow('edges', edges)

	for (i, cnt) in enumerate(contours):
		if hierarchy[i][3] != -1:
			continue
		cv2.drawContours(frame, [contours[i]], 0,(0,255,0), 3)

		approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
		#print len(approx)
		if len(approx)==5:
			#print "pentagon"
			#cv2.drawContours(frame,[cnt],0,255,-1)
			cv2.drawContours(img,[approx],0,(0,0,255),3)
			draw_pyramid(img, approx)
			#print approx
		else:
			pass
			#cv2.drawContours(frame,[approx],0,(0,0,255),-1)
	cv2.imshow('house', img)
	

if 'cap' in locals():
	cap.release()
cap = cv2.VideoCapture(0)
while(True):
	ret, frame = cap.read()
	house(frame)
	if cv2.waitKey(50) & 0xFF == ord('q'):
		break
cap.release()
