import cv2
import numpy as np
import time


def Lum(color):
	return 0.3*color[0] + 0.59*color[1] + 0.11*color[2]

def SetLum(color, l):
	d = l - Lum(color)
	color[0] = color[0]+d
	color[1] = color[1] + d
	color[2] = color[2] + d
	return ClipColor(color)

def ClipColor(color):
	l = Lum(color)
	n = min([color[0], color[1], color[2]])
	x = max([color[0], color[1], color[2]])

	if(n<0):
		color[0] = l + (((color[0]-l)*l)/(l-n))
		color[1] = l + (((color[1]-l)*l)/(l-n))
		color[2] = l + (((color[2]-l)*l)/(l-n))
	if(x>1):
		color[0] = l + (((color[0]-l)*(1-l))/(x-l))
		color[1] = l + (((color[1]-l)*(1-l))/(x-l))
		color[2] = l + (((color[2]-l)*(1-l))/(x-l))
	return color

def SetSaturation(color):
	min([color[0], color[1], color[2]]) - min([color[0], color[1], color[2]])

def min(array):
	min = 10000
	for i in range(len(array)):
		if(array[i] < min):
			min = array[i]
	return min

def max(array):
	max = -10000
	for i in range(len(array)):
		if(array[i] > max):
			max = array[i]
	return max


img = cv2.imread('5/5.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

img2 = cv2.imread('purple2.jpg')
print(img2[0,0])
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
h1, s1, v1 = cv2.split(hsv)


height, width, channels = img.shape
v2 = v


start = time.clock()
h = 0
while h in range(int(height)):
	w = 0
	while w in range(int(width)):
		color = img[h, w]
		lum = Lum(SetLum([187,57,252], Lum(color)))
		v2[h][w] = lum
		if(w < 381):
			v2[h][w+1] = lum 
			v2[h][w+2] = lum 
		w = w+3
	h = h+1
elapsed = time.clock()
print(elapsed - start)

final_hsv = cv2.merge((h1, s1, v2))
img_eff = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('img_eff', img_eff)


start = time.clock()
for h in range(int(height)):
	for w in range(int(width)):
		white = np.array([255,255,255])
		if((img[h,w]==white).all()):
			continue
		color = img[h, w]
		# print(SetLum([153,50,204], Lum(color)))
		v1[h][w] = Lum(SetLum([187,57,252], Lum(color)))
elapsed = time.clock()
print(elapsed - start)

final_hsv = cv2.merge((h1, s1, v1))
img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('img_full', img2)


img = cv2.imread('5.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

img2 = cv2.imread('purple2.jpg')
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
h1, s1, v1 = cv2.split(hsv)

final_hsv = cv2.merge((h1,s,v))
img3 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('img_none', img3)





cv2.waitKey(0)