# import libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
import time
import matplotlib.pyplot as plt
from PIL import Image
#from skimage import io, measure, color, filters, segmentation
import cv2
import argparse
from skimage.transform import pyramid_gaussian
from sklearn.preprocessing import scale

# silhouetteCoeff determination
def silhouetteCoeff(z):
	t0 = time.time()
	max_silhouette = 0
	max_k = 0
	for i in range(4, 11):
		clt = MiniBatchKMeans(n_clusters = i, random_state = 42)
		clt.fit(z)
		silhouette_avg = silhouette_score(z, clt.labels_, sample_size = 500, random_state = 42)
		#print("k: ", i, " silhouette avg: ", silhouette_avg)
		if (silhouette_avg == 1.0):
			max_k = i
			print("Max k: ", max_k)
			break
		elif (silhouette_avg > max_silhouette):
			max_silhouette = silhouette_avg
			max_k = i
	#print("Max silhouette: ", max_silhouette)
	#print("Max k: ", max_k)
	print("Time for silhouette: ", time.time() - t0)
	return int(max_k)

"""# colorQuantization algorithm
def colorQuantize(img):
	z = img.reshape((-1, 3))
	z = np.float32(z)
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
	sse, label, center = cv2.kmeans(z, 16, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

	center = np.uint8(center)
	res = center[label.flatten()]
	res = res.reshape(img.shape)
	print(res.shape)
	kMeans(res)"""

# kMeans algorithm
def kMeans(img):
	t0 = time.time()
	# apply kMeans, fit data, get histogram, get color bar
	org_img = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	z = img.reshape((-1, 3))
	#print(z.shape)

	# image resize just for silhouetteCoeff
	# Crops images to 300x300, but loses accuracy
	# Try pyrDown (downsampling the images)
	"""ysize, xsize, chan = img.shape
	if ysize and xsize > 300:
		xoff = (xsize - 300) // 2
		yoff = (ysize - 300) // 2
		y = img[yoff:-yoff, xoff:-xoff]
	else:
		y = img
	y = y.reshape((-1, 3))
	print(y.shape)"""

	# downnsample images with gaussian smoothing
	if (img.shape[0] > 250 or img.shape[1] > 250):
		for (i, resized) in enumerate(pyramid_gaussian(org_img, downscale=2)):
			if resized.shape[0] < 100 or resized.shape[1] < 100:
				break
			org_img = resized
		#cv2.imshow("Layer {}".format(i + 1), resized)
		#print(org_img.shape)

	org_img = org_img.reshape((-1, 3))
	org_img = scale(org_img)
	#print(org_img.shape)
	# kmeans
	clt = MiniBatchKMeans(n_clusters = silhouetteCoeff(org_img), random_state = 42)
	clt.fit(z)

	hist = centroidHistogram(clt)
	# bar = plotColors(hist, clt.cluster_centers_)
	print("Time including KMeans: ", time.time() - t0)
	#print("unique labels: ", np.unique(np.array(clt.labels_), axis=0))

	# plt.figure(1)
	# plt.axis("off")
	# plt.subplot(211)
	# plt.imshow(img)
	# plt.subplot(212)
	# plt.imshow(bar)
	# plt.show()

def centroidHistogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist

# def plotColors(hist, centroids):
# 	bar = np.zeros((50, 300, 3), dtype = "uint8")
# 	startX = 0
# 	if centroids.shape[0] <= 8:
# 		for (percent, color) in zip(hist, centroids):
# 			endX = startX + (percent * 300)
# 			cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
# 				color.astype("uint8").tolist(), -1)
# 			startX = endX
# 		return bar
# 	else:
# 		for (percent, color) in zip(hist, centroids):
# 			endX = startX + (0.125 * 300)
# 			cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
# 				color.astype("uint8").tolist(), -1)
# 			startX = endX
# 		return bar

# read image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
kMeans(img)