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
import Queue
import threading

q = Queue.Queue()
k_range = [4, 5, 6, 7, 8]
k_sil = np.zeros([5, 1])

for i in k_range:
	q.put(i)

# getK
def getK(z, i):
	clt = MiniBatchKMeans(n_clusters = i, random_state = 42)
	clt.fit(z)
	silhouette_avg = silhouette_score(z, clt.labels_, sample_size = 250, random_state = 42)
	print("k: ", i, " silhouette avg: ", silhouette_avg)
	k_sil[i - 4] = silhouette_avg
	return int(silhouette_avg)

# worker
def worker(queue, z, i):
	queue_full = True
	while queue_full:
		try:
			i = queue.get(False)
			data = getK(z, i)
			print(data)
		except Queue.Empty:
			queue_full = False

# silhouetteCoeff determination
def silhouetteCoeff(z):
	t0 = time.time()
	#k_sil = np.zeros([5, 1])

	thread_count = 5
	for i in range(thread_count):
		t = threading.Thread(target=worker, args = (q, z, i + 4))
		t.start()
	#print(k_sil)
	#k_sil[i - 4, 0] = i
	#k_sil[i - 4, 1] = silhouette_avg
	max_k = np.argmax(k_sil)
	print(max_k)
	#max_k = k_sil[max_k]
	#print("silhouette time: ", time.time() - t0)
	return int(max_k+4)

# colorQuantization algorithm
# def colorQuantize(img):
# 	z = img.reshape((-1, 3))
# 	z = np.float32(z)
	
# 	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
# 	sse, label, center = cv2.kmeans(z, 16, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# 	center = np.uint8(center)
# 	res = center[label.flatten()]
# 	res = res.reshape(img.shape)
# 	print(res.shape)
# 	kMeans(res)

# kMeans algorithm
def kMeans(img):
	t0 = time.time()
	# apply kMeans, fit data, get histogram, get color bar
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	z = img.reshape((-1, 3))
	print(z.shape)

	# kmeans
	clt = MiniBatchKMeans(n_clusters = silhouetteCoeff(z), random_state = 42)
	clt.fit(z)

	hist = centroidHistogram(clt)
	bar = plotColors(hist, clt.cluster_centers_)
	print("Time: ", time.time() - t0)
	#print("unique labels: ", np.unique(np.array(clt.labels_), axis=0))

	plt.figure(1)
	plt.axis("off")
	plt.subplot(211)
	plt.imshow(img)
	plt.subplot(212)
	plt.imshow(bar)
	plt.show()

def centroidHistogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist

def plotColors(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	return bar

# read image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
kMeans(img)