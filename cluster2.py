import numpy as np
import cv2
import math
import colorsys
from sklearn.cluster import MiniBatchKMeans
import tkFileDialog as filedialog
from Tkinter import *
from tkColorChooser import askcolor
from PIL import ImageTk,Image
import time

# read image, resize to 400x400 and print h, w, c
orgImg = cv2.imread('patterns/1.jpg')
(h, w, c) = orgImg.shape
print h, w, c
#orgImg = cv2.pyrDown(orgImg)
orgImg = cv2.resize(orgImg, (400, 400))
(h, w, c) = orgImg.shape
print h, w, c

orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGB)
orgColors = []
currentColors = []
changeLog = []
changeLogPointers = []
coordinates = []
canvas = []
imageOnCanvas = []
rectangle = []

def getColor(i, labels, center, pixel, indices):
    color = askcolor()
    if color[0] is None:
    	return
    colorArray=[]
    color = color[0]
    color = np.array(color)
    currentColors[i] = color
    changeLog[i].append(currentColors[i])
    changeColor(i, indices, pixel, labels, color)

def changeColor(i, indices, pixel, labels, color):
	global imageOnCanvas
	global canvas
	global img
	img = np.zeros((80,80,3), np.uint8)
	img[:,:] = color
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	labels[i].configure(image = img)
	labels[i].image = img

	img = cv2.imread('curr_img.jpg')
	height, width, channel = img.shape

	for j in range(len(coordinates[indices[i]])):
		img[coordinates[indices[i]][j][1],coordinates[indices[i]][j][0]] = color

	if(i != (len(indices)-1)):
		img_old = cv2.imread('curr_img.jpg')
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)
		hsv2 = cv2.cvtColor(img_old, cv2.COLOR_BGR2HSV)
		h1, s1, v1 = cv2.split(hsv2)
		final_hsv = cv2.merge((h, s, v1))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

	cv2.imwrite('curr_img.jpg', img)

	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)

	canvas.delete(imageOnCanvas)
	canvas.create_image(0, 0, image=img, anchor="nw")

def undoFunction(i, indices, pixel, labels):
	if(changeLogPointers[i] == -3):
		changeLogPointers[i] = len(changeLog[i]) - 1
	if(changeLogPointers[i] == 0):
		return
	if(changeLogPointers[i] == -1):
		changeLogPointers[i] = len(changeLog[i]) - 2

	changeLogPointers[i] = changeLogPointers[i] - 1

	color = changeLog[i][changeLogPointers[i]]
	changeColor(i, indices, pixel, labels, color)

def redoFunction(i, indices, pixel, labels):
	global changeLogPointers
	if(changeLogPointers[i] == -3):
		return
	if(changeLogPointers[i] == -2):
		changeLogPointers[i] = 1
	if(changeLogPointers[i] == len(changeLog[i]) - 1):
		return

	changeLogPointers[i] = changeLogPointers[i] + 1

	print('i: ' + str(changeLogPointers[i]))
	color = changeLog[i][changeLogPointers[i]]
	changeColor(i, indices, pixel, labels, color)

def resetFunction(indices, pixel, labels):
	global changeLog
	global canvas
	global img
	numColors = len(changeLog)
	changeLog = [[] for _ in range(numColors)]
	del changeLogPointers[:]
	for i in range(numColors):
		changeLog[i].append(orgColors[i])
		currentColors.append(orgColors[i])
		changeLogPointers.append(-3)

	for i in range(numColors):
		img = np.zeros((80,80,3), np.uint8)
		img[:,:] = orgColors[i]
		img = Image.fromarray(img)
		img = ImageTk.PhotoImage(img)
		labels[i].configure(image = img)
		labels[i].image = img

	img = orgImg
	cv2.imwrite('curr_img.jpg', img)
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	canvas.delete(imageOnCanvas)
	canvas.create_image(0,0,image=img,anchor="nw")

def saveFunction():
	img = cv2.imread('curr_img.jpg')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	filename = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=(("jpg", "*.jpg"),("png", "*.png"),("All Files", "*.*") ))
	if(filename != ""):
		cv2.imwrite(filename, img)

def printcoords(event, pixel, indices, labels):
	global rectangle
	canvas.delete(rectangle)
	for i in range(len(labels)):
		labels[i].configure(state='normal')
	img = cv2.imread('curr_img.jpg')
	height, width, channels = img.shape
	color = img[event.y, event.x]
	rectangle = canvas.create_rectangle(event.x, event.y, event.x+30, event.y+30, fill='#%02x%02x%02x' % (color[0], color[1], color[2]), outline='black')
	index = indices.index(pixel[width*event.y + event.x][0])
	for i in range(len(labels)):
		if(i != index):
			labels[i].configure(state='disabled')

def generateUI(center, pixel, indices):
	height, width, channels = orgImg.shape
	root = Tk()
	topFrame = Frame(root)
	global canvas
	canvas = Canvas(topFrame, width=width, height=height)
	canvas.grid(row=0, column=0, sticky=N+S+E+W)
	topFrame.grid(row=1, column=0)
	bottomFrame = Frame(root)
	bottomFrame.grid(row=2, column=0)

	rightFrame = Frame(root)
	rightFrame.grid(row=0, column=0)

	labels = []
	buttons = []
	images = []

	global changeLog
	changeLog = [[] for _ in range(len(center))]
	for i in range(len(changeLog)):
		changeLogPointers.append(-3)

	for i in range(len(center)):
		img = np.zeros((80,80,3), np.uint8)
		img[:,:] = center[i]
		img = Image.fromarray(img)
		img = ImageTk.PhotoImage(img)
		images.append(img)

		orgColors.append(center[i])
		changeLog[i].append(center[i])
		currentColors.append(center[i])

		label = Label(bottomFrame, image=images[i])
		labels.append(label)
		label.grid(row=0, column=i)
		button = Button(bottomFrame, text= "Change Color", command=lambda iteration=i :getColor(iteration, labels, center, pixel, indices))
		buttons.append(button)
		button.grid(row=1, column=i)

		undoFrame = Frame(bottomFrame)
		undoFrame.grid(row=2, column=i)

		undo = Button(undoFrame, text="Undo", command=lambda iteration=i :undoFunction(iteration, indices, pixel, labels))
		undo.grid(row=2, column=0)

		redo = Button(undoFrame, text="Redo", command=lambda iteration=i :redoFunction(iteration, indices, pixel, labels))
		redo.grid(row=2, column=1)

	save = Button(rightFrame, text="Save", command=lambda:saveFunction())
	save.grid(row=0, column=0)

	reset = Button(rightFrame, text="Reset", command=lambda:resetFunction(indices, pixel, labels))
	reset.grid(row=0, column=3)
	
	img = cv2.imread('curr_img.jpg')
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	labels.append(canvas)
	global imageOnCanvas
	imageOnCanvas = canvas.create_image(0,0,image=img,anchor="nw")
	canvas.bind("<Button-1>",lambda event: printcoords(event, pixel, indices, labels))

	root.mainloop()

def generateInput():
	#pil_im = Image.fromarray(orgImg)
	#pil_im.show()

	inputText = Tk()
	Label(inputText, text="Number of Threads").grid(row=0, column=0)
	e1 = Entry(inputText)
	e1.grid(row=0,column=1)
	
	Button(inputText, text="Ok", command=lambda:kmeans(e1.get(), inputText)).grid(row=1,column=0)
	mainloop()

def step (item):
	r = item[0]
	g = item[1]
	b = item[2]

	repetitions = 1
	lum = math.sqrt( .241 * r + .691 * g + .068 * b )

	h, s, v = colorsys.rgb_to_hsv(r,g,b)

	h2 = int(h * repetitions)
	lum2 = int(lum * repetitions)
	v2 = int(v * repetitions)

	return (h2, lum, v2)

def kmeans(K, inputText):
	inputText.destroy()
	global orgImg

	t0 = time.time()
	# color quantization
	"""orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2LAB)
	orgImg = orgImg.reshape((orgImg.shape[0] * orgImg.shape[1], 3))
	clt = MiniBatchKMeans(n_clusters = 8)
	labels = clt.fit_predict(orgImg)
	quant = clt.cluster_centers_.astype("uint8")[labels]

	quant = quant.reshape((h, w, 3))
	orgImg = orgImg.reshape((h, w, 3))

	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
	orgImg = cv2.cvtColor(orgImg, cv2.COLOR_LAB2RGB)

	cv2.imshow("orgImg", np.hstack([orgImg, quant]))
	#cv2.waitkey(0)
	#orgImg = np.hstack([orgImg, quant])
	quant = cv2.cvtColor(quant, cv2.COLOR_RGB2BGR)
	orgImg = quant
	print(time.time() - t0)"""
	
	# kmeans algorithm
	img = orgImg
	height, width, channels = img.shape
	cv2.imwrite('curr_img.jpg', img)

	z = img.reshape((-1, 3))
	z = np.float32(z)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, pixel, center = cv2.kmeans(z, int(K), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	global coordinates
	coordinates = [[] for _ in range(len(center))]
	for i in range(len(pixel)):
		coordinates[pixel[i][0]].append([i%width, int(i/width)])

	center2 = sorted(center, key=step, reverse=True)
	indices = []
	for i in range(len(center2)):
		for j in range(len(center)):
			if list(center[j]) == list(center2[i]):
				indices.append(j)
	print(center2)
	generateUI(center2, pixel, indices)

generateInput()