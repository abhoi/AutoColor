import numpy as np
import cv2
import math
import colorsys
import tkinter.filedialog
from tkinter import *
from tkinter.colorchooser import *
from PIL import ImageTk,Image

org_img = cv2.imread('22.07.17_1.jpg')
org_img = cv2.cvtColor(org_img,cv2.COLOR_BGR2RGB)
orgColors = [] #save the original colors for reset
currentColors = [] #the current color so you know what to append to the changelog
changelog = [] #stores colors for undo and redo
changelogPointers = [] #pointer for changelog so you know your position of undo/redo
coordinates = [] #splits the kmeans nearest clusters in array of length k so you don't have to loop over the entire image when changing the colors
canvas = []
image_on_canvas = []
rectangle = []

def getminDistance(color, colors):
	minimum = 100000000
	for i in range(len(colors)):
		distance = math.pow((colors[i][0] - color[0]),2) + math.pow((colors[i][1] - color[1]),2) + math.pow((colors[i][2] - color[2]),2)
		if distance < minimum:
			minimum = distance
			changed_color = colors[i]
	return changed_color

def getColor(i, labels, center, pixel, indices):
    color = askcolor()
    if color[0] is None:
    	return
    color_array=[]
    color = color[0]
    color = np.array(color)
    currentColors[i] = color
    changelog[i].append(currentColors[i])
    changeColor(i, indices, pixel, labels, color)


def changeColor(i, indices, pixel, labels, color):
	global image_on_canvas
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

	# for h in range(height):
	# 	for w in range(width):
	# 		if indices[pixel[(h*width)+w][0]] == i:
	# 			img[h,w] = color

	# white = np.array([255,255,255])
	# black = np.array([0,0,0])
	# img_old = cv2.imread('curr_img.jpg')

	# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# h, s, v = cv2.split(hsv)

	# hsv2 = cv2.cvtColor(img_old, cv2.COLOR_BGR2HSV)
	# h1, s1, v1 = cv2.split(hsv2)

	# final_hsv = cv2.merge((h, s, v1))
	# img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

	cv2.imwrite('curr_img.jpg', img)


	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	
	# labels[len(labels)-1].configure(image = img)
	# labels[len(labels)-1].image = img

	canvas.delete(image_on_canvas)
	canvas.create_image(0,0,image=img,anchor="nw")

def undoFunction(i, indices, pixel, labels):
	if(changelogPointers[i] == -3):
		changelogPointers[i] = len(changelog[i]) - 1
	if(changelogPointers[i] == 0):
		return
	if(changelogPointers[i] == -1):
		changelogPointers[i] = len(changelog[i]) - 2

	changelogPointers[i] = changelogPointers[i] - 1

	color = changelog[i][changelogPointers[i]]
	changeColor(i, indices, pixel, labels, color)

	


def redoFunction(i, indices, pixel, labels):
	if(changelogPointers[i] == -3):
		return
	if(changelogPointers[i] == -2):
		changelogPointers[i] = 1
	if(changelogPointers[i] == len(changelog[i]) - 1):
		return

	changelogPointers[i] = changelogPointers[i] + 1

	print('i: ' + str(changelogPointers[i]))
	color = changelog[i][changelogPointers[i]]
	changeColor(i, indices, pixel, labels, color)


def resetFunction(indices, pixel, labels):
	global changelog
	global canvas
	global img
	numColors = len(changelog)
	changelog = [[] for _ in range(numColors)]
	del changelogPointers[:]
	for i in range(numColors):
		changelog[i].append(orgColors[i])
		currentColors.append(orgColors[i])
		changelogPointers.append(-3)

	for i in range(numColors):
		img = np.zeros((80,80,3), np.uint8)
		img[:,:] = orgColors[i]
		img = Image.fromarray(img)
		img = ImageTk.PhotoImage(img)
		labels[i].configure(image = img)
		labels[i].image = img

	img = org_img
	cv2.imwrite('curr_img.jpg', img)
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	canvas.delete(image_on_canvas)
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
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	height, width, channels = img.shape
	color = img[event.y, event.x]
	rectangle = canvas.create_rectangle(event.x, event.y, event.x+30, event.y+30, fill='#%02x%02x%02x' % (color[0], color[1], color[2]), outline='black')
	# print('#%02x%02x%02x' % (img[event.x,event.y][0], img[event.x,event.y][1], img[event.x,event.y][2]))
	index = indices.index(pixel[width*event.y + event.x][0])
	for i in range(len(labels)):
		if(i != index):
			labels[i].configure(state='disabled')

def generateUI(center, pixel, indices):
	height, width, channels = org_img.shape
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

	global changelog
	changelog = [[] for _ in range(len(center))]
	for i in range(len(changelog)):
		changelogPointers.append(-3)


	for i in range(len(center)):
		img = np.zeros((80,80,3), np.uint8)
		img[:,:] = center[i]
		img = Image.fromarray(img)
		img = ImageTk.PhotoImage(img)
		images.append(img)

		orgColors.append(center[i])
		changelog[i].append(center[i])
		currentColors.append(center[i])

		label = Label(bottomFrame, image=images[i])
		# label.configure(relief=FLAT)
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
	# label = Label(topFrame, image=img)
	# labels.append(label)
	# label.grid(row=0, column=0)
	labels.append(canvas)
	global image_on_canvas
	image_on_canvas = canvas.create_image(0,0,image=img,anchor="nw")
	canvas.bind("<Button-1>",lambda event: printcoords(event, pixel, indices, labels))

	root.mainloop()

def generateInput():
	pil_im = Image.fromarray(org_img)
	pil_im.show()


	input_text = Tk()
	Label(input_text, text="Number of Threads").grid(row=0, column=0)
	e1 = Entry(input_text)
	e1.grid(row=0,column=1)
	
	Button(input_text, text="Ok", command=lambda:kmeans(e1.get(), input_text)).grid(row=1,column=0)
	mainloop()

def getKey(item):
	return item[0]

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

def kmeans(K, input_text):
	input_text.destroy()
	img = org_img
	height, width, channels = img.shape
	cv2.imwrite('curr_img.jpg', img)
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,pixel,center=cv2.kmeans(Z,int(K),None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


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

	generateUI(center2, pixel, indices)





generateInput()
