from colorthief import ColorThief
import numpy as np
import cv2

color_thief = ColorThief('patterns/1.jpg')
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
# build a color palette
palette = color_thief.get_palette(color_count=4)

print(dominant_color)
print(palette)

bar = np.zeros((50, 300, 3), dtype="uint8")
startX = 0
for color in zip(palette):
	endX = startX + (0.25 * 300)
	cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
				color.astype("uint8").tolist(), -1)
	startX = endX
plt.imshow(bar)
plt.show()