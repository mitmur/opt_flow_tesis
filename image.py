import cv2
import os
import re

# Read the image folder
# Video name to be created
image_folder = '/Test/Test001_gt'

"""
images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
"""

def captura(folder):
	filenames = glob.glob(folder+"/*.tif")
	filenames.sort()
	images = [cv2.imread(img) for img in filenames]

	for img in images:
	    print img



if __name__ == '__main__':
	main()
	#captura(image_folder)