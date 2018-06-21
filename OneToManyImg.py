import cv2
import numpy as np
import imutils
import os
from PIL import Image
from PIL import ImageFilter
import configparser
import sys

confPath=sys.argv[1]

Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(confPath+"/config.ini")
IMAGE_FILE_PATH=Config.get('tensorflow','IMAGE_FILE_PATH')
IMAGE=Config.get('tensorflow','IMAGE')
IMAGE_TYPE=Config.get('tensorflow','IMAGE_TYPE')

IMAGE_FILE=IMAGE_FILE_PATH+'/'+IMAGE+'.'+IMAGE_TYPE
print(IMAGE_FILE)


img = cv2.imread(IMAGE_FILE)
height, width = img.shape[:2]


def translationRotation(ang1,ang2,diff):
	#This function will help in rotating the image into different angles as entered by the user
	for angle in np.arange(int(ang1), int(ang2), int(diff)):
		rotated = imutils.rotate_bound(img, angle)
		imagefile = IMAGE_FILE_PATH + '/' + IMAGE + '_' + str(angle) + '.' + IMAGE_TYPE

		cv2.imwrite(imagefile, rotated)
		print(angle)

def noisy(noise_typ):
	#This function will introduce noise to an image
	if noise_typ == "gauss":
		row, col, ch = img.shape
		mean = 0
		var = 0.1
		sigma = var ** 0.5
		gauss = np.random.normal(mean, sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = img + gauss
		return noisy
	elif noise_typ == "s&p":
		row, col, ch = img.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(img)
		# Salt mode
		num_salt = np.ceil(amount * img.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
				  for i in img.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
				  for i in img.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(img))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(img * vals) / float(vals)
		return noisy


	elif noise_typ == "speckle":
		row, col, ch = img.shape
		gauss = np.random.randn(row, col, ch)
		gauss = gauss.reshape(row, col, ch)
		noisy = img + img * gauss
		return noisy
def deterioratingImage():
	#This function will resize the image
	basewidth = 300
	img1 = Image.open(IMAGE_FILE)
	wpercent = (basewidth / float(img1.size[0]))
	hsize = int((float(img1.size[1]) * float(wpercent)))
	img = img1.resize((basewidth, hsize), Image.ANTIALIAS)
	imagefile = IMAGE_FILE_PATH + '/' + IMAGE + '_resize' + '.' + IMAGE_TYPE
	img.save(imagefile)

def blurImage():
	#This function will blur the image
	img1 = Image.open(IMAGE_FILE)
	imagefile = IMAGE_FILE_PATH + '/' + IMAGE + '_blur' + '.' + IMAGE_TYPE
	blurred_image = img1.filter(ImageFilter.BLUR)
	blurred_image.save(imagefile)



def main():
	ONE_TO_MANY_IMAGE=input('         Please enter 1 to rotate the image into different angles\n         Please enter 2 to create noisy image\n         Please enter 3 to deteriorate image\n         Please enter 4 to create blurred image')
	if ONE_TO_MANY_IMAGE=='1':
		ang1=input("         enter Starting angle")
		ang2=input("         enter Ending angle")
		diff=input("         enter the difference in the angles")
		translationRotation(ang1,ang2,diff)
	elif ONE_TO_MANY_IMAGE=='2':
		noise_typ = "s&p"
		noise = noisy(noise_typ)
		imagefile = IMAGE_FILE_PATH + '/' + IMAGE + '_noise' + '.' + IMAGE_TYPE
		cv2.imwrite(imagefile, noise)
	elif ONE_TO_MANY_IMAGE=='3':
		deterioratingImage()
	elif ONE_TO_MANY_IMAGE=='4':
		blurImage()
	cv2.destroyAllWindows()







main()


