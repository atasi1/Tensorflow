# import the necessary packages
import numpy as np
import argparse
import cv2 as cv

# construct the argument parse and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-i","--image1", help="path to the image file",type=str)
print(ap)
args = ap.parse_args()
print(args)
'''
# load the image
#image = cv.imread(args["image1"])
image = cv.imread('/home/atasi/resolve/finding_shapes_example.png')
#print(image)
#print(image)
# find all the 'black' shapes in the image
lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
#print(lower)
shapeMask = cv.inRange(image,lower, upper)
#shapeMask=cv.Canny(image,lower, upper)


'''gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edged = cv.Canny(gray, 50, 100)
cv.imshow("Original", image)'''

(cnts, _) = cv.findContours(shapeMask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
mask = np.ones(image.shape[:2], dtype="uint8") * 255

for c in cnts:
    # if the contour is bad, draw it on the mask
    cv.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv.imshow("Image", image)

    cv.waitKey(0)
# find the contours in the mask
'''(cnts, _) = cv.findContours(shapeMask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print("I found %d black shapes" % (len(cnts)))
cv.imshow("Mask", shapeMask)'''
