import configparser
import os
from PIL import Image
import cv2
import sys

confPath=sys.argv[1]


Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(confPath+"/config.ini")
IMAGE_FILE_PATH=Config.get('tensorflow','IMAGE_FILE_PATH')

def getImageHeightWidth():
    height=[]
    width=[]
    for imgFile in os.listdir(IMAGE_FILE_PATH):
        print(imgFile)
        img = cv2.imread(IMAGE_FILE_PATH+'/'+imgFile)
        print(img)
        h, w = img.shape[:2]
        height.append(h)
        width.append(w)
    for h in height:
        print(h)
    height.sort()
    width.sort()
    maxHeight=height[len(height)-1]
    minHeight=height[0]
    maxWidth=width[len(width)-1]
    minWidth=width[0]
    print('max height = ',maxHeight,'\nmin height = ',minHeight,'\nmax width = ',maxWidth,'\nmin width = ',minWidth)



getImageHeightWidth()