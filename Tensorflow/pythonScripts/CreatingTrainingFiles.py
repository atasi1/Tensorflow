import configparser
import os
import tensorflow as tf
from lxml import etree
from models.research.object_detection.utils import dataset_util
from models.research.object_detection.utils import label_map_util


import sys
progPath=sys.argv[1]

Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(progPath+"/config.ini")
ANNOTATION_FILE_PATH=Config.get('tensorflow','ANNOTATION_FILE_PATH')
IMAGE_FILE_PATH=Config.get('tensorflow','IMAGE_FILE_PATH')
TRAINING_FILE_PATH=Config.get('tensorflow','TRAINING_FILE_PATH')
PBTXTFILE_PATH=Config.get('tensorflow','PBTXTFILE_PATH')
PBTXTFILE_NAME=Config.get('tensorflow','PBTXTFILE_NAME')
PBTXTFILE=PBTXTFILE_PATH+'/'+PBTXTFILE_NAME+'.pbtxt'

flags = tf.app.flags
flags.DEFINE_string('label_map_path', PBTXTFILE,
                    'Path to label map proto')
FLAGS = flags.FLAGS


def getListOfImgFiles():
    #It will contain the list of image files
    LIST_OF_IMAGE_FILES=[]
    for xmlFile in os.listdir(IMAGE_FILE_PATH):
        LIST_OF_IMAGE_FILES.append(xmlFile)
    for im in LIST_OF_IMAGE_FILES:
        print('actual file : '+im)
    return LIST_OF_IMAGE_FILES

def creatingTrainingFile(LIST_OF_IMAGE_FILES,imageFilelist,stringToWrite):
    #This file will create the string to be written into trainin file
    #It will associate a 1 if an image file contains a particular object
    #Else it will associate a -1 if the image file does not contain the object
    for imagefile1 in LIST_OF_IMAGE_FILES:
        flag=0
        for imageFile2 in imageFilelist:

            if imagefile1 == imageFile2:
                stringToWrite = stringToWrite + imagefile1+ ' ' + '1\n'
                flag = flag+1

        if flag==0:
            stringToWrite = stringToWrite + imagefile1 + ' ' + '-1\n'

    return stringToWrite

def main():
    #contains the class names from pbtxt file
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)


    for l in label_map_dict:
        print("label : "+l)

        stringToWrite = ''
        file_list = [] #contains all the files from the annotation file

        c=0
        #taking all the files from the annotation directory
        for xmlFile in os.listdir(ANNOTATION_FILE_PATH):
            with tf.gfile.GFile(ANNOTATION_FILE_PATH + '/' + xmlFile, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)

            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation'] #Getting data from annotation tag
            imageFile = data['filename'] #Getting data for filename
            objects = data['object'] #Getting datails of an object from xml file
            objectname=''
            allObjects=[]
            for object in objects:
                objectname = object['name'].strip()
                allObjects.append(objectname)

            if not allObjects.__contains__(l):
                continue
            else:
                #Adding the files from xml files into a list
                if not file_list.__contains__(imageFile):
                    file_list.append(imageFile)


        LIST_OF_IMAGE_FILES = getListOfImgFiles()
        stringToWrite = creatingTrainingFile(LIST_OF_IMAGE_FILES, file_list, stringToWrite)
        TRAINING_FILE = TRAINING_FILE_PATH + '/' + l + '_train.txt'
        trainingfileopen = open(TRAINING_FILE, 'w')
        trainingfileopen.write(stringToWrite)
        trainingfileopen.close()
        print('created')



if __name__ == '__main__':
    main()

