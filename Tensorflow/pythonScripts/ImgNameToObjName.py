import configparser
import os
from models.research.object_detection.utils import dataset_util
from lxml import etree
import tensorflow as tf

import sys
progPath=sys.argv[1]

Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(progPath+"/config.ini")
IMG_TO_OBJECT_FILE_PATH=Config.get('tensorflow','IMG_TO_OBJECT_FILE_PATH')
fileopen=open(IMG_TO_OBJECT_FILE_PATH+'/imgnameToclassname.txt','w')
ANNOTATION_FILE_PATH=Config.get('tensorflow','ANNOTATION_FILE_PATH')

for xmlFile in os.listdir(ANNOTATION_FILE_PATH):
    print('xml file : '+xmlFile)

    with tf.gfile.GFile(ANNOTATION_FILE_PATH + '/' + xmlFile, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    objects = data['object']
    objectname=""
    obj=[]
    for object in objects:
        if obj.__contains__(object['name']):

            continue


        obj.append(object['name'])
        objectname = objectname+" "+object['name']
    imageFile = data['filename']
    fileopen.write(imageFile+'='+objectname+'\n')
fileopen.close()


