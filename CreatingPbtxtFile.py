import configparser
import tensorflow as tf
import os
from models.research.object_detection.utils import dataset_util
from lxml import etree
from pathlib import Path
import sys
progPath=sys.argv[1]

Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(progPath+"/config.ini")
PBTXTFILE_PATH=Config.get('tensorflow','PBTXTFILE_PATH')
PBTXTFILE_NAME=Config.get('tensorflow','PBTXTFILE_NAME')
PBTXT_FILE=PBTXTFILE_PATH+'/'+PBTXTFILE_NAME+'.pbtxt'
ANNOTATION_FILE_PATH=Config.get('tensorflow','ANNOTATION_FILE_PATH')


def generateObjId():
    idcount=0
    with open(PBTXT_FILE) as pb:
        idcount = pb.read().count('id')
        #print('no of ids : ', idcount)
    return idcount+1

def writingToPbtxt(objectname):
    #This function will write to the pbtxt file,
    #The file will contain object name, along with a unique id

    pbtxtfileopen = open(PBTXT_FILE, 'a')
    objectId = generateObjId() #Generating a unique id

    #creating the item that will be written to the pbtxt file
    stringToWrite='item\n{\nid: '+str(objectId)+'\nname: '+'\''+objectname+'\'\n}\n'
    pbtxtfileopen.write(stringToWrite)
    pbtxtfileopen.close()


def main():
    LIST_OF_OBJECTS = []
    count = 1
    pbtxtFilepath=Path(PBTXT_FILE)

    #if pbtxt file does not exist then create the file, or if it exists then remove it
    if not pbtxtFilepath.exists():
        open(PBTXT_FILE,'w').close()
    else: os.system('rm '+PBTXT_FILE)

    for xmlFile in os.listdir(ANNOTATION_FILE_PATH):
        #print(xmlFile)

        #rotating through all the annotation files
        with tf.gfile.GFile(ANNOTATION_FILE_PATH + '/' + xmlFile, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation'] #reading from annotation tag
        objects = data['object'] #reading the attributes of the object
        filename=data['path'] #reading the path of the image file
        for object in objects:
            objectname = object['name'].strip() #getting the object name
            #print(objectname)
            if LIST_OF_OBJECTS.__contains__(objectname):
                #if a particular object has already occured then move to the next object
                continue
            else:
                #add the object to the object list
                LIST_OF_OBJECTS.append(objectname)
                #print(filename)
                writingToPbtxt(objectname) #write the object to the pbtxt file

        #print(count)
        count = count + 1
    print('The Number of Classes in the Dataset is : ',len(LIST_OF_OBJECTS))






if __name__ == '__main__':
    main()

