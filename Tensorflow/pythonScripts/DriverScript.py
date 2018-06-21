import os
import sys
import configparser
import time

configPath=sys.argv[1]

Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(configPath+"/config.ini")
PATH_TO_PYTHON_SCRIPTS=Config.get('tensorflow','PATH_TO_PYTHON_SCRIPTS')
ANNOTATION_FILE_PATH=Config.get('tensorflow','ANNOTATION_FILE_PATH')
IMAGE_FILE_PATH=Config.get('tensorflow','IMAGE_FILE_PATH')


print("                                                            WELCOME TO THE POC OF RESOLVE DIGITAL")
print("                                                WE ARE GOING TO CREATE THE TFR DATASET FOR THE PILL IMAGES")
print("         please give the following inputs\n")

print("         Please check your image files in the following path : "+IMAGE_FILE_PATH)
print("         Please check your annotation files in the following path : "+ANNOTATION_FILE_PATH)



configFile=configPath+'/config.ini'
print(configFile)
#file=open(configFile,'w')
def putConfiguration(conf):

    if conf[0]=='ANNOTATION_FILE_PATH':
        conf[1]=ANNOTATION_FILE_PATH
    elif conf[0]=='IMAGE_FILE_PATH':
        conf[1]=IMAGE_FILE_PATH




def main():


    try:
        print('         HERE CREATING THE PBTXT FILE : ')
        time.sleep(2)
        os.system('python3 ' + PATH_TO_PYTHON_SCRIPTS + '/' + 'CreatingPbtxtFile.py' + ' ' + configPath)
    except:
        print('         exception in creating pbtxt file, kindly check the following path and rerun the program:')
        PBTXTFILE_PATH = Config.get('tensorflow', 'PBTXTFILE_PATH')
        PBTXTFILE_NAME = Config.get('tensorflow', 'PBTXTFILE_NAME')
        PBTXTFILE = PBTXTFILE_PATH + '/' + PBTXTFILE_NAME
        print('         ' + PBTXTFILE)
        time.sleep(3)
        sys.exit(99)

    try:
        print('         HERE CREATING THE TRAINING FILE : ')
        time.sleep(2)
        os.system('python3 ' + PATH_TO_PYTHON_SCRIPTS + '/' + 'CreatingTrainingFiles.py' + ' ' + configPath)
    except:
        print('         exception in creating training file, kindly check the following path and rerun the program:')
        TRAINING_FILE_PATH = Config.get('tensorflow', 'TRAINING_FILE_PATH')
        TRAINING_FILE = TRAINING_FILE_PATH + '/'
        print('         ' + TRAINING_FILE)
        time.sleep(3)
        sys.exit(99)

    try:
        print('         HERE CREATING THE IMAGE TO OBJECT MAPPING FILE : ')
        time.sleep(2)
        os.system('python3 ' + PATH_TO_PYTHON_SCRIPTS + '/' + 'ImgNameToObjName.py' + ' ' + configPath)
    except:
        print(
            '         exception in creating the image name to object name mapping file, kindly check the following path and rerun the program:')
        IMG_TO_OBJECT_FILE_PATH = Config.get('tensorflow', 'IMG_TO_OBJECT_FILE_PATH')
        IMAGE_TO_CLASS_MAPPING_FILE_NAME = Config.get('tensorflow', 'IMAGE_TO_CLASS_MAPPING_FILE_NAME')
        IMAGE_TO_CLASS_MAPPING_FILE = IMG_TO_OBJECT_FILE_PATH + '/' + IMAGE_TO_CLASS_MAPPING_FILE_NAME
        print('         ' + IMAGE_TO_CLASS_MAPPING_FILE)
        time.sleep(3)
        sys.exit(99)

    try:
        print('         HERE CREATING THE TFR DATASET : ')
        time.sleep(2)
        os.system('python3 ' + PATH_TO_PYTHON_SCRIPTS + '/' + 'CreateTfrDataset.py' + ' ' + configPath)
    except:
        print('         exception in creating the TFR dataset')
        time.sleep(3)
        sys.exit(99)

    OUTPUT_DIRECTORY = Config.get('tensorflow', 'OUTPUT_DIRECTORY')
    OUTPUT_FILE_NAME_TRAIN = Config.get('tensorflow', 'OUTPUT_FILE_NAME_TRAIN')
    OUTPUT_FILE_NAME_EVAL = Config.get('tensorflow', 'OUTPUT_FILE_NAME_EVAL')
    print('         kindly check the training and evaluation file in the following path : \n\n')
    print('         ' + OUTPUT_DIRECTORY + '/' + OUTPUT_FILE_NAME_TRAIN + '\n\n')
    print('         ' + OUTPUT_DIRECTORY + '/' + OUTPUT_FILE_NAME_EVAL)


main()
