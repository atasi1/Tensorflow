import hashlib
import io
import logging
import os
import random
from lxml import etree
import PIL.Image
import tensorflow as tf
import sys
sys.path.insert(0, os.getcwd())
from models.research.object_detection.utils import dataset_util
from models.research.object_detection.utils import label_map_util

import configparser

progPath=sys.argv[1]

Config=configparser.ConfigParser()
Config._interpolation=configparser.ExtendedInterpolation()
Config.read(progPath+"/config.ini")
DATA_DIRECTORY=Config.get('tensorflow','DATA_DIRECTORY')
OUTPUT_DIRECTORY=Config.get('tensorflow','OUTPUT_DIRECTORY')
PBTXTFILE_PATH=Config.get('tensorflow','PBTXTFILE_PATH')
PBTXTFILE_NAME=Config.get('tensorflow','PBTXTFILE_NAME')
PBTXTFILE=PBTXTFILE_PATH+'/'+PBTXTFILE_NAME+'.pbtxt'
IMG_TO_OBJECT_FILE_PATH=Config.get('tensorflow','IMG_TO_OBJECT_FILE_PATH')
IMAGE_TO_CLASS_MAPPING_FILE_NAME=Config.get('tensorflow','IMAGE_TO_CLASS_MAPPING_FILE_NAME')
IMAGE_TO_CLASS_MAPPING_FILE=IMG_TO_OBJECT_FILE_PATH+'/'+IMAGE_TO_CLASS_MAPPING_FILE_NAME

flags = tf.app.flags
flags.DEFINE_string('data_dir', DATA_DIRECTORY+'/', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', OUTPUT_DIRECTORY+'/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', PBTXTFILE,
                    'Path to label map proto')


FLAGS = flags.FLAGS

def get_class_name_from_filename(file_name):
    #This function will get the class name by a filename

    objfile=open(IMAGE_TO_CLASS_MAPPING_FILE,'r')
    imgfile_obj = {}
    for line in objfile:
        k, v = line.strip().split('=')
        imgfile_obj[k.strip()] = v.strip()

    objfile.close()
    return imgfile_obj[file_name]


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,value,
                       ignore_difficult_instances=False):
    img_path = os.path.join(image_subdirectory, data['filename'])
    #This function will crete the tf example set
    with open(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])


    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue
        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = get_class_name_from_filename(data['filename'])

        for each_class_name in class_name.split(" "):

            classes_text.append(each_class_name.encode('utf8'))
            classes.append(label_map_dict[each_class_name])


        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(writer,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,value):
    #This function will create the TFR Record for both training and evaluation and will write it to the file
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = os.path.join(annotations_dir, example.split('.')[0] + '.xml')
        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, label_map_dict, image_dir,value)

        writer.write(tf_example.SerializeToString())


def main():
    try:
        data_dir = FLAGS.data_dir
        label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

        logging.info('Reading from Pet dataset.')
        image_dir = Config.get('tensorflow', 'IMAGE_FILE_PATH')
        annotations_dir = Config.get('tensorflow', 'ANNOTATION_FILE_PATH')
        examples_path = Config.get('tensorflow', 'TRAINING_FILE_PATH') + '/'
        for exfile in os.listdir(examples_path):

            examplefile = examples_path + exfile

            examples_list = dataset_util.read_examples_list(examplefile)

            random.seed(42)
            random.shuffle(examples_list)

            num_examples = len(examples_list)
            num_train = int(0.7 * num_examples)
            train_examples = examples_list[:num_train]

            val_examples = examples_list[num_train:]

            NO_OF_TRAINING_EVALUATION_FILE = Config.get('tensorflow', 'NO_OF_TRAINING_EVALUATION_FILE')
            NO_OF_TRAINING_EVALUATION_FILE_OPEN = open(NO_OF_TRAINING_EVALUATION_FILE, 'w')
            NO_OF_TRAINING_EVALUATION_FILE_OPEN.write(
                'size of training : ' + str(len(train_examples)) + '  \nsize of evaluation : ' + str(len(val_examples)))
            logging.info('%d training and %d validation examples.',
                         len(train_examples), len(val_examples))

            OUTPUT_FILE_NAME_TRAIN = Config.get('tensorflow', 'OUTPUT_FILE_NAME_TRAIN')
            OUTPUT_FILE_NAME_EVAL = Config.get('tensorflow', 'OUTPUT_FILE_NAME_EVAL')
            train_output_path = os.path.join(FLAGS.output_dir, OUTPUT_FILE_NAME_TRAIN)
            val_output_path = os.path.join(FLAGS.output_dir, OUTPUT_FILE_NAME_EVAL)

            writertrain = tf.python_io.TFRecordWriter(train_output_path)

            writerval = tf.python_io.TFRecordWriter(val_output_path)

            create_tf_record(writertrain, label_map_dict, annotations_dir,
                             image_dir, train_examples, 'training')

            writertrain.close()
            create_tf_record(writerval, label_map_dict, annotations_dir,
                             image_dir, val_examples, 'eval')

            writerval.close()
    except:
        print('problems in creating the TFR dataset')
    print('TFR Dataset is created successfully')


main()
