from Functions import *
import tflearn
import cv2
import glob
from random import shuffle

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.reset_default_graph()

TRAIN_DIR = r'I:\data\images\gace_on_images\practice\disc\*.jpg'
TRAIN_DIR2 = r'I:\data\images\gace_on_images\practice\sphere\*.jpg'
TEST_DIR = r'I:\data\images\gace_on_images\check\*.jpg'
IMG_SIZE = 24
LR = 1e-3
MODEL_NAME = "classifier-{}-{}.model".format(LR,'2conv-basic')

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet) #,tensorboard_dir='log')

data = loadLabelData('I:\data\info\galaxy_data.dat',1)
train_data = create_train_data(data,IMAGE_SIZE)

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=2, validation_set=({'input': test_x}, {'targets': test_y}),
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#predicting values 
