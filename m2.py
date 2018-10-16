from Functions import *
import tflearn
import cv2
import glob
from random import shuffle

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
tf.reset_default_graph()
'''
problems:
the model is not working properly because the index is out of bounds?? -> solution currenly unknown!
sphere images are not loading properly because a file is missing(ghost file)? -> find a function which checks if a file really exists
'''

TRAIN_DIR = r'I:\data\images\gace_on_images\practice\disc\*.jpg'
TRAIN_DIR2 = r'I:\data\images\gace_on_images\practice\sphere\*.jpg'
TEST_DIR = r'I:\data\images\gace_on_images\check\*.jpg'
IMG_SIZE = 24
LR = 1e-3
MODEL_NAME = "classifier-{}-{}.model".format(LR,'2conv-basic')


data = loadLabelData('I:\data\info\galaxy_data.dat',1) #load a dictionary containing the imageID's and the label

def loadImage(image):
    print("loading image|"+str(image))
    image = cv2.imread(image)
    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE),3)
    return image



def create_train_data():
    training_data = []

    for img in glob.glob(TRAIN_DIR):
        #label = data[img]
        img = loadImage(img)
        training_data.append([np.array(img),np.array([1,0])])

    for img2 in glob.glob(TRAIN_DIR2): #problem file is found which is not there and cannot be created
        #label2 = data[img2]
        img2 = loadImage(img2)
        training_data.append([np.array(img2),np.array([0,1])])

    shuffle(training_data)
    #np.save('train_data.npy',training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in os.listdir(TEST_DIR):
        i = img.replace(".jpg","")
        z = i.replace("\data\images\gace_on_images\check\galrand_","")
        path = os.path.join(TEST_DIR,img)
        #if(os.path.isdir(path)):
        img = cv2.imread(str(path))
        #img = cv2.resize(img,(IMG_SIZE,IMG_SIZE),3)
        testing_data.append([np.array(img), z])
    return testing_data

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

train_data = create_train_data()

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=2, validation_set=({'input': test_x}, {'targets': test_y}),
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
