from Model import *

IMAGE_DIR = r'' #wacth the backslash
LABEL_DIR = r''
TRAIN_DIR = r''
TEST_DIR = r''
LEARNING_RATE = 1e-3
NAME = "classifier-{}-{}.model".format(LEARNING_RATE,'2conv-basic')

def Initialize():
    '''creates the relevant directories and splits the images'''
    create_directories(IMAGE_DIR)
    splitImageData(loadLabelData(LABEL_DIR,1),IMAGE_DIR,TRAIN_DIR,TEST_DIR)

def runModel():
    '''creates a new object of the model and runs it'''
    model = KrModel(TRAIN_DIR, TEST_DIR, LABEL_DIR,NAME,LEARNING_RATE)
    model.run()

# comment out this function if it is done

runModel()
