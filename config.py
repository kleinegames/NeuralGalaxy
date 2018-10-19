from Model import *


IMAGE_DIR = r'' #wacth the backslash
'''run the intialize function now'''
LABEL_DIR = r'' #put the path to the galaxy.dat file here
TRAIN_DIR = r'' #put the path to your training directory here
TEST_DIR = r'' #put the path to your testing directory here
LEARNING_RATE = 1e-3 # the learning rate of the model
NAME = "classifier-{}-{}.model".format(LEARNING_RATE,'2conv-basic') # the model name
''''run the runModel function now'''


def Initialize():
    '''creates the relevant directories and splits the images'''
    if(IMAGE_DIR != ''):
        create_directories(IMAGE_DIR)
        splitImageData(loadLabelData(LABEL_DIR,1),IMAGE_DIR,TRAIN_DIR,TEST_DIR)
    else:
        print("Please enter the image file directory into the IMAGE_DIR variable")

def runModel():
    '''creates a new object of the model and runs it'''
    if(LABEL_DIR != '' and TRAIN_DIR != '' and TEST_DIR != ''):
        model = KrModel(TRAIN_DIR, TEST_DIR, LABEL_DIR,NAME,LEARNING_RATE)
        model.run()
    else:
        print("Please make sure that the TRAIN_DIR, LABEL_DIR AND TEST_DIR are filled.")

runModel()
