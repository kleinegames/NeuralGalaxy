import sys
from Model import *

'''use sysargv to create seperate entities for laptop and leiden'''
if(len(sys.argv[1:]) > 0):
    if(sys.argv[1] == 'leiden'):
        IMAGE_DIR = r''
        LABEL_DIR = r''
        TRAIN_DIR = r''
        TEST_DIR = r''
        OS = 'lin'
    elif(sys.argv[1] == "laptop"):
        IMAGE_DIR = r'I:\data\images\gace_on_images'
        LABEL_DIR = r'I:\data\info\galaxy_data.dat'
        TRAIN_DIR = r'I:\data\images\gace_on_images\training'
        TEST_DIR = r'I:\data\images\gace_on_images\testing'
        OS = 'win'
else:
    print("please pass either leiden or laptop as a argument")
    quit()

LEARNING_RATE = 1e-3
NAME = "classifier-{}-{}.model".format(LEARNING_RATE,'2conv-basic')

def Initialize():
    '''creates the relevant directories and splits the images'''
    create_directories(IMAGE_DIR)
    splitImageData(loadLabelData(LABEL_DIR,1),IMAGE_DIR,TRAIN_DIR,TEST_DIR)

def RunModel():
    '''creates a new object of the model and runs it'''
    model = KrModel(TRAIN_DIR, TEST_DIR, LABEL_DIR,NAME,LEARNING_RATE, OS)
    # model.run()
    model.checkModel(1)



#Initialize()
RunModel()
