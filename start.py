import sys
from model import *
from pyfiglet import figlet_format

'''
to do list:
- add in callbacks to compute test accuracy every single epoch
-add save and load functionality to the model
-experiment with increased neuron count in convolutional layers
-use background free images to see possible acc increase
-create the SSFR classifier model
-integrate the SSFR and KR model into one compact module
'''

os.system('cls' if os.name == 'nt' else 'clear')
print(figlet_format('EAGLE SHAPE MODEL', font='slant'))


LEARNING_RATE = 1e-3
NAME = "classifier-{}-{}.model".format(LEARNING_RATE,'2conv-basic')

'''use sysargv to create seperate entities for laptop and leiden'''
if(len(sys.argv[1:]) > 0):
    if(sys.argv[1] == 'leiden'):
        IMAGE_DIR = r'put the directory path for the image files here'
        LABEL_DIR = r'put the directory path for the labels here'
        TRAIN_DIR = r'put the created training directory here'
        TEST_DIR = r'put the created testing directory here'

    elif(sys.argv[1] == "laptop"):
        IMAGE_DIR = r'I:\data\images\gace_on_images'
        LABEL_DIR = r'I:\data\info\galaxy_data.dat'
        TRAIN_DIR = r'I:\data\images\gace_on_images\training'
        TEST_DIR = r'I:\data\images\gace_on_images\testing'

else:
    print("please pass either leiden or laptop as a argument")
    quit()

def initialize():
    '''creates the relevant directories and splits the images'''
    create_directories(IMAGE_DIR)
    train_test_split(load_labels(LABEL_DIR,1),IMAGE_DIR,TRAIN_DIR,TEST_DIR)

def run_training_session():
    '''creates a new object of the model and runs it'''
    model = KrModel(TRAIN_DIR, TEST_DIR, LABEL_DIR,NAME,LEARNING_RATE)
    # model.run()
    model.train_model(1)

#initialize()
run_training_session()
