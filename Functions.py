import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
import glob
from random import shuffle

def getImageShape(Kr):
    '''Returns a label based on a kappa_rotation value'''
    if(Kr < 0.325):
        return [0,1] #"non-discy galaxy"
    #elif( Kr > 0.325 and Kr < 0.51):
    #    return "partially discy galaxy "
    else:
        return [1,0] #"discy galaxy"

def loadLabelData(path,id): #path = 'I:\data\info\galaxy_data.dat'
    '''Returns either a list of Kr values or a dictionary of labels and their identifiers'''
    property_data = np.genfromtxt(path, dtype = (int,int,float,int,float,float,float,float,float), comments = "#", usecols = (0,5))
    KrData = [i[1] for i in property_data.tolist()]
    GalaxyId =  [i[0] for i in property_data.tolist()]
    LabelData = [getImageShape(x) for x in KrData]
    GaLabs = {}
    for x in range(0,len(GalaxyId)):
        #GaLabs["galrand_"+str(GalaxyId[x])+".jpg"] = LabelData[x]
        GaLabs["galrand_"+str(GalaxyId[x])+".jpg"] = LabelData[x]
    if(id == 0):
        return KrData
    elif(id == 1):
        return GaLabs

def loadImage(image,image_size):
    ''''''
    print("loading image|"+str(image))
    image = cv2.imread(image)
    image = cv2.resize(image,(image_size,image_size),3)
    return image

def create_directories(path):
    Traindir = path+r'\training'
    Testdir = path+r'\testing'
    if not os.path.exists(Traindir):
        os.makedirs(Traindir)
    if not os.path.exists(Testdir):
        os.makedirs(Testdir)


def splitImageData(dict,image_dir,train_dir,test_dir):
    '''Splits the image dataset into a test and training directory '''
    i = 0
    for x in dict:
        if(i < (len(dict)-2500)):
            os.rename(image_dir+'galrand_'+str(x)+".jpg",train_dir+'galrand_'+str(x)+".jpg")
            print("Image added to training set")
        else:
            os.rename(image_dir+'galrand_'+str(x)+".jpg",test_dir+'galrand_'+str(x)+".jpg")
            print("image added to testing")
        i = i+1


def plotKrHistogram(path,binstep):
    '''plots a histogram of the Kr distribution'''
    data = loadLabelData(path,0)
    bins = np.arange(0.0,1.0,binstep)
    print("average:"+ str(np.mean(data)))
    print("variance:"+str((np.std(data))**2))
    print("STD:"+str(np.std(data)))
    plt.hist(data, bins, color = "maroon", rwidth = 5)
    plt.title("Distribution of samples based on Kappa_rotation")
    plt.xlabel('Kappa_rotation')
    plt.ylabel('frequency')
    plt.show()

def create_train_data(labelData,image_size,train_dir):
    '''loads the image and label datasets into a numpy array'''
    training_data = []
    for img in glob.glob(train_dir):
        #label = labelData[img]
        img = loadImage(img,image_size)
        training_data.append([np.array(img),np.array([1,0])])
    shuffle(training_data)
    #np.save('train_data.npy',training_data)
    return training_data

def process_test_data(test_dir):
    '''loads the image and label datasets into a numpy array'''
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
