import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
import glob
from random import shuffle
from tqdm import tqdm

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
        GaLabs["\galrand_"+str(GalaxyId[x])+".jpg"] = LabelData[x]
    if(id == 0):
        return KrData
    elif(id == 1):
        return GaLabs

def loadImage(image,image_size):
    '''loads and resizes the image'''
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
    train,test = splitTestingTraining(list(dict.keys()),2500)
    print("filling training directory")
    for y in tqdm(range(0,len(train))):
        x = train[y]
        os.rename(image_dir+str(x),train_dir+str(x))
    print("filling testing directory")
    for y in tqdm(range(0,len(test))):
        x = test[y]
        os.rename(image_dir+str(x),test_dir+str(x))


def splitTestingTraining(array,size):
    test = np.random.choice(array,size,replace= False)
    train = set(array)-set(test)
    return list(train),test


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

def create_train_data(labelData,image_size,train_dir, save = False):
    '''loads the image and label datasets into a numpy array'''
    training_data = []
    if(os.path.exists('train_data.npy')):
        training_data = np.load('train_data.npy')
    else:
        print("loading image data for training:")
        for img in tqdm(glob.glob(train_dir)):
            lb = img.replace(train_dir.replace("\*.jpg",""),"")
            label = labelData[lb]
            img = loadImage(img,image_size)
            training_data.append([np.array(img),np.array(label)])
            shuffle(training_data)
        if(save == True):
            np.save('train_data.npy',training_data)
    return training_data

def process_test_data(test_dir,image_size, save = False):
    '''loads the image and label datasets into a numpy array'''
    testing_data = []
    print("loading image data for testing:")
    for img in tqdm(glob.glob(test_dir)):
        z = img.replace(test_dir.replace("\*.jpg",""),"")
        img = loadImage(img,image_size)
        testing_data.append([np.array(img), z])
    if(save == True):
        np.save('testing_data.npy',testing_data)
    return testing_data
