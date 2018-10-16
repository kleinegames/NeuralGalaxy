import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''the neural stuff
https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/inception/inception/data/build_image_data.py
https://kwotsin.github.io/tech/2017/01/29/tfrecords.html follow this guide!! make a TFrecord file
https://medium.com/@ipaar3/how-i-built-a-convolutional-image-classifier-using-tensorflow-from-scratch-f852c34e1c95
using retrain.py to transfer learn the inception model
'''


def getImageShape(Kr):
    '''Simple function that takes a Kappa_rotation and returns a label.'''
    if(Kr < 0.325):
        return [0,1] #"non-discy galaxy"
    #elif( Kr > 0.325 and Kr < 0.51):
    #    return "partially discy galaxy "
    else:
        return [1,0] #"discy galaxy"

def loadLabelData(path,id): #path = 'I:\data\info\galaxy_data.dat'
    '''A function wh ich takes a path and identifier and returns something'''
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

def orderImageData(dict):
    '''a function which splits the dataset into directories based on Kappa_rotation value '''
    for x in dict:
        if(dict[x] == 0):
            # take the picture and move into the non-disc folder
            os.rename('I:\data\images\gace_on_images\galrand_'+str(x)+".jpg",'I:\data\images\gace_on_images\sq-disc\galrand_'+str(x)+".jpg")
        #elif(dict[x] == 1):
            #os.rename("I:\data\images\gace_on_images\galrand_"+str(x)+".jpg","I:\data\images\gace_on_images\partial-disc\galrand_"+str(x)+".jpg")
        else:
            os.rename("I:\data\images\gace_on_images\galrand_"+str(x)+".jpg","I:\data\images\gace_on_images\disc\galrand_"+str(x)+".jpg")

def getImageData(array,debug=0): #input = glob.glob("\data\images\gace_on_images\disc\*.jpg")
    '''a function which loads all images into a numpy array given an array of directory positions'''
    images = np.array([])
    for i in array:
        image_string = tf.read_file(str(i))
        image = tf.image.decode_image(image_string)
        np.append(images,image)
        if(debug == 1):
            with tf.Session() as session:
                img_value = session.run(image)
                print(np.min(img_value), np.max(img_value), np.mean(img_value))
    return images

def splitImageData(dict):
    '''a function which splits the dataset into a test and training directory '''
    i = 0
    for x in dict:
        if(i < (len(dict)-2500)):
            os.rename('I:\data\images\gace_on_images\galrand_'+str(x)+".jpg",'I:\data\images\gace_on_images\practice\galrand_'+str(x)+".jpg")
            print("image added to training")
        else:
            os.rename('I:\data\images\gace_on_images\galrand_'+str(x)+".jpg",'I:\data\images\gace_on_images\check\galrand_'+str(x)+".jpg")
            print("image added to testing")
        i = i+1


def plotKrHistogram(path,binstep):
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

'''
#load the label and imagedata
#reduce imagesize
#format the data for tensorflow
'''
