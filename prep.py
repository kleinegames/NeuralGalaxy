
from Functions import *
import os
import glob


dict = loadLabelData('I:\data\info\galaxy_data.dat',1)
list = glob.glob("\data\images\gace_on_images\check\*.jpg")
for x in list:
    y = x.replace(".jpg","")
    z = y.replace("\data\images\gace_on_images\check\galrand_","")
    dict.pop(z)
#10435 samples

def orderImageData(dict):
    '''a function which splits the dataset into directories based on Kappa_rotation value '''
    for x in dict:
        if(dict[x] == 0):
        # take the picture and move into the non-disc folder
            os.rename('I:\data\images\gace_on_images\practice\galrand_'+str(x)+".jpg",'I:\data\images\gace_on_images\practice\sphere\galrand_'+str(x)+".jpg")
            #elif(dict[x] == 1):
            #os.rename("I:\data\images\gace_on_images\galrand_"+str(x)+".jpg","I:\data\images\gace_on_images\partial-disc\galrand_"+str(x)+".jpg")
        else:
            os.rename("I:\data\images\gace_on_images\practice\galrand_"+str(x)+".jpg","I:\data\images\gace_on_images\practice\disc\galrand_"+str(x)+".jpg")


orderImageData(dict)
