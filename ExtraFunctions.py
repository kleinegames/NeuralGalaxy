

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


def checkImageData(array,debug=0): #input = glob.glob("\data\images\gace_on_images\disc\*.jpg")
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
