from utils import *
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR) #suppress the keepdims warning

class KrModel:

    def __init__(self,train_dir,test_dir,label_dir,name,lr):
        self.slash = os.sep
        self.train_dir = train_dir
        self.test_dir = test_dir+self.slash+r'*.jpg'
        self.label_dir = label_dir
        self.name = name
        self.lr = lr
        self.img_size = 62



    def create_model(self):
        tf.reset_default_graph() #reset the tensorflow graph
        convnet = input_data(shape=[None, self.img_size, self.img_size, 3], name='input')

        convnet = conv_2d(convnet, 32, 4, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 4, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 4, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 4, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 4, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 4, activation='relu')
        convnet = max_pool_2d(convnet, 2)


        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.lr, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet,tensorboard_dir='log')
        return model

    def train(self):
        TRAIN_DIR = self.train_dir+self.slash+r'*.jpg'
        TEST_DIR = self.test_dir
        LABEL_DIR = self.label_dir
        LR = self.lr
        MODEL_NAME = self.name
        IMG_SIZE = self.img_size
        model = self.create_model()

        data = load_labels(LABEL_DIR,1)
        train_data = create_train_data(data,IMG_SIZE,TRAIN_DIR,True)
        testIndices = np.random.choice(len(train_data),500,replace = False)
        test = []
        index = []
        for x in range(0,500):
            index.append(testIndices[x])
            test.append(train_data[testIndices[x]])
        train = [train_data[i] for i in range(0,len(train_data)) if i not in index]
        X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_epoch= True, show_metric = True, run_id=MODEL_NAME)          # callbacks = predict_test_accuracy(model,self.test_dir,self.img_size,load_labels(self.label_dir,1))
        #model.save(self.name)
        return model

    def train_model(self,runs):
        score = []
        runs = [i for i in range(0,runs)]
        Labels = load_labels(self.label_dir,1)
        for i in runs: #train the model several times to account for randomness
            print("number of training sessions: "+str(i+1))
            model =  self.train() #trains the model
            score = predict_test_accuracy(model,self.test_dir,self.img_size,Labels) #tests the model
            print("the test accuracies computed over " + str(i+1) + " runs are"+str(score))
            print("the average test accuracy: "+str(sum(score)/len(score)))
