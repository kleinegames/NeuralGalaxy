from Functions import *
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
tf.reset_default_graph()


class KrModel:
    def __init__(self,train_dir,test_dir,label_dir,name,lr):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.label_dir = label_dir
        self.name = name
        self.lr = lr

    def run(self):
        TRAIN_DIR = self.train_dir+r'\*.jpg'
        TEST_DIR = self.test_dir+r'\*.jpg'
        LABEL_DIR = self.label_dir
        IMG_SIZE = 24
        LR = self.lr
        MODEL_NAME = self.name #"classifier-{}-{}.model".format(LR,'2conv-basic')

        data = loadLabelData(LABEL_DIR,1)
        train_data = create_train_data(data,IMG_SIZE,TRAIN_DIR)
        #train_data = np.load('train_data.npy')


        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet,tensorboard_dir='log')

        train = train_data[:-500]
        test = train_data[-500:]

        X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#predicting values
