import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense,Input,Flatten,Conv2D,MaxPool2D,Dropout,Lambda
from keras.models import Model
from keras.losses import Loss
from keras.optimizers import RMSprop
import tensorflow.keras.backend as k



(train_x,train_y),(test_x,test_y) = mnist.load_data()

train_x = train_x/255
test_x = test_x/255

# print(train_y)

def make_pair(train_x,train_y):

    #get unique class list
    num_class = np.unique(train_y)

    '''
        for i in range(len(num_classes)):
            idx = np.where(train_y == i)[0]
            digit_index.append(idx)

    '''

    #get array for each class consist of index of each element
    digit_index = [np.where(train_y == i)[0] for i in range(len(num_class))]

    pos_pair = []
    pos_label = []


    for idx in range(len(train_x)):

        #get a image
        current_img = train_x[idx]
        #corresponding label
        label = train_y[idx]

        #randomly select index position of any element belongs to current label
        idxb = np.random.choice(digit_index[label])

        #get image of selected index position
        pair_image = train_x[idxb]

        #append pair of image
        pos_pair.append([current_img,pair_image])
        #label as 1
        pos_label.append(1)

        #get a index which is label is not equal to current label
        negidx = np.where(train_y != label)[0]

        #get random image from current label
        neg_image = train_x[np.random.choice(negidx)]

        #append both image and label
        pos_pair.append([current_img,neg_image])
        pos_label.append(0)


    return (np.array(pos_pair),np.array(pos_label))
    # return negidx

# print(make_pair(train_x,train_y))

#make train pair dataset
train_pair,label_pair = make_pair(train_x,train_y)

#make test pair dataset
test_pair,test_label_pair = make_pair(test_x,test_y)


#input of first model
input_a = Input(shape=(28,28,1))

#input of second model
input_b = Input(shape=(28,28,1))


def network():

    input = Input(shape=(28,28,1))

    x = Conv2D(filters=64,kernel_size=(2,2),strides=(1,1))(input)
    x = MaxPool2D(strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(64,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32,activation='relu')(x)


    return Model(inputs=input,outputs=x)

def euclidean_distance(vector):

    x,y = vector
    sum_squares = k.sum(k.square(x-y),axis=1,keepdims=True)
    return k.sqrt(k.maximum(sum_squares,k.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1,shape2 = shapes
    return (shape1[0],1)

base_network = network()

#build first model
network_1 = base_network(input_a)

#second model
network_2 = base_network(input_b)

#define lambda layer
lambda_layer = Lambda(function=euclidean_distance,output_shape=eucl_dist_output_shape,name='outputshape')

#output layer
output = lambda_layer([network_1,network_2])

#full architecuture
model = Model(inputs=[input_a,input_b],outputs=output)

#define loss
class ContrastiveLoss(Loss):
    margin = 0
    def __init__(self,margin):
        super().__init__()
        self.margin = margin

    
    def call(self,y_true,y_pred):
        square_pred = k.square(y_pred)
        margin_square = k.square(k.maximum(self.margin-y_pred,0))
        return (y_true*square_pred+(1-y_true)*margin_square)

rms = RMSprop()

model.compile(loss=ContrastiveLoss(margin=1),optimizer=rms)

history = model.fit(
    #pass each image from pair to model
    x = [train_pair[:,0],train_pair[:,1]],
    y = label_pair,
    epochs = 20,
    batch_size = 128,
    validation_data = ([test_pair[:,0],test_pair[:,1]],test_label_pair)
)