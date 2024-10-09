import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense,Input,Flatten
from keras.models import Model
from keras.callbacks import Callback

(train_x,train_y),(test_x,test_y) = mnist.load_data()

width,height = train_x.shape[1],train_y.shape[2]

train_x = train_x/255
test_x = test_x/255


num_classes = test_y.shape[1]

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


def network():

    input = Input(shape=(width,height))

    x = Flatten()(input)
    x = Dense(128,activation='relu')(x)

    output = Dense(10,activation='softmax')(x)

    model = Model(inputs=input,outputs=output)

    return model

class MyCallback(Callback):

    def on_epoch_end(self,epoch,logs={}):
        if logs.get('loss') < 0.4 :
            print("loss is getting lower")
            self.model.stop_training = True

model = network()
callback = MyCallback()


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),epochs=20,callbacks=[callback])

