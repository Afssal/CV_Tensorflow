import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense,Input,Flatten

(train_x,train_y),(test_x,test_y) = mnist.load_data()

width,height = train_x.shape[1],train_x.shape[2]

train_x = train_x/255
test_x = test_x/255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


num_classes = test_y.shape[1]


class Mycallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('loss')<0.4):
            print("loss is getting lower so....")
            self.model.stop_training = True


def network():

    model = Sequential(
        [
            Input(shape=(width,height)),
            Flatten(),
            Dense(64,activation='relu'),
            Dense(num_classes,activation='softmax')
        ]
    )
    return model

callbacks = Mycallback()
model = network()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),epochs=20,callbacks=[callbacks])