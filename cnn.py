import tensorflow as tf
from tensorflow import keras

fmnist = tf.keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fmnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('loss') < 0.4):
            print("loss is less so training is stopping")
            self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ]
)

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=20,callbacks=[callbacks])

model.evaluate(test_images,test_labels)

model.save("mymodel.keras")

