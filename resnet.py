import tensorflow_datasets as tfds

import tensorflow as tf
from keras.layers import Layer,Dense,Conv2D,BatchNormalization,Activation,Add,GlobalAveragePooling2D,MaxPool2D
from keras.models import Model

class IdentityBlock(Model):

    def __init__(self,filters,kernel_size):
        super(IdentityBlock,self).__init__(name='')


        self.conv1 = Conv2D(filters,kernel_size,padding='same')
        self.batch = BatchNormalization()
        self.act = Activation('relu')
        self.add = Add()

    def call(self,inputs):
      x = self.conv1(inputs)
      x = self.batch(x)
      x = self.act(x)
      x = self.conv1(x)
      x = self.batch(x)
      x = self.add([x,inputs])
      x = self.act(x)

      return x
    
class ResNet(Model):
  def __init__(self,num_classes):
    super(ResNet,self).__init__()

    self.conv1 = Conv2D(64,7,padding='same')
    self.batch = BatchNormalization()
    self.max = MaxPool2D(3,strides=2,padding='same')
    self.idblock1 = IdentityBlock(64,3)
    self.idblock2 = IdentityBlock(64,3)
    self.globalpool = GlobalAveragePooling2D()
    self.classifier = Dense(num_classes,activation='softmax')
    self.act = Activation('relu')

  def call(self,inputs):

    x = self.conv1(inputs)
    x = self.batch(x)
    x = self.act(x)
    x = self.max(x)
    x = self.idblock1(x)
    x = self.idblock2(x)
    x = self.globalpool(x)
    return self.classifier(x)
  
def preprocess(features):
  return tf.cast(features['image'],tf.float32)/255.,features['label']

resnet = ResNet(10)
resnet.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

dataset = tfds.load('mnist',split=tfds.Split.TRAIN)
dataset = dataset.map(preprocess).batch(32)