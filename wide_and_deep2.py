import tensorflow as tf
from keras.layers import Dense, Flatten,Input, Lambda, Dropout,concatenate
from keras.models import Model
from keras.utils import plot_model


class WideAndDeep(Model):

  def __init__(self,units=32,activation='relu',**kwargs):
    super().__init__(**kwargs)

    self.dense_1 = Dense(units,activation=activation)
    self.dense_2 = Dense(units,activation=activation)

    self.output_1 = Dense(1,activation='sigmoid')
    self.output_2 = Dense(1,activation='sigmoid')

  
  def call(self,inputs):
    input_a,input_b = inputs

    x = self.dense_1(input_a)
    x = self.dense_2(x)
    concat = concatenate([x,input_b])

    output_1 = self.output_1(x)
    output_2 = self.output_2(concat)

    return output_1,output_2

model = WideAndDeep()


input_a = Input(shape=(28,28,))
input_b = Input(shape=(28,28,))

output_1,output_2 = model([input_a,input_b])

model_ = Model(inputs=[input_a,input_b],outputs=[output_1,output_2])

plot_model(model_,to_file='model_arch.png',show_shapes=True,show_layer_names=True)

