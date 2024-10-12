import tensorflow as tf
from keras.layers import Dense, Flatten,Input, Lambda, Dropout,concatenate
from keras.models import Model
from keras.utils import plot_model


input_a = Input(shape=(28,28,))
input_b = Input(shape=(28,28,))
dense_1 = Dense(128,activation='relu')(input_a)
dense_2 = Dense(128,activation='relu')(input_a)
concat = concatenate([dense_2,input_b])
output = Dense(1,activation='sigmoid')(concat)
model = Model(inputs=[input_a,input_b],outputs=output)


plot_model(model,to_file='model_arch.png',show_shapes=True,show_layer_names=True)


