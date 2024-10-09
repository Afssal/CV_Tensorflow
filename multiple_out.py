import pandas as pd
import tensorflow as tf
from keras.layers import Dense,Input,Flatten
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_excel('ENB2012_data.xlsx')

print(df.columns)

X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]

Y = df[['Y1','Y2']]


#function to convert target columns to numpy array and return as tuple
def form(data):

  y1 = data['Y1']
  y1 = np.array(y1)
  y2 = data['Y2']
  y2 = np.array(y2)
  return y1,y2



train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3)

train_y = form(train_y)
test_y = form(test_y)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


def network():

    input = Input(shape=(len(X.columns),))

    x = Flatten()(input)
    dense_1 = Dense(128,activation='relu')(x)

    #for first output
    output_1 = Dense(1,name='y1')(dense_1)


    dense_2 = Dense(64)(dense_1)
    #for second output
    output_2 = Dense(1,name='y2')(dense_2)

    model = Model(inputs=input,outputs=[output_1,output_2])

    return model

model = network()

plot_model(model,to_file='model_arch.png',show_shapes=True,show_layer_names=True)

model.compile(loss={
    'y1':'mae',
    'y2':'mae'
},optimizer='adam',metrics={
    'y1':tf.keras.metrics.RootMeanSquaredError(),
    'y2':tf.keras.metrics.RootMeanSquaredError()
})

model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),epochs=20)
