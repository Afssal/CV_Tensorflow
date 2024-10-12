import tensorflow as tf
from keras.layers import Dense, Flatten, UpSampling2D, MaxPooling2D,GlobalAvgPool2D

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

def preprocess_image(image):
  image = image.astype('float32')
  out_image = tf.keras.applications.resnet50.preprocess_input(image)
  return out_image

train_x = preprocess_image(train_images)
test_x = preprocess_image(test_images)


def feature_extractor(inputs):

  feature_extractor = tf.keras.applications.ResNet50(
      input_shape=(224,224,3),
      include_top=False,
      weights='imagenet'
  )(inputs)
  return feature_extractor

def classifier(inputs):
  x = GlobalAvgPool2D()(inputs)
  x = Flatten()(x)
  x = Dense(1024,activation='relu')(x)
  x = Dense(512,activation='relu')(x)
  x = Dense(10,activation='softmax',name='classification')(x)
  return x

def final_model(inputs):

  resize = UpSampling2D(size=(7,7))(inputs)

  resnet_feature_extractor = feature_extractor(resize)
  classification_output = classifier(resnet_feature_extractor)

  return classification_output


def compiler():

  inputs = tf.keras.Input(shape=(32,32,3))

  classification_output = final_model(inputs)

  model = tf.keras.Model(inputs=inputs,outputs = classification_output)

  model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

model = compiler()

csv_file = tf.keras.callbacks.CSVLogger('log.csv')

#early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,mode='min',patience=3)

#save model
model_save = tf.keras.callbacks.ModelCheckpoint('model.keras',save_best_only=True)

#tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

history = model.fit(
    train_x,train_labels,
    epochs=10,
    validation_data = (test_x,test_labels),
    callbacks = [csv_file,early_stop,model_save,tensorboard]
)