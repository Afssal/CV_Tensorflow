import tensorflow as tf

'''BackupAndRestore callback is intended to recover training from an interruption
that has happened in the middle of a Model.fit execution '''
backup = tf.keras.callbacks.BackupAndRestore(
    backup_dir='./backup',
    save_freq='epoch'
)
#to save epoch result in csv file
csv_file = tf.keras.callbacks.CSVLogger('log.csv')

#early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,mode='min',patience=3)

#save model
model_save = tf.keras.callbacks.ModelCheckpoint('model.keras',save_best_only=True)

#tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

resnet.fit(dataset,epochs=10,callbacks=[csv_file,early_stop,model_save,tensorboard])
