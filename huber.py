from keras.losses import Loss
import tensorflow as tf


class huberloss(Loss):
    '''
       1/2(a)^2 if |a| <= threshold
       threhold * (|a|-1/2*threshold) if |a|>threshold
    
    '''
    threshold = 1
    def __init__(self,threshold):
        super.__init__()
        self.threshold = threshold

    def call(self,y_true,y_pred):
        #calculate error
        error = y_true-y_pred
        #check whether |a| less than or greater than threshold 
        small_error = tf.abs(error) <= self.threshold

        #1/2(a^2)
        small_error_loss = tf.square(error)/2

        #threhold * (|a|-1/2*threshold)
        big_error_loss = self.threshold * (tf.abs(error)-(0.5*self.threshold))

        return tf.where(small_error,small_error_loss,big_error_loss)