import tensorflow as tf

eps = 1e-8

def loge(i):
    return tf.log(i+eps)

def one_hot_accuracy(pred,gt):
    correct_vector = tf.equal(tf.argmax(pred,1), tf.argmax(gt,1))
    acc = tf.reduce_mean(tf.cast(correct_vector,tf.float32))
    return acc

def mean_softmax_cross_entropy(pred,gt):
    return tf.reduce_mean(softmax_cross_entropy(pred,gt))

def softmax_cross_entropy(pred,gt):
    # tf r1.0 : must use named arguments
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=gt)

def cross_entropy_loss(pred,gt): # last dim is one_hot
    return - tf.reduce_mean(tf.reduce_sum(loge(pred) * gt, axis=tf.rank(pred)-1))

def binary_cross_entropy_loss(pred,gt,l=1.0): # last dim is 1
    return - tf.reduce_mean(loge(pred) * gt + l * loge(1.-pred) * (1.-gt))

def sigmoid_cross_entropy_loss(pred,gt): # same as above but more stable
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=gt)

def mean_sigmoid_cross_entropy_loss(pred,gt): # same as above but more stable
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=gt))
