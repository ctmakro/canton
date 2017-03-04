import tensorflow as tf

eps = 1e-8

def loge(i):
    return tf.log(i+eps)

def one_hot_accuracy(pred,gt):
    correct_vector = tf.equal(tf.argmax(pred,1), tf.argmax(gt,1))
    acc = tf.reduce_mean(tf.cast(correct_vector,tf.float32))
    return acc

def mean_softmax_cross_entropy(pred,gt):
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=gt))
    # tf r1.0 : must use named arguments
    return loss

def cross_entropy_loss(pred,gt):
    return - tf.reduce_mean(tf.reduce_sum(loge(pred) * gt, axis=-1))

def binary_cross_entropy_loss(pred,gt):
    return - tf.reduce_mean(
            tf.reduce_sum(
                loge(pred) * gt + loge(1-pred) * (1-gt), axis=-1)
        ) * 0.5
