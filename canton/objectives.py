import tensorflow as tf

def one_hot_accuracy(pred,gt):
    correct_vector = tf.equal(tf.argmax(pred,1), tf.argmax(gt,1))
    acc = tf.reduce_mean(tf.cast(correct_vector,tf.float32))
    return acc

def mean_softmax_cross_entropy(pred,gt):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,gt))
    return loss
