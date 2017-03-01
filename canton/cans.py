import tensorflow as tf
import numpy as np
import time

from .misc import *

# this library is Python 3 only.

# this library aims to wrap up the ugliness of tensorflow,
# at the same time provide a better interface for NON-STANDARD
# learning experiments(such as GANs, etc.) than Keras.

# a Can is a container. it can contain other Cans.

class Can:
    def __init__(self):
        self.subcans = [] # other cans contained
        self.weights = [] # trainable
        self.variables = [] # should save with the weights, but not trainable
        self.updates = [] # update ops, mainly useful for batch norm
        # well, you decide which one to put into

        self.inference = None

    # by making weight, you create trainable variables
    def make_weight(self,shape):
        initial = tf.truncated_normal(shape, stddev=1e-3)
        w = tf.Variable(initial,name='W')
        self.weights.append(w)
        return w

    def make_bias(self,shape):
        initial = tf.constant(0.0, shape=shape)
        b = tf.Variable(initial,name='b')
        self.weights.append(b)
        return b

    # make a variable that is not trainable, by passing in a numpy array
    def make_variable(self,nparray,name='v'):
        v = tf.Variable(nparray,name=name)
        self.variables.append(v)
        return v

    # put other cans inside this can, as subcans
    def incan(self,c):
        if hasattr(c,'__iter__'): # if iterable
            self.subcans += list(c)
        else:
            self.subcans += [c]
        # return self

    # another name for incan
    def add(self,c):
        self.incan(c)
        return c

    # if you don't wanna specify the __call__ function manually,
    # you may chain up all the subcans to make one:
    def chain(self):
        def call(i):
            for c in self.subcans:
                i = c(i)
            return i
        self.set_function(call)

    # traverse the tree of all subcans,
    # and extract a flattened list of certain attributes.
    # the attribute itself should be a list, such as 'weights'.
    # f is the transformer function, applied to every entry
    def traverse(self,target='weights',f=lambda x:x):
        l = [f(a) for a in getattr(self,target)] + [c.traverse(target,f) for c in self.subcans]
        # the flatten logic is a little bit dirty
        return list(flatten(l, lambda x:isinstance(x,list)))

    # return weight tensors of current can and it's subcans
    def get_weights(self):
        return self.traverse('weights')

    # return update operations of current can and it's subcans
    def get_updates(self):
        return self.traverse('updates')

    # set __call__ function
    def set_function(self,func):
        self.func = func

    # default __call__
    def __call__(self,i,*args,**kwargs):
        if hasattr(self,'func'):
            return self.func(i,*args,**kwargs)
        else:
            raise NameError('You didnt override __call__(), nor called set_function()/chain()')

    def get_value_of(self,tensors):
        sess = get_session()
        values = sess.run(tensors)
        return values

    def save_weights(self,filename): # save both weights and variables
        with open(filename,'wb') as f:
            # extract all weights in one go:
            w = self.get_value_of(self.get_weights()+self.traverse('variables'))
            print('weights (and variables) obtained.')
            np.save(f,w)
            print('successfully saved to',filename)
            return True

    def load_weights(self,filename):
        with open(filename,'rb') as f:
            loaded_w = np.load(f)
            print('successfully loaded from',filename)
            # but we cannot assign all those weights in one go...
            model_w = self.get_weights()+self.traverse('variables')
            if len(loaded_w)!=len(model_w):
                raise NameError('number of weights (variables) from the file({}) differ from the model({}).'.format(len(loaded_w),len(model_w)))
            else:
                assign_ops = [tf.assign(model_w[i],loaded_w[i])
                    for i,_ in enumerate(model_w)]

            sess = get_session()
            sess.run(assign_ops)
            print(len(loaded_w),'weights assigned.')
            return True

    def infer(self,i):
        # run function, return value
        if self.inference is None:
            # the inference graph will be created when you infer for the first time
            # 1. create placeholders with same dimensions as the input
            if isinstance(i,list): # if Can accept more than one input
                x = [tf.placeholder(tf.float32,shape=[None for _ in range(len(j.shape))])
                    for j in i]
            else:
                x = tf.placeholder(tf.float32, shape=[None for _ in range(len(i.shape))])
            y = self.__call__(x)

            # 2. create the inference function
            def inference(k):
                sess = get_session()
                if isinstance(i,list):
                    res = sess.run(y,feed_dict={x[j]:k[j]
                        for j,_ in enumerate(x)})
                else:
                    res = sess.run(y,feed_dict={x:k})
                return res
            self.inference = inference

        return self.inference(i)

    def summary(self):
        print('-------------------')
        print('Directly Trainable:')
        variables_summary(self.get_weights())
        print('-------------------')
        print('Not Directly Trainable:')
        variables_summary(self.traverse('variables'))
        print('-------------------')

def variables_summary(var_list):
    shapes = [v.get_shape() for v in var_list]
    shape_lists = [s.as_list() for s in shapes]
    shape_lists = list(map(lambda x:''.join(map(lambda x:'{:>5}'.format(x),x)),shape_lists))

    num_elements = [s.num_elements() for s in shapes]
    total_num_of_variables = sum(num_elements)
    names = [v.name for v in var_list]

    print('counting variables...')
    for i in range(len(shapes)):
        print('{:>25}  ->  {:<6} {}'.format(
        shape_lists[i],num_elements[i],names[i]))

    print('{:>25}  ->  {:<6} {}'.format(
    'tensors: '+str(len(shapes)),
    str(total_num_of_variables),
    'variables'))

# you know, MLP
class Dense(Can):
    def __init__(self,num_inputs,num_outputs,bias=True):
        super().__init__()
        self.W = self.make_weight([num_inputs,num_outputs])
        self.use_bias = bias
        if bias:
            self.b = self.make_bias([num_outputs])
    def __call__(self,i):
        d = tf.matmul(i,self.W)
        if self.use_bias:
            return d + self.b
        else:
            return d

# apply Dense on last dimension of 1d input. equivalent to conv1d.
class TimeDistributedDense(Dense):
    def __call__(self,i):
        s = tf.shape(i)
        b,t,d = s[0],s[1],s[2]
        i = tf.reshape(i,[b*t,d])
        i = super().__call__(i)
        d = tf.shape(i)[1]
        i = tf.reshape(i,[b,t,d])
        return i

# you know, shorthand
class Lambda(Can):
    def __init__(self,f):
        super().__init__()
        self.set_function(f)

# you know, nonlinearities
class Act(Can):
    def __init__(self,name):
        super().__init__()
        activations = {
            'relu':tf.nn.relu,
            'tanh':tf.tanh,
            'sigmoid':tf.sigmoid,
            'softmax':tf.nn.softmax,
            'elu':tf.nn.elu
        }
        self.set_function(activations[name])

# you know, Yann LeCun
class Conv2D(Can):
    # nip and nop: input and output planes
    # k: dimension of kernel, 3 for 3x3, 5 for 5x5
    def __init__(self,nip,nop,k,std=1,usebias=True,padding='SAME'):
        super().__init__()
        self.nip,self.nop,self.k,self.std,self.usebias,self.padding = nip,nop,k,std,usebias,padding

        self.W = self.make_weight([k,k,nip,nop]) # assume square window
        if usebias==True:
            self.b =self.make_bias([nop])

    def __call__(self,i):
        c = tf.nn.conv2d(i,self.W,
            strides=[1, self.std, self.std, 1],
            padding=self.padding)

        if self.usebias==True:
            return c + self.b
        else:
            return c

# you know, recurrency
class Scanner(Can):
    def __init__(self,f):
        super().__init__()
        self.f = f
    def __call__(self,i,starting_state=None, state_shape=None):
        # previous state is useful when running online.
        if starting_state is None:
            if state_shape is None:
                print('cannot infer state_shape. use shape of input[0] instead. please make sure the input to the Scanner has the same last dimension as the function being scanned.')
                initializer = tf.zeros_like(i[0])
            else:
                initializer = tf.zeros(state_shape, tf.float32)
        else:
            initializer = starting_state
        scanned = tf.scan(self.f,i,initializer=initializer)
        return scanned

# deal with batch input.
class BatchScanner(Scanner):
    def __call__(self, i, **kwargs):
        it = tf.transpose(i, perm=[1,0,2])
        #[Batch, Seq, Dim] -> [Seq, Batch, Dim]
        scanned = super().__call__(it, **kwargs)
        scanned = tf.transpose(scanned, perm=[1,0,2])
        return scanned

# single forward pass version of GRU. Normally we don't use this directly
class GRU_onepass(Can):
    def __init__(self,num_in,num_h):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = Dense(num_in+num_h,num_h,bias=False)
        self.wr = Dense(num_in+num_h,num_h,bias=False)
        self.w = Dense(num_in+num_h,num_h,bias=False)
        self.incan([self.wz,self.wr,self.w])
        # http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        wz,wr,w = self.wz,self.wr,self.w
        c = tf.concat([hidden,inp],axis=1)
        z = tf.sigmoid(wz(c))
        r = tf.sigmoid(wr(c))
        h_c = tf.tanh(w(tf.concat([hidden*r,inp],axis=1)))
        h_new = (1-z) * hidden + z * h_c
        return h_new

# RNN Can generator from cells, similar to tf.nn.dynamic_rnn
def rnn_gen(name, unit_class):
    class RNN(Can):
        def __init__(self,*args):
            super().__init__()
            self.unit = unit_class(*args)
            def f(last_state, new_input):
                return self.unit([last_state, new_input])
            self.bscan = BatchScanner(f)
            self.incan([self.unit,self.bscan])
        def __call__(self,i,**kwargs):
            # given input, what should be the shape of the state?
            s = tf.shape(i) #[batch, timesteps, dim]
            state_shape = [s[0], self.unit.num_h] # [batch, hidden_dim]

            return self.bscan(i,state_shape=state_shape, **kwargs)
    RNN.__name__ = name
    return RNN

# you know, Despicable Me
GRU = rnn_gen('GRU', GRU_onepass)

# you know, LeNet
class AvgPool2D(Can):
    def __init__(self,k,std,padding='SAME'):
        super().__init__()
        self.k,self.std,self.padding = k,std,padding

    def __call__(self,i):
        k,std,padding = self.k,self.std,self.padding
        return tf.nn.avg_pool(i, ksize=[1, k, k, 1],
            strides=[1, std, std, 1], padding=padding)

# you know, He Kaiming
class ResConv(Can): # v2
    def __init__(self,nip,nop,std=1):
        super().__init__()
        # create the necessary cans:
        nbp = int(max(nip,nop)/4) # bottleneck
        self.direct_sum = nip==nop and std==1 # if no downsampling and feature shrinking

        if self.direct_sum:
            self.convs = [Conv2D(nip,nbp,1),Conv2D(nbp,nbp,3),Conv2D(nbp,nop,1)]
        else:
            self.convs = [Conv2D(nip,nbp,1,std=std),Conv2D(nbp,nbp,3),
            Conv2D(nbp,nop,1),Conv2D(nip,nop,1,std=std)]

        self.incan(self.convs) # add those cans into collection

    def __call__(self,i):
        def bnr(i):
            bn = lambda x:x
            relu = tf.nn.relu
            return bn(relu(i))

        if self.direct_sum:
            ident = i
            i = bnr(i)
            i = self.convs[0](i)
            i = bnr(i)
            i = self.convs[1](i)
            i = bnr(i)
            i = self.convs[2](i)
            out = ident+i
        else:
            i = bnr(i)
            ident = i
            i = self.convs[0](i)
            i = bnr(i)
            i = self.convs[1](i)
            i = bnr(i)
            i = self.convs[2](i)
            ident = self.convs[3](ident)
            out = ident+i
        return out
