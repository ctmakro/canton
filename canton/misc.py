import tensorflow as tf
import os
_SESSION = None
_TRAINING = True

def flatten(items,enter=lambda x:isinstance(x, list)):
    # http://stackoverflow.com/a/40857703
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if enter(x):
            yield from flatten(x)
        else:
            yield x

# borrowed from Keras
def get_session():
    """Returns the TF session to be used by the backend.
    If a default TensorFlow session is available, we will return it.
    Else, we will return the global Keras session.
    If no global Keras session exists at this point:
    we will create a new global session.
    Note that you can manually set the global session
    via `K.set_session(sess)`.
    # Returns
        A TensorFlow session.
    """
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    # if not _MANUAL_VAR_INIT:
    #     _initialize_variables()
    return session


def set_session(session):
    """Sets the global TF session.
    """
    global _SESSION
    _SESSION = session

def set_training_state(state=True):
    global _TRAINING
    _TRAINING = state

def get_training_state():
    global _TRAINING
    return _TRAINING

def set_variable(value,variable=None):
    """Load some value into session memory by creating a new variable.
    If an existing variable is given, load the value into the given variable.
    """
    sess = get_session()
    if variable is not None:
        assign_op = tf.assign(variable,value)
        sess.run([assign_op])
        return variable
    else:
        variable = tf.Variable(initial_value=value)
        sess.run([tf.variables_initializer([variable])])
    return variable

def get_variables_of_scope(collection_name,scope_name):
    var_list = tf.get_collection(collection_name, scope=scope_name)
    return var_list

def ph(shape,*args,**kwargs):
    return tf.placeholder(tf.float32,shape=[None]+shape,*args,**kwargs)

def gvi():
    return tf.global_variables_initializer()
