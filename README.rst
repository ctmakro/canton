======
Canton
======

The Canton library is a lightweight wrapper around TensorFlow, focused on **intuitive programmatical modeling and weight sharing**. It provides flexible ways to define, train, evaluate and save your computational networks.

    Canton is named after the city of Guangzhou. The French came a long time ago; they used to call this city "Canton", which sounds like "Guangdong" when pronounced in French, which is actually the name of the province, not the city. Since then, all westerners start to use the word Canton. The Yue language, a dialect of Chinese commonly used in Guangzhou and the United States, is known as "Cantonese" in English for this reason.

The Canton Philosophy
=====================

- The network units, and the weights associated with them, should be tied together as one, not seperated.
- Therefore, obtaining the weight tensors of any weighted action (or a set of actions bound together, or a network) should be as easy as calling ``some_action.get_weights()``, not ``tf.very_long_method_name(some_collection).some_other_method(some_name_prefixes)``.
- One should by default be able to create a unit once and apply it everywhere, while maintaining only one set of weights for that unit.

Usage
=====

Check `this tutorial <https://github.com/ctmakro/canton/blob/master/tutorial.ipynb>`_.

No explicit documentation. Please consider reading the source code (only 3 files).

Story Behind
============

TensorFlow is cool in general, but some of its designs are disasterous. The official way to share variables(weights) between copies of networks is to use ``tf.variable_scope(scopename, reuse=True)`` and ``tf.get_variable(name)``. It then became the programmer's responsibility to specify(and keep track of) the scope names, variable names and flags. As a programmer, I soon realized that *There are only two hard things in Computer Science: cache invalidation and naming things.*

TensorFlow is from Google, where CS PhDs write all the code, so that mustn't be their problem. In order to deal with my own incompetence, I wrote this library.

    Keras also wrapped the quirks and weirdness of TensorFlow and allows for rapid prototyping, but if you want to introduce your own calculation and/or manipulation operations into the model, you must first inherit Keras' Layer class, then (in some cases) specify a shape inference function, which is boring and inefficient. Besides that, Keras does not support anything other than the kaggle-styled, input-to-output-chained, one-loss-updates-everything architecture. I tried various method to wrap aroud Keras(in order to add my own functionality), but the internal complexity of Keras continuously freaked me off(I read almost every page of its documentation and half its code).

    Other learning frameworks also made various attempts on solving the same problem, using fancy descriptions like "imperative vs declarative". Well, maybe they do need a lot of PhDs to solve the *second hardest thing in Computer Science...*

Install
=======

pip install canton

Support Python 3 only.

dependencies:

- tensorflow-1.0.0
