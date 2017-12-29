from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.models import Model
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, Input, Flatten, Dense
from keras.layers import BatchNormalization, Dropout, PReLU

import tensorflow as tf
import numpy as np

from triplets_generator import DataGenerator
import evaluate

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

K.set_image_dim_ordering('tf')

resnet_input = Input(shape=(224,224,3))
resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=resnet_input)

"""net = resnet_model.get_layer('activation_46').output
net = MaxPooling2D((7, 7), name='max_pool')(net)
net = Flatten(name='flatten')(net)

p_drop = 0.5
net = Dense(512, name='fc1')(net)
#net = BatchNormalization()(net)
net = PReLU()(net)
net = Dropout(rate=p_drop)(net)"""

net = resnet_model.output
net = Flatten(name='flatten')(net)
net = Dense(512, activation='relu', name='fc1')(net)
net = Dense(512, name='embded')(net)
net = Lambda(l2Norm, output_shape=[512])(net)


base_model = Model(resnet_model.input, net, name='resnet_model')
base_model.summary()

""" Train just the new layers, let the pretrained ones be as they are (they'll be trained later) """
for layer in resnet_model.layers:
    layer.trainable = False


""" Building triple siamese architecture """

input_shape=(224,224,3)
input_anchor = Input(shape=input_shape, name='input_anchor')
input_positive = Input(shape=input_shape, name='input_pos')
input_negative = Input(shape=input_shape, name='input_neg')

net_anchor = base_model(input_anchor)
net_positive = base_model(input_positive)
net_negative = base_model(input_negative)

positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])

stacked_dists = Lambda( 
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists'
)([positive_dist, negative_dist])

model = Model([input_anchor, input_positive, input_negative], [net_anchor,net_positive, net_negative], name='gen')

model.load_weights('./model_weights.h5')

evaluate.test(base_model)
