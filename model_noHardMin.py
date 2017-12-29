from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.models import Model
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, Input, Flatten, Dense
from keras.layers import BatchNormalization, Dropout, PReLU
from keras import optimizers
from keras.models import model_from_json

import os
import numpy as np
from triplets_generator import DataGenerator
import evaluate

def l2Norm(x):
	return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def triplet_loss(_, y_pred):
	margin = K.constant(1)
	return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(_, y_pred):
	return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def mean_pos_dist(_, y_pred):
	return K.mean(y_pred[:,0,0])

def mean_neg_dist(_, y_pred):
	return K.mean(y_pred[:,1,0])

""" Building the resnet feature map model """

K.set_image_dim_ordering('tf')

resnet_input = Input(shape=(224,224,3))
resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=resnet_input)

"""net = resnet_model.get_layer('activation_31').output
net = MaxPooling2D((7, 7), name='max_pool')(net)
#net = Flatten(name='flatten')(net)

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
#base_model.summary()

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

model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')
model.summary()

""" Training """
batch_size = 5
training_generator = DataGenerator(dim_x=224, dim_y=224, batch_size=batch_size, dataset_path='./20_classes').generate()
#validation_generator = DataGenerator(dim_x = 224, dim_y = 224, batch_size = batch_size, dataset_path = './places365-dataset/20_classes').generate()

opt = optimizers.Adam(lr=0.0005)
model.compile(loss=triplet_loss, optimizer=opt, metrics=[accuracy, mean_pos_dist, mean_neg_dist])

model.fit_generator(generator = training_generator,
					steps_per_epoch = 248300//batch_size,
					epochs = 1)

model.save_weights("model_1epoch.h5")
print("Saved model to disk")

model.summary()

opt = optimizers.Adam(lr=0.00004)
model.compile(loss=triplet_loss, optimizer=opt, metrics=[accuracy, mean_pos_dist, mean_neg_dist])
model.fit_generator(generator = training_generator,
					steps_per_epoch = 248300//batch_size,
					epochs = 1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

