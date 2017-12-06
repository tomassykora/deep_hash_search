from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras import backend as K
from keras.models import Model
from keras.layers import AveragePooling2D, GlobalMaxPooling2D, Lambda, Input, Flatten, Dense
from keras import optimizers

import numpy as np
from triplets_generator import DataGenerator

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def triplet_loss(_, y_pred):
	margin = K.constant(1)
	#return K.mean(K.square(y_pred[0]) - K.square(y_pred[1]) + margin)
	return K.mean(K.maximum(K.constant(0), K.square(y_pred[0]) - K.square(y_pred[1]) + margin))

def accuracy(_, y_pred):
	return K.mean(y_pred[0] < 0.4 and y_pred[1] > 0.6)


""" Building the resnet feature map model """

K.set_image_dim_ordering('tf')

base_model = ResNet50(weights='imagenet', include_top = True)
#base_model.summary()

net = base_model.get_layer('activation_40').output
net = AveragePooling2D((7, 7), name='avg_pool')(net)
net = Flatten(name='flatten')(net)
net = Dense(256, activation='relu', name='fc')(net)
net = Lambda(l2Norm, output_shape=[256])(net)


model = Model(base_model.input, net, name='new_model')
model.summary()

""" Building triple siamese architecture """

input_shape=(224,224,3)
input_anchor = Input(shape=input_shape, name='input_anchor')
input_positive = Input(shape=input_shape, name='input_pos')
input_negative = Input(shape=input_shape, name='input_neg')

net_anchor = model(input_anchor)
net_positive = model(input_positive)
net_negative = model(input_negative)

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
training_generator = DataGenerator(dim_x = 224, dim_y = 224, batch_size = batch_size, dataset_path = './places365-dataset/20_classes').generate()
#validation_generator = DataGenerator(dim_x = 224, dim_y = 224, batch_size = batch_size, dataset_path = './places365-dataset/20_classes').generate()


opt = optimizers.Adam()
model.compile(loss=triplet_loss, optimizer=opt)

"""for i in range(5):
	anchor = np.ones((10,224,224,3))
	positive = np.ones((10,224,224,3))
	negative = np.ones((10,224,224,3))

	model.fit(x=[anchor, positive, negative], y=np.ones((len(anchor),2,1)), batch_size=5, epochs=i+1, initial_epoch=i)"""
model.fit_generator(generator = training_generator,
                    steps_per_epoch = 98500//batch_size)