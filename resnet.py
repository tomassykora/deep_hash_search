from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.models import Model
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, Input, Flatten, Dense
from keras.layers import BatchNormalization, Dropout, PReLU
from keras import optimizers
import tensorflow as tf
import numpy as np
import os
from triplets_generator import DataGenerator
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import operator
from collections import OrderedDict
def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def euclidean_distanceX(a,b):
    return K.sqrt(K.sum(K.square((a-b)), axis=1))

def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def mean_pos_dist(_, y_pred):
    return K.mean(y_pred[:,0,0])

def mean_neg_dist(_, y_pred):
    return K.mean(y_pred[:,1,0])
def fake_loss(__,_):
    return K.constant(0)
""" Building the resnet feature map model """


def sim_sort(anch,filenames,predictions):
    def euclidean_distance(x, y):
        return np.linalg.norm(x - y)
    sims={}
    for i, candidate in enumerate(predictions):
        sims[filenames[i]]=euclidean_distance(anch, candidate)
    return OrderedDict(sorted(sims.items(), key=lambda t: t[1]))

def AP(sim, class_name, class_len):
    correct=0.0
    prec_sum=0.0
    try:#python2
        ititems=sim.iteritems()
    except:
        ititems=sim.items()
    for i,(file,score) in enumerate(ititems):
        if i == 0: continue
        print (file,score)
        if file.split("/")[0]==class_name:
            correct+=1
            prec_sum+=correct/i
    return (prec_sum/(min(class_len,i)))

def MAP(preds,batch_files):
    sumAP=0
    for pred,file in zip(preds,batch_files):
        print ("similarity for %s:"%file)
        sim = sim_sort(pred, batch_files, preds)
        ap=(AP(sim, file.split("/")[0], 49))
        print (ap)
        sumAP+=ap
    return sumAP/len(batch_files)

def test(model):
    dataset_path="./data_test"
    classes = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    batch=[]
    batch_files = []
    for Id, c in enumerate(classes):
        pos_dir = os.path.join(dataset_path, c)
        imgs = os.listdir(pos_dir)
        for img_path in imgs:
            print (os.path.join(c,img_path))
            img = image.load_img(os.path.join(dataset_path, os.path.join(c,img_path)), target_size=(224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img = np.squeeze(img)
            batch.append(img)
            batch_files.append(os.path.join(c,img_path))
    preds=(model.predict([np.asarray(batch)]))
    print(preds)
    print (preds.shape)
    print (batch_files[0])
    sim=sim_sort(preds[0],batch_files,preds)
    #print (sim)
    print (AP(sim,batch_files[0].split("/")[0],50))
    print ("MAP: %s"%MAP(preds,batch_files))
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
opt = optimizers.Adam(lr=0.0002)

model_generator = Model([input_anchor, input_positive, input_negative], [net_anchor,net_positive, net_negative], name='gen')
model_generator.compile(loss=fake_loss, optimizer=opt)
base_model.compile(loss=fake_loss, optimizer=opt)

model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')
model.summary()

""" Training """
batch_size = 5
graph = tf.get_default_graph()

training_generator = DataGenerator(model_generator,graph,dim_x=224, dim_y=224, batch_size=batch_size, dataset_path='./data_train').generate()
#validation_generator = DataGenerator(dim_x = 224, dim_y = 224, batch_size = batch_size, dataset_path = './places365-dataset/20_classes').generate()


model.compile(loss=triplet_loss, optimizer=opt, metrics=[accuracy, mean_pos_dist, mean_neg_dist])

model.fit_generator(generator = training_generator,
                    steps_per_epoch = 45470/(batch_size),
                    epochs = 1)
test(base_model)