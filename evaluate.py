import numpy as np
import os
import operator
from collections import OrderedDict

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

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
