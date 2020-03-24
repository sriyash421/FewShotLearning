import pickle
import random
import numpy as np
import cv2
import os
import tqdm

dict = {}
with open("cifar-10-batches-py/data_batch_1",'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

images = {}
for i in tqdm.trange(dict[b'data'].shape[0]) :
    label = dict[b'labels'][i]
    image = dict[b'data'][i][:]
    r = image[0:1024].reshape(32,32)
    g = image[1024:2048].reshape(32,32)
    b = image[2048:].reshape(32,32)
    temp = np.zeros((32,32,3))
    temp[:,:,0] = (r-r.mean())/(r.std())*0.5+0.5
    temp[:,:,1] = (g-g.mean())/(g.std())*0.5+0.5
    temp[:,:,2] = (b-b.mean())/(b.std())*0.5+0.5
    if label not in images.keys() :
        images[label] = []
    images[label].append(temp)

for i in tqdm.trange(10000) :
    labels = random.sample(range(10),5)
    for j in range(5) :
        input_label = labels[0]
        labels.pop()
        dir_path = os.path.join("data","{}".format(5*i+j)) 
        os.mkdir(dir_path)
        output = [(random.choice(images[input_label]),input_label)]
        output.append((random.choice(images[input_label]),input_label))
        for k in labels :
            output.append((random.choice(images[k]),k))
        flag = 1
        for k in output :
            im_name = os.path.join(dir_path,"{}_{}.jpg".format(k[1],'I' if flag else 'S'))
            cv2.imwrite(im_name, k[0])
            flag = 0
            labels.append(input_label)