import numpy as np
from PIL import Image
import pickle
import os
import random
import sys
from nn_Feedforward_and_BackProp import L_layer_neural_network_model, predict

learning_rate = 0.001
num_iterations = 3000
train_model = False
Labels_known = True

train_path = '/home/rakumar/NN/train/'
test_path =  '/home/rakumar/NN/TEST/'
model_save_path = '/home/rakumar/NN/model_parameters.clf'

if train_model:
    train_dir = train_path
    train_list = sorted(os.listdir(train_dir))

    data = []
    for imgdir in train_list:
        images = sorted(os.listdir(train_dir+imgdir))
        label = imgdir
        for image in images:
            data.append((train_dir+imgdir+'/'+image, label))

    imgs=[]
    lbs=[]
    for eachdata in data:
        im = np.asarray(Image.open(eachdata[0]))
        im = im.reshape((im.shape[0]*im.shape[1], -1))
        imgs.append(im)

        l = np.zeros((10,1))
        l[int(eachdata[1])] = 1
        lbs.append(np.array(l))

    X = np.array(imgs[0])
    Y = np.array(lbs[0])
    for i in range(1, len(imgs)):
        X = np.append(X, imgs[i], axis=1)
        Y = np.append(Y, lbs[i], axis=1)
        
    # Y = np.array([ [0], [0], [1], [0], [0], [0], [0], [0], [0], [0] ])

    layers_dims = [28*28, 500, 180, 10]

    parameters = L_layer_neural_network_model(X, Y, layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost=True)

    # Training accuracy
    totalCorrectCount = 0
    totalImgs = len(imgs)
    for i in range(len(imgs)):
        pred, prob = predict(imgs[i], parameters)
        if  pred == np.argmax(lbs[i]):
            totalCorrectCount += 1
    accuracy = totalCorrectCount/totalImgs
    print("Training accuracy = ", accuracy)

    with open(model_save_path, 'wb') as f:
                pickle.dump(parameters, f)

with open(model_save_path, 'rb') as f:
            parameters = pickle.load(f)

# predict
images_dir = sorted(os.listdir(test_path))

totalCorrectCount = 0
totalImgs = len(images_dir)
for i, img in enumerate(images_dir):
    im = Image.open(test_path+img)
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.asarray(im)
    im = im.reshape((im.shape[0]*im.shape[1], -1))
    if Labels_known:
        label = np.zeros((10,))
        label[ int(img.split('_')[0]) ] = 1
        pred, prob = predict(im, parameters, y=label)
        if label[pred] == 1:
            totalCorrectCount += 1 
    else:
        pred, prob = predict(im, parameters)
    
    print(img.split('/')[-1]+" ==> "+str(pred) + '  (prob = '+ str(prob) + ')')

if Labels_known:
    accuracy = totalCorrectCount/totalImgs
    print("Test accuracy = ", accuracy)