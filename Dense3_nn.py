#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''
from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import six.moves
from load_data_nn import load_motion_data
import random
import numpy as np
import sys
import time
from keras import backend as K
def NNmodel(mom,node_list=[]):
    np.random.seed(1024)  # for reproducibility
    #Load data
    data, label = load_motion_data()
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    nodelist = node_list

    #print (nodelist)


    label = np_utils.to_categorical(label, 3) #modify 10 to 3




    model = Sequential()
    result = open("record_NDense"+str(len(node_list))+"_MO"+str(mom)+"_NN.txt", 'a')


    for node in range(0,len(node_list),1):
        #print(str(node+1)+" DENSE "+ str(nodelist[node])  +" node")
        if node ==0:
            model.add(Dense(int(nodelist[node]), init='normal', input_dim=13))
        elif node ==6:
            model.add(Dense(int(nodelist[node-1]), init='identity', trainable=False))
        else:
            model.add(Dense(int(nodelist[node]), init='normal'))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))


    model.add(Dense(3, init='normal')) #10 ->3
    model.add(Activation('softmax'))


    sgd = SGD(lr=0.05, decay=1e-6, momentum=mom, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



    outstring =""
    for i in range(0,len(node_list),1):
        outstring += str(i+1)+" Dense" +" " +str(nodelist[i])+"\n"
    result.write(outstring)

    start = time.time()
    val_acclist =[]
    down_count =0
    convergence_flag = 0
    max_value = 0
    return_accuracy = 0
    for i in range(0, 20, 1):
        hist = model.fit(data, label, nb_epoch=1, batch_size=100, shuffle=True, verbose=0, validation_split=0.3,
                 )

        val_acclist.append (hist.history['val_acc'][0])
        #print (val_acclist)
        if val_acclist[i]>max_value:
            max_value = val_acclist[i]
        #計算收斂
        if i >4:
            convergence_flag =abs(val_acclist[i]-val_acclist[i-1])+\
                              abs(val_acclist[i]-val_acclist[i-2])+\
                              abs(val_acclist[i]-val_acclist[i-3])+\
                              abs(val_acclist[i]-val_acclist[i-4])
        #準確度連摔4次
        if i >0 :
            if val_acclist[i] < val_acclist[i-1] :
                down_count += 1
            else:
               down_count = 0

        if down_count >=4:
            result.write ("down_count is match 4\n")
            result.write(str(val_acclist[i-4])+"\n")
            result.write("max_value: "+str(max_value)+"\n")
            return_accuracy = val_acclist[i-4]
            break
        elif convergence_flag<0.05:
            if  i>4:
                result.write("accuracy is convergence\n")
                result.write(str(val_acclist[i]) + "\n")
                result.write("max_value: " + str(max_value) + "\n")
                return_accuracy = val_acclist[i]
                break
        elif i==19:
            result.write("END 20\n")
            result.write(str(val_acclist[i]) + "\n")
            result.write("max_value: " + str(max_value) + "\n")
            return_accuracy = val_acclist[i]
            break
 #   for i in range(0,20,1):
 #       result.write("val_acc:"+str(hist.history['val_acc'][i])+"\n")

    end = time.time()
    excutetime = end-start
    result.write("excutetime:"+str(excutetime)+"\n")
    result.close()
    return  return_accuracy

