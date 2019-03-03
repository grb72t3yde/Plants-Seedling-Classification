import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
from glob import glob
import itertools

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

scale = 70
seed  = 7
BS = 75
epochs = 35

def img_get_resize(path_to_img,training_set,training_label,mcount):
    images = glob(path_to_img)
    num = len(images)
    count = 1
    for i in images:
        print(str(count)+'/'+str(num),end = '\r')
        training_set.append(cv2.resize(cv2.imread(i) , (scale,scale) ) )
        if(mcount==1):
            training_label.append( i.split('/')[-2] )
        else:
            training_label.append( i.split('/')[-1] )
        count = count+1
    training_set = np.asarray(training_set)
    if mcount ==1:
        training_label = pd.DataFrame(training_label)
    return training_set,training_label

def img_masked(new_train,training_set):
    getEx = True
    for i in training_set:
        blurr = cv2.GaussianBlur(i,(5,5),0)
        hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
        lower = (25,40,50)
        upper = (75,255,255)
        mask = cv2.inRange(hsv,lower,upper)
        struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
        boolean = mask>0
        new = np.zeros_like(i,np.uint8)
        new[boolean] = i[boolean]
        new_train.append(new)

    new_train = np.asarray(new_train)
    return new_train

def create_model():
    np.random.seed(seed) #make sure same input generate same result ? why whaere it influence
    model = Sequential()

    model.add( Conv2D( filters=64, kernel_size = (5,5),input_shape = (scale,scale,3),activation='relu' ) )
    model.add( BatchNormalization(axis=3) )
    model.add( Conv2D( filters=64, kernel_size = (5,5),input_shape = (scale,scale,3),activation='relu' ) )
    model.add( MaxPooling2D( 2, 2) )
    model.add( BatchNormalization(axis=3) )
    model.add( Dropout(0.1) )

    model.add( Conv2D( filters=128, kernel_size = (5,5),input_shape = (scale,scale,3),activation='relu' ) )
    model.add( BatchNormalization(axis=3) )
    model.add( Conv2D( filters=128, kernel_size = (5,5),input_shape = (scale,scale,3),activation='relu' ) )
    model.add( MaxPooling2D( 2, 2) )
    model.add( BatchNormalization(axis=3) )
    model.add( Dropout(0.1) )

    model.add( Conv2D( filters=256, kernel_size = (5,5),input_shape = (scale,scale,3),activation='relu' ) )
    model.add( BatchNormalization(axis=3) )
    model.add( Conv2D( filters=256, kernel_size = (5,5),input_shape = (scale,scale,3),activation='relu' ) )
    model.add( MaxPooling2D( 2, 2) )
    model.add( BatchNormalization(axis=3) )
    model.add( Dropout(0.1) )

    model.add(Flatten())
    model.add( Dense(256,activation='relu') )
    model.add( BatchNormalization() )
    model.add( Dropout(0.5) )

    model.add( Dense(12, activation = 'softmax') )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_img(H):
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N),H.history["acc"],label = "train_acc")
    plt.plot(np.arange(0,N),H.history["val_acc"],label = "val_acc")
    plt.plot(np.arange(0,N),H.history["loss"],label = "train_loss")
    plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.legend(loc="lowerleft")
    plt.savefig("plot.png")

def get_callbacks(filepath,patience = 5):
        lr_reduce = ReduceLROnPlateau(monitor = "val_acc",factor = 0.4, patience = 3,verbose = 1,min_lr=0.00001)
        msave = ModelCheckpoint(filepath,monitor = "val_acc",verbose = 1,save_best_only = True,mode = 'max')
        return [lr_reduce,msave]

def main():

    '''image blur->hsv->masked'''
    path = './dataSet/train/*/*.png'
    training_set = []
    training_label = [];
    training_set,training_label = img_get_resize(path,training_set,training_label,1)
    new_train = []
    new_train = img_masked(new_train,training_set)
   # print(training_label)

    '''label -> ont-hot-encoding'''
    labels = preprocessing.LabelEncoder()
    labels.fit(training_label[0])
    encodedlabels = labels.transform(training_label[0]) #change to number
    clearalllabels = np_utils.to_categorical(encodedlabels)# one-hot encoding
    classes = clearalllabels.shape[1]

    '''training'''
    new_train = new_train/255
    (trainX,testX ,trainY ,testY) = train_test_split(new_train,clearalllabels,test_size = 0.1,random_state = seed,stratify=clearalllabels)
    generator = ImageDataGenerator(rotation_range = 180,zoom_range=0.1,width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,vertical_flip = True)
    generator.fit(trainX)
    model = create_model()
   # model.load_weights(filepath='../input/model-weight/model_weight_SGD.hdf5')
    callbacks = get_callbacks('../input/model-weight/model_weight_SGD.hdf5',patience=6)
    H = model.fit_generator(generator.flow(trainX, trainY, batch_size = BS), validation_data = (testX, testY), steps_per_epoch = trainX.shape[0], epochs = epochs,callbacks = callbacks)
    print("saving model to disk")
    #sys.stdout.flush()
    model.save('mymodel.h5')
    plot_img(H)

    '''test_data img deal(masking)'''
    path = './dataSet/test/*.png'
    testimages = []
    test = []
    testimages,test = img_get_resize(path,testimages,test,2)
    new_test = []
    new_test = img_masked(new_test,testimages)

    '''test + write '''
    new_test=new_test/255
    prediction = model.predict(new_test)
    # PREDICTION TO A CSV FILE
    pred = np.argmax(prediction,axis=1)
    predStr = labels.classes_[pred]
    result = {'file':test,'species':predStr}
    result = pd.DataFrame(result)
    result.to_csv("Prediction.csv",index=False)

main()