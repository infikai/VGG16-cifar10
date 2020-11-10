##train.py
import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

from tensorflow.keras.datasets import cifar10

### 1. Show Pic
## Vis
def vis_pic():
    '''
	Call this function to show ten random picture in cifar-10 train picture with labe.
	Usage:
	vis_pic()
	-> windows show ten pictures.
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    label_dict = {0:"airplain",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    for i in range(0, 10):
        k = random.randint(1,50000)
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(x_train[k], cmap='binary')
        title = str(k) + "." + label_dict[y_train[k][0]] + str(y_train[k])
        ax.set_title(title, fontsize=10)
        plt.axis('off')
    plt.show()

### 2. Show hyper -> set_hyperparameters(1)
## Set hyperparameters
def set_hyperparameters(case):
    '''
	This function to set the parameters use to training.
	If you want to change the value of batch, learning rate, epoch just modify this function.
	Usage
	If you want to show(print) the hyperparameters your have set:
	epoch, batch, opt = set_hyperparameters(1)
	-> will print what you have set.

	If youu just want set:
	epoch, batch, opt = set_hyperparameters(0)
        -> just return the value of (epoch, batch, opt)
    '''
    batch = 32
    lr = 0.001
    epoch = 20
    opt = keras.optimizers.Adam(lr)
    if (case == 1):
        print('Batch size:', batch)
        print('learning rate:', lr)
        print('optimizer: Adam')
    return epoch, batch, opt

### 3. Summary
## Create Model VGG-16
def creating_vgg16():
    '''
	Creating VGG16 model, for cifar-10 the inputs shape is (32,32,3).
	When calling this function, it will show the 'summary' of the miodel.
	Usage:
	creating_vgg16()
	-> (model summary of 'my_VGG16')

    '''
    model = models.Sequential(name="my_VGG16")
    model.add(keras.Input(shape=(32,32,3)))
    #model.add(keras.Input(shape=(100,100,3)))
    # Block 1
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1'))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2), name='block1_pool'))
    # Block 2
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2), name='block2_pool'))
    # Block 3
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2), name='block3_pool'))
    # Block 4
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2), name='block4_pool'))
    # Block 5
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2), name='block5_pool'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,  activation='relu'))
    model.add(layers.Dense(4096,  activation='relu'))
    model.add(layers.Dense(10,  activation='softmax'))

    epoch, batch, opt = set_hyperparameters(0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

### Train
def training():
    '''
	Training model.
	It will detect if there have trained model in folder.
	If have, it will skip training.
	Usage:
	h = training()
	-> return h, it is the history of acc and loss corresbonding to epoch.
    '''

    target_folder = 'saved'

    if os.path.exists(target_folder):
        print('Already trained')
        h_h = np.load('my_history.npy',allow_pickle='TRUE').item()
	return h_h
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        nb_classes = 10
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)
	
        log_dir = 'saved'
        os.mkdir(log_dir)

        model = creating_vgg16()
        epoch, batch, opt = set_hyperparameters(0)
        h = model.fit(x_train, y_train, batch_size=batch, epochs=epoch, validation_data=(x_test, y_test))
        print('Saving model to disk...')
        model.save('saved/whole_model')
	np.save('my_history.npy',h.history)
        return h.history	
    '''
    acc, val_acc = h.history['accuracy'], h.history['val_accuracy']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    history_dict = h.history
    print(history_dict.keys())
    '''
    return h

### 5. Test
def test(num):
    '''
        Testing a picture in test datasets, and show the result in a window.
	The number of 'num' need between 0~9999.
	Usage:
	test(x_test,  10):
	-> (window of result)
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    nb_classes = 10
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    model = models.load_model('saved/whole_model')
    test_img = x_test[num:num+1]
    predicted = model.predict(test_img, verbose=1)
    print(predicted)
    plt.figure(figsize = (17,5))
    plt.subplot(121)
    plt.imshow(x_test[num], cmap='binary')

    plt.subplot(122)
    plt.hist(predicted[0], color = 'blue')
    plt.title('Possibility of item in the picture')
    plt.show

def accuracy_curve(h_h):
    '''
	plot a chart of relation between epoch and acc, loss
	'h' is the history return from model.fit.
	Usage:
        accuracy_curve(h)
        -> (show the chart)
    '''
    acc, loss, val_acc, val_loss = h_h['accuracy'], h_h['loss'], h_h['val_accuracy'], h_h['val_loss']
    epoch = len(acc)
    plt.figure(figsize = (17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('saved/acc_and_loss.png')

### 4. Show Accuracy
def Show_Accuracy():
    '''
	To show the trained-model's accuracy by the chart.
	Usage:
	Show_Accuracy()
	-> (window show result)
    '''
    # data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    nb_classes = 10
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    # parameters
    h = training()
    accuracy_curve(h)

def main(): 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    vis_pic(x_train, y_train)
    nb_classes = 10
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    '''
    ### Datasets for testing code
    x_train = np.random.random((100, 100, 100, 3))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((20, 100, 100, 3))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    '''
    epoch, batch, opt = set_hyperparameters(0)
    h = training()
    Show_Accuracy()
    accuracy_curve(h)
    # 0 < num <= 19
    test(x_test, 10)

if __name__ == '__main__':
    main()

