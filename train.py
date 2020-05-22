import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
import sys
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import BatchNormalization

# load train and test dataset
def load_dataset():
    (trainX, trainy) , (testX, testy) = cifar10.load_data()
    ''' There are 10 classes and they are represented as unique integers, 
    so we perform one hot encoding to convert labels into a 10 element binary vector. '''
    trainY = np_utils.to_categorical(trainy)
    testY = np_utils.to_categorical(testy)
    return trainX, trainY, testX, testY

def prepare_pixels(train, test):
    #The pixel values need to be scaled 
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #nomralising
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    return train_norm, test_norm

def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

def define_model():
    #Here we can define a 3-block VGG style architecture
    model = Sequential()
    '''
    The architecture involves stacking convolutional layers of size 3x3 followed by
    a max-pooling layer. These blocks of convolutions are formed and can be repeated 
    where the dimensions of each block can be increased
    Padding is used so as to avoid loss of information
    '''
    model.add(Conv2D(32,(3,3), activation ='relu', kernel_initializer='he_uniform', padding ='same', input_shape = (32, 32, 3)))
    model.add(Conv2D(32,(3,3), activation ='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), activation ='relu', kernel_initializer='he_uniform', padding ='same'))
    model.add(Conv2D(64,(3,3), activation ='relu', kernel_initializer='he_uniform', padding ='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128,(3,3), activation ='relu', kernel_initializer='he_uniform', padding ='same'))
    model.add(Conv2D(128,(3,3), activation ='relu', kernel_initializer='he_uniform', padding ='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))

    '''
    Coupling with the feature interpeter which assigns which label does the image belongs to
    Perform Flattening of the features before interpretation

    '''
    model.add(Flatten())
    model.add(Dense(128, activation ='relu', kernel_initializer ='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    #COMPILE MODEL
    '''
    The model was optimised using a Stochastic Gradient Descent optimiser
    The learning rate was set as 0.001
    and momentum as 0.9
    Since this is a multi-class classification we will used categorical crossentropy loss

    '''          
    opt = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])          
    return model

def train():
    #Here we will load our datasets
    trainX, trainY, testX, testY = load_dataset()
    print('dataset loaded')
    #Now we will prepare the pixels
    trainX, testX = prepare_pixels(trainX, testX)
    #model definition
    model = define_model()
    #Fit the model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX,testY))
    # save model
    model.save('final_model.h5')


train()