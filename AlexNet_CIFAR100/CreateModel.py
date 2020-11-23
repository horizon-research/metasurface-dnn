#Model from https://github.com/toxtli/alexnet-cifar-10-keras-jupyter/blob/master/alexnet_test1.ipynb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from NoiseLayer import GaussianNoise_new
#from keras.utils import np_utils
#from keras.preprocessing.image import ImageDataGenerator
from keras import __version__ as keras_version
import warnings
warnings.filterwarnings("ignore")


def create_classifier():
    classifier = Sequential()
    
    classifier.add(Conv2D(48, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(32, 32, 3)))
    classifier.add(GaussianNoise_new())
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    classifier.add(BatchNormalization())
    
    classifier.add(Conv2D(96, (3,3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    classifier.add(BatchNormalization())
    
    classifier.add(Conv2D(192, (3,3), activation='relu', padding='same'))
    classifier.add(Conv2D(192, (3,3), activation='relu', padding='same'))
    classifier.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    classifier.add(BatchNormalization())
    
    classifier.add(Flatten())
    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(100, activation='softmax'))
    classifier.compile(optimizer = Adam(lr=5.0e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    return classifier