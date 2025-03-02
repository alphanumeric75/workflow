import tensorflow  as tf
import numpy as np 

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train ),(x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
x_train , x_test = x_train / 255.0 , x_test / 255.0

sample_size = int(0.5 * x_train.shape[0])
index = np.random.choice(x_train.shape[0], sample_size) 
x_train_small , y_train_small = x_train[index] , y_train[index]
# x_train_small = tf.image.resize(x_train_small ,(128,128))
# x_test = tf.image.resize(x_test,(128 ,128))

data_gen = ImageDataGenerator(
    rotation_range = 15, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1, 
    horizontal_flip = True , 
    zoom_range = 0.2
)
data_gen.fit(x_train_small) 

base_model = MobileNetV2(
    weights = 'imagenet', 
    include_top = False, 
    input_shape  = (32,32,3)
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs = base_model.input, outputs = x)
model.compile(
    optimizer = 'Adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics  = ['accuracy']
)

model.fit(data_gen.flow(x_train_small, y_train_small,batch_size = 64), epochs = 50, validation_data = (x_test, y_test)) 
