'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# path to the model weights files.
weights_path = '../keras/examples/resnet50_weights.h5'
# top_model_weights_path = 'fc_model.h5'
top_model_weights_path = 'bottleneck_fc_res_sjx.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'new_train_cnn/train'
validation_data_dir = 'new_train_cnn/validation'
nb_train_samples = 6000
nb_validation_samples = 2000
# epochs = 50
epochs = 20
batch_size = 20

# build the resnet50 network
model = applications.ResNet50(weights='imagenet', include_top=False, input_shape = (224,224,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# # add the model on top of the convolutional base
# model.add(top_model)

# # CREATE AN "REAL" MODEL FROM VGG16
# # BY COPYING ALL THE LAYERS OF VGG16
# new_model = Sequential()
# for l in model.layers:
#     new_model.add(l)
#
# # CONCATENATE THE TWO MODELS
# new_model.add(top_model)
from keras.models import Model
new_model = Model(input=model.input, output=top_model(model.output))


# set the first 80/142 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# for layer in new_model.layers[:80]:
for layer in new_model.layers[:142]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
new_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
new_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=epochs,
    shuffle=True,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

new_model.save_weights('first_res_sjx.h5')

