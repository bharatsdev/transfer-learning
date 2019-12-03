import os

import numpy as np
from imutils import paths
from keras import Input
from keras import Model
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.conv.fcheadnet import FCHeadNet
from preprocessing.aspectawarepreprocessor import AspectAwarePreProcessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simpledatasetloader import SimpleDatasetLoader

batchSize = 32
# Create Image Generator object for data augmentation
print('[INFO] : Initialize the image Generator ....!')
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# grab the list of images that we'll be describing, then extract   the class label names from the image paths
print('[INFO] : Loading Image Form File System...!')
imgDir = 'dataset/flowers17/images'
imagePaths = list(paths.list_images(imgDir))
classNames = [p.split(os.path.sep)[-2] for p in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreProcessor(224, 224)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessor=[aap, iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=21)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a set of FC layers followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model on top of the base model -- this will become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model (this needs to be done after our setting our  layers to being non-trainable
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss=['categorical_crossentropy'], metrics=['accuracy'])

# train the head  of the network for a  few epochs (all other layers are frozen) -- this will allow to the new FC
# layers to start to become initialized with actual  'learned ' values versus pure random

model.fit_generator(aug.flow(trainX, trainY, batch_size=batchSize), epochs=25,
                    verbose=1,
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) / batchSize)

print("[INFO] : Evaluating after initialization...")
predictions = model.predict(testX, batch_size=batchSize)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set  of CONV layers along with our set of FC layers
print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=batchSize),
                    validation_data=(testX, testY), epochs=100, steps_per_epoch=len(trainX) // batchSize, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=batchSize)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=classNames))
# save the model to disk
print("[INFO] : Serializing model...")
model.save('flowers17.model')
# model.summary()
# for (idx, layer) in enumerate(model.layers):
#     print('[INFO] : {} \t {}'.format(idx, layer.__class__.__name__))
