#In[1]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
import os
from utilities import rle2mask , mask2rle

#%matplotlib inline


# %%
# data containing defect images with segmentation mask
defect_class_mask_df = pd.read_csv('train.csv')
# %%
# data containing defective and non defective images
all_images_df = pd.read_csv('defect_and_no_defect.csv')

# %%
train_dir = 'train_images/'




#%%
basemodel = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256,256,3)))
for layer in basemodel.layers:
  layers.trainable = False

headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(1, activation = 'sigmoid')(headmodel)

model = Model(inputs = basemodel.input, outputs = headmodel)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with least validation loss
checkpointer = ModelCheckpoint(filepath="weights.keras", verbose=1, save_best_only=True)

#%%
model_json = model.to_json()
with open("resnet-classifier-model.json","w") as json_file:
  json_file.write(model_json)



# %%
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(defect_class_mask_df, test_size=0.2)

# %%
#creating separate list for imageId, classId and rle to pass into the generator

train_ids = list(X_train.ImageId)
train_class = list(X_train.ClassId)
train_rle = list(X_train.EncodedPixels)

val_ids = list(X_val.ImageId)
val_class = list(X_val.ClassId)
val_rle = list(X_val.EncodedPixels)

#%%
from utilities import DataGenerator

#creating image generators

training_generator = DataGenerator(train_ids,train_class, train_rle, train_dir)
validation_generator = DataGenerator(val_ids,val_class,val_rle, train_dir)
# %%
def resblock(X, f):


  # making a copy of input
  X_copy = X
  X = Conv2D(f, kernel_size = (1,1), strides = (1,1), kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)

  X = Conv2D(f, kernel_size = (3,3), strides =(1,1), padding = 'same', kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)

  # Short path
  # Read more here: https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

  X_copy = Conv2D(f, kernel_size = (1,1), strides =(1,1), kernel_initializer ='he_normal')(X_copy)
  X_copy = BatchNormalization()(X_copy)

  # Adding the output from main path and short path together

  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X






# %%
from keras.layers import UpSampling2D, concatenate

def upsample_concat(x, skip):
  """
  This function upsamples the input tensor x and concatenates it with the skip connection tensor.
  """
  x = UpSampling2D((2,2))(x)
  merge = concatenate([x, skip], axis=3)
  return merge

input_shape = (256,256,1)

#Input tensor shape
X_input = Input(input_shape)

#Stage 1
conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(X_input)
conv1_in = BatchNormalization()(conv1_in)
conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(conv1_in)
conv1_in = BatchNormalization()(conv1_in)
pool_1 = MaxPool2D(pool_size = (2,2))(conv1_in)

#Stage 2
conv2_in = resblock(pool_1, 32)
pool_2 = MaxPool2D(pool_size = (2,2))(conv2_in)

#Stage 3
conv3_in = resblock(pool_2, 64)
pool_3 = MaxPool2D(pool_size = (2,2))(conv3_in)

#Stage 4
conv4_in = resblock(pool_3, 128)
pool_4 = MaxPool2D(pool_size = (2,2))(conv4_in)

#Stage 5
conv5_in = resblock(pool_4, 256)

#Upscale stage 1
up_1 = upsample_concat(conv5_in, conv4_in)
up_1 = resblock(up_1, 128)

#Upscale stage 2
up_2 = upsample_concat(up_1, conv3_in)
up_2 = resblock(up_2, 64)

#Upscale stage 3
up_3 = upsample_concat(up_2, conv2_in)
up_3 = resblock(up_3, 32)

#Upscale stage 4
up_4 = upsample_concat(up_3, conv1_in)
up_4 = resblock(up_4, 16)

#Final Output
output = Conv2D(4, (1,1), padding = "same", activation = "sigmoid")(up_4)

model_seg = Model(inputs = X_input, outputs = output )



# %%
from utilities import focal_tversky, tversky_loss, tversky
# %%
adam = tf.keras.optimizers.Adam(learning_rate = 0.05, epsilon = 0.1) # Changed 'lr' to 'learning_rate'
model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])
# %%
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
# Changed the file extension to '.keras'
checkpointer = ModelCheckpoint(filepath="resunet-segmentation-weights.keras", verbose=1, save_best_only=True)
# %%
# save the model for future use

model_json = model_seg.to_json()
with open("resunet-segmentation-model.json","w") as json_file:
  json_file.write(model_json)
# %%
from utilities import focal_tversky, tversky_loss, tversky # Make sure this line is present

with open('resunet-segmentation-model.json', 'r') as json_file:
    json_savedModel= json_file.read()

# load the model with custom_objects
model_seg = tf.keras.models.model_from_json(json_savedModel, custom_objects={'focal_tversky': focal_tversky})

model_seg.load_weights('weights_seg.hdf5')
adam = tf.keras.optimizers.Adam(learning_rate = 0.05, epsilon = 0.1) # Changed 'lr' to 'learning_rate' as it was defined in a previous code block.
model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])
# %%
test_df = pd.read_csv('test.csv')
# %%
from utilities import prediction
# make prediction
image_id, defect_type, mask = prediction(test_df, model, model_seg)













# %%
from tensorflow.keras import Model, callbacks
# Directory and Data loading
train_dir = os.path.join(os.path.expanduser("~"), r"C:\Pranjal")
all_images_df = pd.read_csv(os.path.join(train_dir, "defect_and_no_defect.csv"))

train, test = train_test_split(all_images_df, test_size=0.15)
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.15)
train['label'] = train['label'].astype(str)
train_generator = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(train_dir, 'train_images'),
    x_col="ImageID",
    y_col="label",
    subset="training",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(256, 256)
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(train_dir, 'train_images'),
    x_col="ImageID",
    y_col="label",
    subset="validation",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(256, 256)
)

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory=os.path.join(train_dir, 'train_images'),
    x_col="ImageID",
    y_col=None,
    batch_size=16,
    shuffle=False,
    class_mode=None,
    target_size=(256, 256)
)

# Residual Block
def residual_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    res = layers.Conv2D(filters, (1, 1), padding='same')(res)
    x = layers.Add()([x, res])
    return x

# Downsampling Block
def downsample_block(x, filters):
    x = residual_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

# Upsampling Block
def upsample_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x

# ResUNet Model Architecture
inputs = layers.Input(shape=(256, 256, 3))

# Encoder
s1, p1 = downsample_block(inputs, 64)
s2, p2 = downsample_block(p1, 128)
s3, p3 = downsample_block(p2, 256)
s4, p4 = downsample_block(p3, 512)

# Bridge
b1 = residual_block(p4, 1024)

# Decoder
d1 = upsample_block(b1, s4, 512)
d2 = upsample_block(d1, s3, 256)
d3 = upsample_block(d2, s2, 128)
d4 = upsample_block(d3, s1, 64)

# Output
outputs = layers.GlobalAveragePooling2D()(d4)  # Global average pooling to reduce spatial dimensions
outputs = layers.Dense(2, activation='softmax')(outputs) # Output layer with 1 neuron and sigmoid activation

# Model creation
model = Model(inputs, outputs)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
checkpointer = callbacks.ModelCheckpoint(filepath="resunet_weights.keras", verbose=1, save_best_only=True)



# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks


# Train the model
history = model.fit(
    train_generator,
    epochs=5,  # Adjust the number of epochs as needed
    validation_data=valid_generator,
    callbacks=[earlystopping, checkpointer]  # Include your callbacks
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# %%
