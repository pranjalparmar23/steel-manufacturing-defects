#In[1]
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam



#%%
# Enable mixed precision for faster training (requires TensorFlow 2.4+ and GPU)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# %%
# Paths and Data Preparation
train_dir = r"C:\Pranjal"  # Update with your directory path where images and CSV are located
all_images_df = pd.read_csv(os.path.join(train_dir, "defect_and_no_defect.csv"))

# Split data
train, test = train_test_split(all_images_df, test_size=0.15, random_state=42)
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.15)

# Convert labels to string format for categorical class mode
train['label'] = train['label'].astype(str)

# Adjustable Parameters
batch_size = 32  # Adjust batch size based on GPU memory
learning_rate = 0.001  


# %%
# Image Generators for Train, Validation, and Test sets
train_generator = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(train_dir, 'train_images'),
    x_col="ImageID",
    y_col="label",
    subset="training",
    batch_size=batch_size,
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
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    target_size=(256, 256)
)

test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory=os.path.join(train_dir, 'train_images'),
    x_col="ImageID",
    y_col=None,
    batch_size=batch_size,
    shuffle=False,
    class_mode=None,
    target_size=(256, 256)
)
# %%
# Define Residual Block for ResUNet
def residual_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    res = layers.Conv2D(filters, (1, 1), padding='same')(res)
    x = layers.Add()([x, res])
    return x

# Define Downsampling Block
def downsample_block(x, filters):
    x = residual_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

# Define Upsampling Block
def upsample_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x

# Build the ResUNet Model
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

# Output layer
outputs = layers.GlobalAveragePooling2D()(d4)
outputs = layers.Dense(2, activation='softmax')(outputs)

# Create the model
model = Model(inputs, outputs)


# %%
model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
checkpoint = callbacks.ModelCheckpoint(filepath="resunet_weights.keras", 
                                       verbose=1, save_best_only=True)
# %%
# Train the model
history = model.fit(
    train_generator,
    epochs=1,  # Increase epochs if needed
    validation_data=valid_generator,
    callbacks=[early_stopping, checkpoint]
)











#%%
import tensorflow as tf
print(tf.__version__)

# %%
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

























# %%
import tensorflow as tf

# Check if TensorFlow detects a GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow detected the following GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected by TensorFlow.")

# %%
