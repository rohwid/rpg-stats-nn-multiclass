import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


sf_train = pd.read_csv('datasets/dota_2_training.csv')
corr_matrix = sf_train.corr()
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

sf_val = pd.read_csv('datasets/dota_2_validation.csv')
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

train_data = sf_train.values
val_data = sf_val.values

train_x = train_data[:,2:]
val_x = val_data[:,2:]

train_y = to_categorical(train_data[:,1])
val_y = to_categorical(val_data[:,1])

# Create neural network
inputs = Input(shape=(16,))
h_layer = Dense(18, activation='sigmoid')(inputs)

# Softmax activation for multiclass classification
outputs = Dense(3, activation='softmax')(h_layer)

# Initiate the model
model = Model(inputs=inputs, outputs=outputs)

# Optimizer and learning rate
sgd = SGD(lr=0.001)

# Compile the model with cross entropy loss
model.compile(
    optimizer=sgd, 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Print model summary
print(model.summary())

# All keras callbacks
checkpoint = ModelCheckpoint(
    filepath='./dota_2_hero_classification.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=2000,
    verbose=0,
    mode='min',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=0,
    min_delta=0.001,
    min_lr=0.0001
)

tf_board = TensorBoard(log_dir='./tensorboard')

# Call the keras callbacks
callbacks = [checkpoint, early_stop, reduce_lr, tf_board]

# Train the model and use validation data
history = model.fit(
    train_x, 
    train_y, 
    batch_size=16,
    epochs=20000,
    callbacks=callbacks,
    verbose=1,
    validation_data=(val_x, val_y)
)

#
# History - Disabled because better to look at TensorBoard
#

# Summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# Summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# Save the weight as json
model_json = model.to_json()

with open('dota_2_hero_classification.json', 'w') as json_file:
    json_file.write(model_json)

# Predict all validation data
predict = model.predict(val_x)

# Visualize prediction
df = pd.DataFrame(predict)
df.columns = ['Strength', 'Agility', 'Intelligent']
df.index = val_data[:,0]
print(df)