"""## TASK 3. Change the **kernel size** to 5*5, redo the experiment, plot the learning errors along with the epoch, and report the testing error and accuracy on the test set."""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
###  CHANGING KERNAL SIZE TO 5*5  ###
model.add(Conv2D(6, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
###  CHANGING KERNAL SIZE TO 5*5  ###
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

# https://keras.io/optimizers/ 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.95, decay=0.0),
              metrics=['accuracy'])

Train_mdl1 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

### PLOT FOR TASK 3  ###
from matplotlib import pyplot as plt
train = []
test = []

for item in Train_mdl1.history['loss']:
  train.append(item)
for item in Train_mdl1.history['val_loss']:
  test.append(item)

plt.plot(train)
plt.plot(test)
plt.title('3. Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train - loss', 'Test - val_loss'], loc='upper left')
plt.show()

train = []
test = []

for item in Train_mdl1.history['accuracy']:
  train.append(item)
for item in Train_mdl1.history['val_accuracy']:
  test.append(item)

plt.plot(train)
plt.plot(test)
plt.title('3. Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Train - accuracy', 'Test - val_accuracy'], loc='upper left')
plt.show()