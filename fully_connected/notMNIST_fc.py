
## Libraries
from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range
from time import time
import random
import numpy as np
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
import keras.callbacks as kcb

def norm_input(x): return (x-mean_px)/std_px

## Load data
pickle_file = './DL Udacity/notMNIST_cln.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

## Reformat for keras input
def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size * image_size)).astype('float32')
  labels = (np.arange(num_labels) == labels[:,None]).astype('float32')
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
mean_px, std_px = train_dataset.mean(), train_dataset.std()
print('After reformatting,')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# define vars
input_num_units = image_size * image_size
hidden_num_units = 512

# create model
model = Sequential([
  Lambda(norm_input, input_shape=(input_num_units, )),
  Dense(hidden_num_units, init='he_normal'),
  BatchNormalization(),
  Activation('relu'),
  Dropout(0.5),
  Dense(hidden_num_units, init='he_normal'),
  BatchNormalization(),
  Activation('relu'),
  Dropout(0.5),
  Dense(num_labels, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Training!!
class CallMetric(kcb.Callback):
    def on_train_begin(self, logs={}):
        self.best_acc = 0.0
        self.accs = []
        self.val_accs = []
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
    	self.accs.append(logs.get('acc'))
    	self.val_accs.append(logs.get('val_acc'))
    	self.losses.append(logs.get('loss'))
    	self.val_losses.append(logs.get('val_loss'))
    	if logs.get('val_acc') > self.best_acc:
    		self.best_acc = logs.get('val_acc')
    		print("\nThe BEST val_acc to date.")

epochs = 12
batch_size = 64

print("Start training...")
t0 = time()
metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath="./fc_4l.h5", monitor='val_acc', verbose=1, save_best_only=True)
trained_model = model.fit(train_dataset, train_labels, nb_epoch=epochs, batch_size=batch_size, 
                          callbacks=[metricRecords, checkpointer], 
                          validation_data=(valid_dataset, valid_labels))
print("\nElapsed time:", time()-t0, 'seconds\n\n')


model.load_weights('./fc_4l.h5')
randSample = random.sample(np.arange(train_dataset.shape[0]), 20000)
pred_tr = model.predict_classes(train_dataset[randSample])
print("train accuracy:", np.mean(pred_tr==np.argmax(train_labels[randSample], 1)))

pred_val = model.predict_classes(valid_dataset)
print("validation accuracy:", np.mean(pred_val==np.argmax(valid_labels, 1)))

pred_test = model.predict_classes(test_dataset)
print("test accuracy:", np.mean(pred_test==np.argmax(test_labels, 1)))
print('\n')

# Save performance data figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(np.arange(epochs)+1, metricRecords.accs, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_accs, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('fc_4l_acc.png')

plt.plot(np.arange(epochs)+1, metricRecords.losses, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_losses, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Log loss')
plt.legend(loc='lower right')
plt.savefig('fc_4l_loss.png')


# no augmentation
# 1276.57 sec, tr_acc=.9445, val_acc=.9052, test_acc=.9547 (layers: 784-2048-10) (12 epochs)
# 1626.12 sec, tr_acc=.9188, val_acc=.9045, test_acc=.9532 (layers: 784-1024-1024-10) (12 epochs)

