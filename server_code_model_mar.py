import tensorflow
import numpy as np
import segmentation_models as sm
import matplotlib.pyplot as plt

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from skimage.transform import resize
from skimage.io import imsave

#from segmentation_models.losses import bce_jaccard_loss
#from segmentation_models.metrics import iou_score

sm.set_framework('tf.keras')

img_rows = int(192)
img_cols = int(192)
smooth = 1.

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Loading data

print('-'*20)
print('Loading the data')
print('-'*20)

def load_train_data():
    imgs_train = np.load('imgs_train2021mar.npy')
    masks_train = np.load('masks_train2021mar.npy')
    return imgs_train, masks_train


def load_test_data():
    imgs_test = np.load('imgs_test2021mar.npy')
    return imgs_test


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

# Preprocess data

imgs_train, imgs_mask_train = load_train_data()
imgs_test = load_test_data()

imgs_train = preprocess(imgs_train)
imgs_mask_train = preprocess(imgs_mask_train)

imgs_train = preprocess_input(imgs_train)
imgs_mask_train = preprocess_input(imgs_mask_train)


#imgs_train /= imgs_train.max()
#imgs_train = imgs_train.astype('float32')

#imgs_mask_train /= imgs_mask_train.max()
#imgs_mask_train = imgs_mask_train.astype('float32')

imgs_test = load_test_data()

# Defining the model

N = imgs_train.shape[-1] # number of channels

base_model = Unet('resnet34', encoder_weights='imagenet')

inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)


# Compile the model

model.compile('Adam', loss=binary_crossentropy, metrics=[dice_coef, 'accuracy'])


# Training
print('-'*20)
print('Training the model')
print('-'*20)

my_callbacks = [ModelCheckpoint('weights.h5', monitor='dice_coef', save_best_only=True)] 
#        EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')]
#    model.load_weights('weights.h5')

history = model.fit(
    x=imgs_train,
    y=imgs_mask_train,
    batch_size=32,
    epochs=70,
    verbose=1,
    validation_split=0.2,
    callbacks=[my_callbacks]
)


# Predict
print('-'*20)
print('Predicting')
print('-'*20)

model.load_weights('weights.h5')

imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('imgs_mask_test.npy', imgs_mask_test)

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Model dice coeff')
plt.ylabel('Dice coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('results_transfer_learning.png')




print('-'*20)
print("Finished")
