
import keras
keras.__version__
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # +1 added to avoid 0/0 division
    return (2.0 * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-5)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
def dice_coef_loss_b(y_true, y_pred):
    return 100*(binary_crossentropy(y_true, y_pred)+(1 - dice_coef(y_true, y_pred)))
def dice_coef_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))

imgs_utrecht = np.load('train/ori/utrecht_flair(256)aug.npy',allow_pickle=True)
mask_utrecht = np.load('train/ori/utrecht_mask(256)aug.npy',allow_pickle=True)

imgs_amsterdam = np.load('train/ori/amsterdam_flair(256)aug.npy',allow_pickle=True)
mask_amsterdam = np.load('train/ori/amsterdam_mask(256)aug.npy',allow_pickle=True)

imgs_singapore = np.load('train/ori/singapore_flair(256)aug.npy',allow_pickle=True)
mask_singapore = np.load('train/ori/singapore_mask(256)aug.npy',allow_pickle=True)


x1=0
x2 =100

print(mask_singapore.shape)
train_img_ut = imgs_utrecht[x1:x2,:,:,0:32]
train_mask_ut = mask_utrecht[x1:x2,:,:,0:32]
#
train_img_ut2 = imgs_utrecht[x1:x2,:,:,32:64]
train_mask_ut2 = mask_utrecht[x1:x2,:,:,32:64]
#
train_img_ut3 = imgs_utrecht[x1:x2,:,:,64:96]
train_mask_ut3 = mask_utrecht[x1:x2,:,:,64:96]

train_img_ut4 = imgs_utrecht[x1:x2,:,:,96:128]
train_mask_ut4 = mask_utrecht[x1:x2,:,:,96:128]
train_img_set= np.concatenate((train_img_ut,train_img_ut2,train_img_ut3,train_img_ut4), axis=0)
train_mask_set = np.concatenate((train_mask_ut,train_mask_ut2,train_mask_ut3,train_mask_ut4),axis=0)
del imgs_utrecht,mask_utrecht,train_img_ut,train_img_ut2,train_img_ut3,train_img_ut4,train_mask_ut,train_mask_ut2,train_mask_ut3,train_mask_ut4

train_img_am = imgs_amsterdam[x1:x2,:,:,0:32]
train_mask_am = mask_amsterdam[x1:x2,:,:,0:32]
##
train_img_am2 = imgs_amsterdam[x1:x2,:,:,32:64]
train_mask_am2 = mask_amsterdam[x1:x2,:,:,32:64]

train_img_am3 = imgs_amsterdam[x1:x2,:,:,64:96]
train_mask_am3 = mask_amsterdam[x1:x2,:,:,64:96]

train_img_am4 = imgs_amsterdam[x1:x2,:,:,96:128]
train_mask_am4 = mask_amsterdam[x1:x2,:,:,96:128]

train_img_set= np.concatenate((train_img_set,train_img_am,train_img_am2,train_img_am3,train_img_am4), axis=0)
train_mask_set= np.concatenate((train_mask_set,train_mask_am,train_mask_am2,train_mask_am3,train_mask_am4), axis=0)
del imgs_amsterdam,mask_amsterdam,train_img_am,train_img_am2,train_img_am3,train_img_am4,train_mask_am,train_mask_am2,train_mask_am3,train_mask_am4

train_img_si = imgs_singapore[x1:x2,:,:,0:32]
train_mask_si = mask_singapore[x1:x2,:,:,0:32]
train_img_si2 = imgs_singapore[x1:x2,:,:,32:64]
train_mask_si2 = mask_singapore[x1:x2,:,:,32:64]
train_img_si3 = imgs_singapore[x1:x2,:,:,64:96]
train_mask_si3 = mask_singapore[x1:x2,:,:,64:96]
train_img_si4 = imgs_singapore[x1:x2,:,:,96:128]
train_mask_si4 = mask_singapore[x1:x2,:,:,96:128]

train_img_set=np.expand_dims(np.concatenate((train_img_set,train_img_si,train_img_si2,train_img_si3,train_img_si4), axis=0),4)
train_mask_set= np.expand_dims(np.concatenate((train_mask_set,train_mask_si,train_mask_si2,train_mask_si3,train_mask_si4), axis=0),4)
del imgs_singapore,mask_singapore,train_mask_si,train_mask_si2,train_mask_si3,train_mask_si4,train_img_si,train_img_si2,train_img_si3,train_img_si4
from tensorflow.keras.utils import to_categorical
train_mask_set = to_categorical(train_mask_set, num_classes=3)


print(np.mean(train_img_set))
##### Always standardized the image arrays for neural networks #####
train_img_set -= np.mean(train_img_set)
print(np.std(train_img_set))
train_img_set /= np.std(train_img_set)
print(np.amax(train_img_set))



epochs=70
lr = 0.001
batch_size=4

from sklearn.model_selection import train_test_split

# majority of original images are used for validation.
train_img1, val_img1, train_mask1, val_mask1 = train_test_split(
    train_img_set, train_mask_set, test_size=0.05, random_state=48)
del train_img_set,train_mask_set
from sklearn.utils import shuffle

train_shuffled, train_mask_shuffled = shuffle(train_img1, train_mask1, random_state=12)
val_shuffled, val_mask_shuffled = shuffle(val_img1, val_mask1, random_state=12)
del train_img1,train_mask1,val_img1,val_mask1

################################### Build 3D model architecture (memory intensive!) ##############################

#
# model = aspp_unet(n_channel=24)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam, SGD

from tensorflow.keras.utils import Sequence


class TrainDataGenerator(Sequence):
    def __init__(self, images, masks, batch_size):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_masks = self.masks[index * self.batch_size:(index + 1) * self.batch_size]

        # 随机垂直翻转
        for i in range(len(batch_images)):
            if np.random.rand() > 0.5:
                batch_images[i] = np.flip(batch_images[i], axis=0)
                batch_masks[i] = np.flip(batch_masks[i], axis=0)

        # # 随机水平翻转
        for i in range(len(batch_images)):
            if np.random.rand() > 0.5:
                batch_images[i] = np.flip(batch_images[i], axis=1)
                batch_masks[i] = np.flip(batch_masks[i], axis=1)

        # # 随机深度方向翻转
        for i in range(len(batch_images)):
            if np.random.rand() > 0.5:
                batch_images[i] = np.flip(batch_images[i], axis=2)
                batch_masks[i] = np.flip(batch_masks[i], axis=2)

        for i in range(len(batch_images)): 
            if np.random.rand() > 0.5:
                batch_images[i] = batch_images[i].transpose(1,0,2,3)
                batch_masks[i] = batch_masks[i].transpose(1,0,2,3)
        return np.array(batch_images), np.array(batch_masks)
        # for i in range(len(batch_images)):
        #     if np.random.rand() > 0.8:
        #         batch_images[i] = np.roll(batch_images[i], 128, axis=0)
        #         batch_masks[i] = np.roll(batch_masks[i], 128, axis=0)
        # for i in range(len(batch_images)):
        #     if np.random.rand() > 0.8:
        #         batch_images[i] = np.roll(batch_images[i], 128, axis=1)
        #         batch_masks[i] = np.roll(batch_masks[i], 128, axis=1)
        for i in range(len(batch_images)):
            if np.random.rand() > 0.8:
                batch_images[i] = np.roll(batch_images[i], 16, axis=2)
                batch_masks[i] = np.roll(batch_masks[i], 16, axis=2)

class DataGenerator(Sequence):
    def __init__(self, images, masks, batch_size):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_masks = self.masks[index * self.batch_size:(index + 1) * self.batch_size]
        return batch_images, batch_masks


train_generator = TrainDataGenerator(train_shuffled, train_mask_shuffled, batch_size=batch_size)
val_generator = DataGenerator(val_shuffled, val_mask_shuffled, batch_size=batch_size)



strategy = tf.distribute.MirroredStrategy()
from unets import *
# Define the scope of the code to be within the strategy
with strategy.scope():
    # Compile the model within the strategy
    model = 3DSAUNet()

    
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss_b, metrics=[dice_coef])
savename="3dsaunet.h5"
def step_decay(epoch):
    print("lr is:", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)
print_lrate = LearningRateScheduler(step_decay)
lrate =ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
                                      cooldown=0, min_lr=1e-8)
model_save =ModelCheckpoint(filepath="3dsaunet_70.h5", verbose=0)
best_weight = ModelCheckpoint(filepath=savename, verbose=1, monitor='val_loss', mode='auto',
                          save_best_only=True)



hist = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[lrate, print_lrate, model_save, best_weight])
