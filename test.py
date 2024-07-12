# import matplotlib.pyplot as plt

import numpy as np
# import pandas as pd

# from nilearn import plotting
import nibabel as nib
import cv2
import os
import glob


path1 = '/data/run01/scw6462/WMH/WMHdataset/test/Utrecht/'
path2 = '/data/run01/scw6462/WMH/WMHdataset/test/Singapore/'

path3 = '/data/run01/scw6462/WMH/WMHdataset/test/Amsterdam/Philips_VU .PETMR_01/'
path4 = '/data/run01/scw6462/WMH/WMHdataset/test/Amsterdam/GE3T/'
path5 = '/data/run01/scw6462/WMH/WMHdataset/test/Amsterdam/GE1T5/'

from unets import *
# from attentionunet import AttUnet3D


model = caunet_aspp()
model.summary()
model.load_weights("models256/caunet_aspp.h5")






mean=39.941482263808105
std=136.52766677127232




def resize_image_and_mask(image, mask, target_size):
    # 获取原始图像和掩膜的尺寸
    original_size = np.array(image.shape)
    # 计算尺寸差异
    size_diff = target_size - original_size

    pad_before = size_diff // 2
    pad_after = size_diff - pad_before
    pad_width = [(pad_before[i], pad_after[i]) for i in range(len(size_diff))]
    resized_image = np.pad(image, pad_width, mode='constant')
    resized_mask = np.pad(mask, pad_width, mode='constant')

    return resized_image, resized_mask


def restore_image_and_mask(resized_image, resized_mask, original_size):
    # 计算尺寸差异
    size_diff = original_size - np.array(resized_image.shape)
    # 计算裁剪的边界
    start = -size_diff // 2
    end = start + original_size
    # 还原图像和掩膜的尺寸
    restored_image = resized_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    restored_mask = resized_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return restored_image, restored_mask

import numpy as np

def crop_image(image,target_size = (256, 256, 256)):

    original_size = image.shape

    # 计算每个维度上需要移除的切片数量
    remove_slices = [(original_size[i] - target_size[i]) // 2 for i in range(3)]

    # 移除切片
    cropped_image = image[
        remove_slices[0]:remove_slices[0]+target_size[0],
        remove_slices[1]:remove_slices[1]+target_size[1],
        remove_slices[2]:remove_slices[2]+target_size[2]
    ]

    return cropped_image

def getAVD(testArray, resultArray):
    testSum = np.sum(testArray)
    resultSum = np.sum(resultArray)
    return float(abs(testSum - resultSum)) / float(testSum) * 100
def calculate_dice_coefficient(mask1, mask2):
    # 将掩膜转换为二进制形式
    mask1_binary = np.zeros_like(mask1, dtype=bool)
    mask2_binary = np.zeros_like(mask2, dtype=bool)

    # 根据标签值设置二进制形式的掩膜
    for label in [1, 2]:
        mask1_binary[mask1 == label] = True
        mask2_binary[mask2 == label] = True

    # 计算每个标签的交集和总和
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    sum_masks = mask1_binary.sum() + mask2_binary.sum()

    # 计算 Dice 系数
    dice_coefficient = (2.0 * intersection) / (sum_masks + 1e-6)  # 添加一个小值以避免分母为零
    return dice_coefficient





all_dice = []
all_avd = []

dice = []
avd = []
data_path = sorted(os.listdir(path1), key=lambda x: int(os.path.splitext(x)[0]))
print(data_path)
for dataindex in data_path:
    image_path = os.path.join(path1,dataindex,'pre/FLAIR.nii.gz')
    mask_path = os.path.join(path1,dataindex,'wmh.nii.gz')
    print(mask_path)

    image_data = nib.load(image_path).get_fdata()


    mask_data = nib.load(mask_path).get_fdata()
    print(mask_data.dtype)
    image_resized, mask_resized = resize_image_and_mask(image_data, mask_data, (256, 256, 128))
    print(np.unique(mask_resized))

    image_resized -= mean
    image_resized /= std
    print(np.amax(image_resized))
    image_resized1 = image_resized[:,:,0:32]
    image_resized1 = np.expand_dims(image_resized1, -1)
    image_resized1 = np.expand_dims(image_resized1, 0)

    image_resized2 = image_resized[ :, :, 32:64]
    image_resized2 = np.expand_dims(image_resized2, -1)
    image_resized2 = np.expand_dims(image_resized2, 0)

    image_resized3 = image_resized[:, :, 64:96]
    image_resized3 = np.expand_dims(image_resized3, -1)
    image_resized3 = np.expand_dims(image_resized3, 0)

    image_resized4 = image_resized[:, :, 96:128]
    image_resized4 = np.expand_dims(image_resized4, -1)
    image_resized4 = np.expand_dims(image_resized4, 0)


    pred1 = model.predict(image_resized1, verbose=1)[0, :, :, :, :]
    pred1 =  np.argmax(pred1, axis=-1)
    print(np.amax(pred1))
    pred2 = model.predict(image_resized2, verbose=1)[0, :, :, :, :]
    pred2 = np.argmax(pred2, axis=-1)
    pred3 = model.predict(image_resized3, verbose=1)[0, :, :, :, :]
    pred3 = np.argmax(pred3, axis=-1)
    pred4 = model.predict(image_resized4, verbose=1)[0, :, :, :, :]
    pred4 = np.argmax(pred4, axis=-1)

    pred = np.concatenate((pred1,pred2,pred3,pred4),axis=-1)

    image,mask = restore_image_and_mask(image_resized,pred,mask_data.shape)
    mask = mask.astype(np.float64)
    dice_score = calculate_dice_coefficient(mask_data,mask)
    avd_score = getAVD(mask_data,mask)
    print(dice_score)
    dice.append(dice_score)
    avd.append(avd_score)
    all_dice.append(dice_score)
    all_avd.append(avd_score)
    output_folder = "results/Utrecht/"+str(dataindex)
    os.makedirs(output_folder, exist_ok=True)
    mask = nib.Nifti1Image(mask, np.eye(4))
    nifti_filename = os.path.join(output_folder, "wmh.nii.gz")
    nib.save(mask, nifti_filename)



print(np.mean(dice),np.std(dice))

#
#
dice = []
avd = []
data_path = sorted(os.listdir(path2), key=lambda x: int(os.path.splitext(x)[0]))
print(data_path)
for dataindex in data_path:
    image_path = os.path.join(path2,dataindex,'pre/FLAIR.nii.gz')
    mask_path = os.path.join(path2,dataindex,'wmh.nii.gz')
    print(mask_path)

    image_data = nib.load(image_path).get_fdata()


    mask_data = nib.load(mask_path).get_fdata()
    image_resized, mask_resized = resize_image_and_mask(image_data, mask_data, (256, 256, 128))
    image_resized -= mean
    image_resized /= std
    image_resized1 = image_resized[:,:,0:32]
    image_resized1 = np.expand_dims(image_resized1, -1)
    image_resized1 = np.expand_dims(image_resized1, 0)

    image_resized2 = image_resized[ :, :, 32:64]
    image_resized2 = np.expand_dims(image_resized2, -1)
    image_resized2 = np.expand_dims(image_resized2, 0)

    image_resized3 = image_resized[:, :, 64:96]
    image_resized3 = np.expand_dims(image_resized3, -1)
    image_resized3 = np.expand_dims(image_resized3, 0)

    image_resized4 = image_resized[:, :, 96:128]
    image_resized4 = np.expand_dims(image_resized4, -1)
    image_resized4 = np.expand_dims(image_resized4, 0)


    pred1 = model.predict(image_resized1, verbose=1)[0, :, :, :, :]
    pred1 =  np.argmax(pred1, axis=-1)
    pred2 = model.predict(image_resized2, verbose=1)[0, :, :, :, :]
    pred2 = np.argmax(pred2, axis=-1)
    pred3 = model.predict(image_resized3, verbose=1)[0, :, :, :, :]
    pred3 = np.argmax(pred3, axis=-1)
    pred4 = model.predict(image_resized4, verbose=1)[0, :, :, :, :]
    pred4 = np.argmax(pred4, axis=-1)

    pred = np.concatenate((pred1,pred2,pred3,pred4),axis=-1)
    image,mask = restore_image_and_mask(image_resized,pred,mask_data.shape)
    mask = mask.astype(np.float64)
    dice_score = calculate_dice_coefficient(mask_data,mask)
    avd_score = getAVD(mask_data,mask)
    print(dice_score)
    dice.append(dice_score)
    avd.append(avd_score)
    all_dice.append(dice_score)
    all_avd.append(avd_score)
    output_folder = "results/Singapore/"+str(dataindex)
    os.makedirs(output_folder, exist_ok=True)
    mask = nib.Nifti1Image(mask, np.eye(4))
    nifti_filename = os.path.join(output_folder, "wmh.nii.gz")
    nib.save(mask, nifti_filename)


print(np.mean(dice),np.std(dice))



dice = []
avd = []
data_path = sorted(os.listdir(path3), key=lambda x: int(os.path.splitext(x)[0]))
print(data_path)
for dataindex in data_path:
    image_path = os.path.join(path3,dataindex,'pre/FLAIR.nii.gz')
    mask_path = os.path.join(path3,dataindex,'wmh.nii.gz')
    print(mask_path)

    image_data = nib.load(image_path).get_fdata()


    mask_data = nib.load(mask_path).get_fdata()
    print(mask_data.shape)
    image_data,mask_resized= resize_image_and_mask(image_data,mask_data,(336,336,128))

    print(image_data.shape)
    image_resized = crop_image(image_data, (256, 256, 128))
    image_resized -= mean
    image_resized /= std

    image_resized1 = image_resized[:,:,0:32]
    image_resized1 = np.expand_dims(image_resized1, -1)
    image_resized1 = np.expand_dims(image_resized1, 0)

    image_resized2 = image_resized[ :, :, 32:64]
    image_resized2 = np.expand_dims(image_resized2, -1)
    image_resized2 = np.expand_dims(image_resized2, 0)

    image_resized3 = image_resized[:, :, 64:96]
    image_resized3 = np.expand_dims(image_resized3, -1)
    image_resized3 = np.expand_dims(image_resized3, 0)

    image_resized4 = image_resized[:, :, 96:128]
    image_resized4 = np.expand_dims(image_resized4, -1)
    image_resized4 = np.expand_dims(image_resized4, 0)


    pred1 = model.predict(image_resized1, verbose=1)[0, :, :, :, :]
    pred1 =  np.argmax(pred1, axis=-1)
    pred2 = model.predict(image_resized2, verbose=1)[0, :, :, :, :]
    pred2 = np.argmax(pred2, axis=-1)
    pred3 = model.predict(image_resized3, verbose=1)[0, :, :, :, :]
    pred3 = np.argmax(pred3, axis=-1)
    pred4 = model.predict(image_resized4, verbose=1)[0, :, :, :, :]
    pred4 = np.argmax(pred4, axis=-1)

    pred = np.concatenate((pred1,pred2,pred3,pred4),axis=-1)
    print(np.unique(pred))
    print(pred.shape)

    image, pred = resize_image_and_mask(image_resized, pred, (336, 336, 128))
    print("pred",pred.shape)
    image, mask = restore_image_and_mask(image, pred, mask_data.shape)
    mask = mask.astype(np.float64)
    dice_score = calculate_dice_coefficient(mask_data,mask)
    avd_score = getAVD(mask_data,mask)
    print(dice_score)
    dice.append(dice_score)
    avd.append(avd_score)
    all_dice.append(dice_score)
    all_avd.append(avd_score)
    output_folder = "results/Amsterdam/Philips_VU .PETMR_01/"+str(dataindex)
    os.makedirs(output_folder, exist_ok=True)
    mask = nib.Nifti1Image(mask, np.eye(4))
    nifti_filename = os.path.join(output_folder, "wmh.nii.gz")
    nib.save(mask, nifti_filename)


print(np.mean(dice),np.std(dice))



#
#
dice = []
avd = []
data_path = sorted(os.listdir(path4), key=lambda x: int(os.path.splitext(x)[0]))
print(data_path)
for dataindex in data_path:
    image_path = os.path.join(path4,dataindex,'pre/FLAIR.nii.gz')
    mask_path = os.path.join(path4,dataindex,'wmh.nii.gz')
    print(mask_path)

    image_data = nib.load(image_path).get_fdata()


    mask_data = nib.load(mask_path).get_fdata()
    image_resized, mask_resized = resize_image_and_mask(image_data, mask_data, (256, 256, 128))
    image_resized -= mean
    image_resized /= std
    image_resized1 = image_resized[:,:,0:32]
    image_resized1 = np.expand_dims(image_resized1, -1)
    image_resized1 = np.expand_dims(image_resized1, 0)

    image_resized2 = image_resized[ :, :, 32:64]
    image_resized2 = np.expand_dims(image_resized2, -1)
    image_resized2 = np.expand_dims(image_resized2, 0)

    image_resized3 = image_resized[:, :, 64:96]
    image_resized3 = np.expand_dims(image_resized3, -1)
    image_resized3 = np.expand_dims(image_resized3, 0)

    image_resized4 = image_resized[:, :, 96:128]
    image_resized4 = np.expand_dims(image_resized4, -1)
    image_resized4 = np.expand_dims(image_resized4, 0)


    pred1 = model.predict(image_resized1, verbose=1)[0, :, :, :, :]
    pred1 =  np.argmax(pred1, axis=-1)
    pred2 = model.predict(image_resized2, verbose=1)[0, :, :, :, :]
    pred2 = np.argmax(pred2, axis=-1)
    pred3 = model.predict(image_resized3, verbose=1)[0, :, :, :, :]
    pred3 = np.argmax(pred3, axis=-1)
    pred4 = model.predict(image_resized4, verbose=1)[0, :, :, :, :]
    pred4 = np.argmax(pred4, axis=-1)

    pred = np.concatenate((pred1,pred2,pred3,pred4),axis=-1)
    image,mask = restore_image_and_mask(image_resized,pred,mask_data.shape)
    mask = mask.astype(np.float64)
    dice_score = calculate_dice_coefficient(mask_data, mask)
    avd_score = getAVD(mask_data, mask)
    print(dice_score)
    dice.append(dice_score)
    avd.append(avd_score)
    all_dice.append(dice_score)
    all_avd.append(avd_score)
    output_folder = "results/Amsterdam/GE3T/" + str(dataindex)
    os.makedirs(output_folder, exist_ok=True)
    mask = nib.Nifti1Image(mask, np.eye(4))
    nifti_filename = os.path.join(output_folder, "wmh.nii.gz")
    nib.save(mask, nifti_filename)


print(np.mean(dice),np.std(dice))
#
#
#
dice = []
avd = []
data_path = sorted(os.listdir(path5), key=lambda x: int(os.path.splitext(x)[0]))
print(data_path)
for dataindex in data_path:
    image_path = os.path.join(path5,dataindex,'pre/FLAIR.nii.gz')
    mask_path = os.path.join(path5,dataindex,'wmh.nii.gz')
    print(mask_path)

    image_data = nib.load(image_path).get_fdata()


    mask_data = nib.load(mask_path).get_fdata()
    image_resized, mask_resized = resize_image_and_mask(image_data, mask_data, (256, 256, 128))
    image_resized -= mean
    image_resized /= std
    image_resized1 = image_resized[:,:,0:32]
    image_resized1 = np.expand_dims(image_resized1, -1)
    image_resized1 = np.expand_dims(image_resized1, 0)

    image_resized2 = image_resized[ :, :, 32:64]
    image_resized2 = np.expand_dims(image_resized2, -1)
    image_resized2 = np.expand_dims(image_resized2, 0)

    image_resized3 = image_resized[:, :, 64:96]
    image_resized3 = np.expand_dims(image_resized3, -1)
    image_resized3 = np.expand_dims(image_resized3, 0)

    image_resized4 = image_resized[:, :, 96:128]
    image_resized4 = np.expand_dims(image_resized4, -1)
    image_resized4 = np.expand_dims(image_resized4, 0)


    pred1 = model.predict(image_resized1, verbose=1)[0, :, :, :, :]
    pred1 =  np.argmax(pred1, axis=-1)
    pred2 = model.predict(image_resized2, verbose=1)[0, :, :, :, :]
    pred2 = np.argmax(pred2, axis=-1)
    pred3 = model.predict(image_resized3, verbose=1)[0, :, :, :, :]
    pred3 = np.argmax(pred3, axis=-1)
    pred4 = model.predict(image_resized4, verbose=1)[0, :, :, :, :]
    pred4 = np.argmax(pred4, axis=-1)

    pred = np.concatenate((pred1,pred2,pred3,pred4),axis=-1)
    image,mask = restore_image_and_mask(image_resized,pred,mask_data.shape)
    mask = mask.astype(np.float64)
    dice_score = calculate_dice_coefficient(mask_data, mask)
    avd_score = getAVD(mask_data, mask)
    print(dice_score)
    dice.append(dice_score)
    avd.append(avd_score)
    all_dice.append(dice_score)
    all_avd.append(avd_score)
    output_folder = "results/Amsterdam/GE1T5/" + str(dataindex)
    os.makedirs(output_folder, exist_ok=True)
    mask = nib.Nifti1Image(mask, np.eye(4))
    nifti_filename = os.path.join(output_folder, "wmh.nii.gz")
    nib.save(mask, nifti_filename)

print(np.mean(dice),np.std(dice))


print("finish:")
print(np.mean(all_dice),np.std(all_dice))
print(np.mean(all_avd),np.std(all_avd))
