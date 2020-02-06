from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from skimage import io
import cv2
from tqdm import tqdm

path = r'D:\Bart\KeypointDet'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

test = pd.read_excel(r'D:\Bart\KeypointDet\List.xlsx', 'Sheet2')

enlarge_pred_dot = 5
down_par = 'downsampled'
# down_par = None
test_images = []
resample_size = 10
# images in test set
test_set = [1, 2, 3, 74, 75, 81, 88, 89, 102, 105, 150, 180, 199,
                253, 267, 298, 301, 319, 344, 380, 410, 430, 470, 471]
# test_set = [199]

print('Loading test images..')
for i in tqdm(range(0, test['Filenaam'].shape[0])):
    if i not in test_set:
        continue
    # print(i, path + '\\images{}'.format(test['Filenaam'][i][-6]) + '\\' + test['Filenaam'][i][80:-6] +
    #       test['Filenaam'][i][-6] + '.png')
    img = io.imread(path + '\\images{}'.format(test['Filenaam'][i][-6]) + '\\' + test['Filenaam'][i][80:-6] +
                    test['Filenaam'][i][-6] + '.png', as_gray=False)
    img = img[::resample_size, ::resample_size, :]
    img = img / 255.0
    img = img.reshape([1920//resample_size, 1080//resample_size, 3])
    test_images.append(img)

test_images = np.array(test_images)
print(test_images.shape)

# load the model we saved
model = load_model('conv_model.h5')
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

pred = model.predict(test_images)
if not down_par == 'downsampled':
    pred += pred * (resample_size-1)
test_stack = np.transpose(np.vstack((test_set, test_set)))
test_stack += 1
print(np.hstack((pred, test_stack))[:, :3])

j = 0

print('Generating predictions..')
for i in tqdm(range(0, test['Filenaam'].shape[0])):
    if i not in test_set:
        continue
    # print(i, path + '\\images{}'.format(test['Filenaam'][i][-6]) + '\\' + test['Filenaam'][i][80:-6] +
    #       test['Filenaam'][i][-6] + '.png')
    img = io.imread(path + '\\images{}'.format(test['Filenaam'][i][-6]) + '\\' + test['Filenaam'][i][80:-6] +
                    test['Filenaam'][i][-6] + '.png', as_gray=False)
    if down_par == 'downsampled':
        img = img[::resample_size, ::resample_size, :]
    else:
        img = img[:, :, :]
    img[int(pred[j, 1])-enlarge_pred_dot: int(pred[j, 1])+enlarge_pred_dot,
    int(pred[j, 0])-enlarge_pred_dot:int(pred[j, 0])+enlarge_pred_dot] = (255, 0, 0)
    # print(j, int(pred[j, 0]), int(pred[j, 1]))
    j += 1
    # Save
    cv2.imwrite(r'D:\Bart\KeypointDet\pred_images\{}_pred_image_{}.png'.format(down_par, i), img)
