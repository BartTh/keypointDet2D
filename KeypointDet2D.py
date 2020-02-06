import os
import pandas as pd
from skimage import io
import numpy as np
import cv2
from sklearn.model_selection import train_test_split as tts
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage as ndi
import configparser
from model_factory import SimpleConvNet
import copy

config = configparser.ConfigParser()
config.read('config.ini')
all_images = []
cwd = os.getcwd()

train = pd.read_excel(os.path.join(cwd, r'data/List.xlsx'), 'Sheet2')
train = train[:int(config['parameters']['stop_val'])]
Y = train.values[:int(config['parameters']['stop_val']), 1:]/int(config['parameters']['resample_size'])

print('Loading images..')
for i in tqdm(range(0, train['Filenaam'].shape[0])):
    if i == int(config['parameters']['stop_val']):
        break
    img = io.imread(cwd + r'/data/images/' + train['Filenaam'][i][80:-1], as_gray=False)
    img = ndi.zoom(img, (1/int(config['parameters']['resample_size']),
                         1/int(config['parameters']['resample_size']),
                               1), order=0)
    all_images.append(copy.deepcopy(img))

    img[int(Y[i, 1]) - 5: int(Y[i, 1]) + 5, int(Y[i, 0]) - 5:int(Y[i, 0]) + 5] = (255, 0, 0)
    # Save
    cv2.imwrite(cwd + '/data/input_im/in_{}.png'.format(train['Filenaam'][i][80:-5]), img)

x = np.array(all_images) / 255.0

print(x.shape, Y.shape)
x_train, x_val, y_train, y_val = tts(x, Y,
                                     random_state=int(config['network_mode']['random_seed']),
                                     test_size=0.1)

x_train, x_test, y_train, y_test = tts(x_train, y_train,
                                       random_state=int(config['network_mode']['random_seed']),
                                       test_size=0.1)

print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)

network = SimpleConvNet(config)
print()
if eval(config['network_mode']['train']):
    model = network.network()

    model.compile(optimizer=config['network_mode']['optimizer'],
                      loss=config['network_mode']['loss'],
                      metrics=['mae', 'acc'])

    aug = ImageDataGenerator(rotation_range=int(config['data_generator']['rotation_range']),
                             zoom_range=float(config['data_generator']['zoom_range']),
                             width_shift_range=float(config['data_generator']['width_shift_range']),
                             height_shift_range=float(config['data_generator']['height_shift_range']),
                             shear_range=float(config['data_generator']['shear_range']),
                             horizontal_flip=config['data_generator']['horizontal_flip'],
                             fill_mode="nearest")

    model.fit_generator(aug.flow(x_train,
                                 y_train,
                                 batch_size=int(config['parameters']['batch_s'])),
                        validation_data=(x_val, y_val),
                        steps_per_epoch=int(config['parameters']['epoch_steps']),
                        epochs=int(config['parameters']['n_epochs']))

    model.save(config['parameters']['model_dir'])


if config['network_mode']['test']:
    network.predict(config['parameters']['model_dir'], x_test)
    print(y_test)

