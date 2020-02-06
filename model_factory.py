from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Flatten, Dense, Dropout, MaxPool2D
from keras.models import load_model
from tqdm import tqdm
import cv2
import os


class SimpleConvNet:

    def __init__(self, config):
        self.n_features = [64, 96, 128, 256, 512]
        self.config = config

    def network(self):
        """Simple convolutional neural network architecture"""

        model = Sequential()

        model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False,
                                input_shape=(1080//int(self.config['parameters']['resample_size']),
                                             1920//int(self.config['parameters']['resample_size']),
                                             3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())

        model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        for feature in self.n_features:
            model.add(Convolution2D(feature, (3, 3), padding='same', use_bias=False))
            model.add(LeakyReLU(alpha=0.1))
            model.add(BatchNormalization())

            model.add(Convolution2D(feature, (3, 3), padding='same', use_bias=False))
            model.add(LeakyReLU(alpha=0.1))
            model.add(BatchNormalization())
            if not feature == self.n_features[-1]:
                model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(2))

        return model

    def predict(self, model_path, test_set):
        print('test')

        model = load_model(model_path)
        model.compile(optimizer=self.config['network_mode']['optimizer'],
                      loss=self.config['network_mode']['loss'],
                      metrics=['mae', 'acc'])

        pred = model.predict(test_set)

        print('Generating predictions..')
        j = 0
        enlarge_pred_dot = 5

        for i in tqdm(range(0, test_set.shape[0])):
            img = test_set[i] * 255

            img[int(pred[j, 1]) - enlarge_pred_dot: int(pred[j, 1]) + enlarge_pred_dot,
            int(pred[j, 0]) - enlarge_pred_dot:int(pred[j, 0]) + enlarge_pred_dot] = (255, 0, 0)

            # print(j, int(pred[j, 0]), int(pred[j, 1]))
            j += 1
            # Save
            cv2.imwrite(os.path.join(self.config['parameters']['output_dir'],
                                     'pred_image_{}.png'.format(i)),
                        img)


