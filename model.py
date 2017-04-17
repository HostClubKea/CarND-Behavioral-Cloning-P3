
import pandas
import sklearn
from sklearn import model_selection
import numpy as np
import skimage
from skimage import transform
import scipy
from scipy import ndimage
import random
import time
import cv2

from multiprocessing import Pool, freeze_support


cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]

def preprocess(image, top_offset=.375, bottom_offset=.125):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = transform.resize(image[top:-bottom, :], (32, 128, 3))
    image = scipy.misc.imresize(image, (64, 64))

    return image

def augment_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 1.0 + random.uniform(-0.7, 0.3)
    image1[:, :, 2] = image1[:, :, 2] * random_bright

    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def simple_shadow(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if bool(random.getrandbits(1)):
        h, w = image1.shape[0], image1.shape[1]
        [x1, x2] = np.random.choice(w, 2, replace=False)
        k = h / (x2 - x1)
        b = -k * x1
        if bool(random.getrandbits(1)):
            for i in range(random.randint(int(h / 2.), h)):
                c = int((i - b) / k)
                image1[i, :c, 2] = image1[i, :c, 2] * .5
        else:
            for i in range(random.randint(int(h / 2.), h)):
                c = int((i - b) / k)
                image1[i, c:, 2] = image1[i, c:, 2] * .5
    else:
        h, w = image1.shape[0], image1.shape[1]
        [y1, y2] = np.random.choice(h, 2, replace=False)
        k = w / (y2 - y1)
        b = -k * y1
        if bool(random.getrandbits(1)):
            for i in range(random.randint(int(w / 2.), w)):
                c = int((i - b) / k)
                image1[:c, i, 2] = image1[:c, i, 2] * .5
        else:
            for i in range(random.randint(int(w / 2.), w)):
                c = int((i - b) / k)
                image1[c:, i, 2] = image1[c:, i, 2] * .5
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def augment_sample(i, data, augment):
    # Randomly select camera
    camera = np.random.randint(len(cameras)) if augment else 1
    # Read frame image and work out steering angle
    #image = ndimage.imread(data[cameras[camera]].values[i].strip(), mode="RGB")
    image = cv2.imread(data[cameras[camera]].values[i].strip())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    angle = data.steering.values[i] + cameras_steering_correction[camera]
    if augment:
        rnd = random.uniform(0, 1)
        if rnd < 0.3:
            image = augment_brightness(image)
        elif rnd < 0.5:
            image = simple_shadow(image)

    # Randomly shift up and down while preprocessing
    v_delta = random.uniform(-0.05, 0.05) if augment else 0

    image = preprocess(
        image,
        top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
        bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
    )
    return (image, angle)

def samples_generator(data, batch_size = 128, augment=False):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.
    """

    while True:
        indices = np.random.permutation(data.count()[0])

        for batch in range(0, len(indices), batch_size):

            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, 32, 128, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images

            from itertools import repeat

            with Pool(3) as p:
                results = p.starmap(augment_sample, zip(batch_indices, repeat(data), repeat(augment))) #p.map(augment_sample, batch_indices)

            x, y = zip(*results)
            x = np.array(x)
            y = np.array(y)

            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]

            yield (x, y)


if __name__ == '__main__':

    freeze_support()
    import keras

    t1 = time.process_time()

    class WeightsLogger(keras.callbacks.Callback):
        """
        Keeps track of model weights by saving them at the end of each epoch.
        """

        def __init__(self):
            super(WeightsLogger, self).__init__()

        def on_epoch_end(self, epoch, logs={}):
            self.model.save('model_epoch_{}.h5'.format(epoch + 1))


    import os
    frames = []
    for folder in ['samples/udacity/', 'samples/forward/', 'samples/forward2/', 'samples/backward/', 'samples/backward2/',
                   'samples/recovery/', 'samples/recovery2/', 'samples/recovery3/']:#, 'samples/recovery4/'
        subdataset = pandas.io.parsers.read_csv(folder + 'driving_log.csv')
        subdataset['left'] = subdataset['left'].apply(lambda x: folder + str.strip(x))
        subdataset['center'] = subdataset['center'].apply(lambda x: folder + str.strip(x))
        subdataset['right'] = subdataset['right'].apply(lambda x: folder + str.strip(x))

        frames.append(subdataset)

    dataset = pandas.concat(frames)
    dataset = dataset.reset_index(drop=True)

    cnt = 0
    eps = 0.1

    indexes = []
    for index, row in dataset.iterrows():
        angle = row['steering']
        if abs(angle) <= eps:
            if cnt < 1:
                cnt += 1
            elif cnt > 7:
                cnt = 0
                indexes.append(index)
            else:
                cnt += 1
                indexes.append(index)
        else:
            cnt = 0

    dataset = dataset.drop(dataset.index[indexes], axis=0)

    train, valid = model_selection.train_test_split(dataset, test_size=.2)

    activation_relu = 'relu'

    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3), output_shape=(64, 64, 3)))
    model.add(keras.layers.convolutional.Convolution2D(3, 1, 1,input_shape=(64, 64, 3), border_mode='valid', init='he_normal', activation='relu'))
    model.add(keras.layers.convolutional.Convolution2D(24, 5, 5,  border_mode='same', subsample=(2, 2), activation='relu'))
    model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.convolutional.Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
    model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.convolutional.Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
    model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.convolutional.Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))
    model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.convolutional.Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))
    model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.core.Flatten())

    # Next, five fully connected layers
    model.add(keras.layers.core.Dense(1164, activation='relu'))
    model.add(keras.layers.core.Dropout(.5))

    model.add(keras.layers.core.Dense(100, activation='relu'))
    model.add(keras.layers.core.Dropout(.25))

    model.add(keras.layers.core.Dense(50, activation='relu'))
    model.add(keras.layers.core.Dropout(.1))

    model.add(keras.layers.core.Dense(10, activation='relu'))

    model.add(keras.layers.core.Dense(1))

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-04), loss='mean_squared_error')

    history = model.fit_generator(
        samples_generator(train, augment=True),
        samples_per_epoch=train.shape[0],
        nb_epoch=20,
        validation_data=samples_generator(valid, augment=False),
        callbacks=[WeightsLogger()],
        nb_val_samples=valid.shape[0]
    )

    elapsed_time1 = time.process_time() - t1
    print("\r\nTotal - " + str(elapsed_time1))

    with open('model.json', 'w') as file:
        file.write(model.to_json())

# backend.clear_session()