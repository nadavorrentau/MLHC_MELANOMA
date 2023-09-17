import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras_unet.models import custom_unet
import keras_tuner as kt

from metrics import dice_loss
from preprocessing import UNIFORM_IMAGE_SHAPE
from segmentation import jaccard_loss, BATCH_SIZE, build_model


def hypermodel(hp):
    filters = hp.Choice('filters', [8, 12, 16, 20, 24, 28])
    num_layers = hp.Choice('num_layers', [3, 4, 5])
    lr = hp.Choice('learning_rate', [1e2, 5e-3, 1e-3, 1e-4, 1e-5])
    unet_model = build_model(lr=lr, num_layers=num_layers, filters=filters, loss='binary_crossentropy')

    return unet_model


def tune_model():
    tuner = kt.Hyperband(hypermodel,
                         objective='val_loss',
                         factor=3,
                         directory='tuning',
                         project_name='tune6')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    gen = ImageDataGenerator(
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.1
    )
    img_gen = gen.flow_from_directory('src/splits/train/',
                                      batch_size=BATCH_SIZE,
                                      target_size=UNIFORM_IMAGE_SHAPE,
                                      subset='training',
                                      class_mode=None,
                                      seed=0)
    mask_gen = gen.flow_from_directory('src/splits/masks/train/',
                                       target_size=UNIFORM_IMAGE_SHAPE,
                                       batch_size=BATCH_SIZE,
                                       subset='training',
                                       class_mode=None,
                                       seed=0)
    valid_img_gen = gen.flow_from_directory('src/splits/train/',
                                            batch_size=BATCH_SIZE,
                                            target_size=UNIFORM_IMAGE_SHAPE,
                                            subset='validation',
                                            class_mode=None,
                                            seed=0)
    valid_mask_gen = gen.flow_from_directory('src/splits/masks/train/',
                                             target_size=UNIFORM_IMAGE_SHAPE,
                                             batch_size=BATCH_SIZE,
                                             subset='validation',
                                             class_mode=None,
                                             seed=0)
    train_gen = zip(img_gen, mask_gen)
    valid_gen = zip(valid_img_gen, valid_mask_gen)
    tuner.search(train_gen, epochs=50, validation_data=valid_gen, steps_per_epoch=8000 // BATCH_SIZE,
                 validation_steps=50, callbacks=[stop_early])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. {best_hps}
    """)

    print(tuner.results_summary)
