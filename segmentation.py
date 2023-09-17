import os
import random
import numpy as np
import pandas as pd
import tensorflow
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import array_to_img
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.models import load_model
from keras_unet.models import custom_unet
from metrics import jaccard_loss, dice_coefficient, jaccard_index, iou_metric, dice_loss, precision, recall, f1_metric
from preprocessing import preprocess_segmentation, UNIFORM_IMAGE_SHAPE, SetGen, load_img_and_mask
from tta_seg import tta_seg

LEARNING_RATE = 1e-4
NUM_EPOCHS = 45
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

MODEL_PATH = 'src/models/seg_model_116'

CUSTOM_OBJECTS = {
    'jaccard_loss': jaccard_loss,
    'dice_coefficient': dice_coefficient,
    'iou_metric': iou_metric,
    'dice_loss': dice_loss,
    'precision': precision,
    'recall': recall,
    'f1_metric': f1_metric,
    'jaccard_index': jaccard_index,
}

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


def build_model(lr=LEARNING_RATE, num_layers=4, filters=32, loss='binary_crossentropy'):
    # build a u-net model
    unet_model = custom_unet(
        input_shape=(*UNIFORM_IMAGE_SHAPE, 3),
        filters=filters,
        num_layers=num_layers,
        dropout=0.,
    )

    unet_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=loss,
        metrics=['accuracy', dice_coefficient, precision, recall, f1_metric, 'AUC', jaccard_index]
    )

    return unet_model


def learn_seg(from_scratch=False):
    # Begin training for a new segmentation model
    preprocess_segmentation(from_scratch=from_scratch)
    unet_model = build_model()
    gen = ImageDataGenerator(
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # zoom_range=0.1,
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
                                       color_mode='grayscale',
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
                                             color_mode='grayscale',
                                             batch_size=BATCH_SIZE,
                                             subset='validation',
                                             class_mode=None,
                                             seed=0)
    train_gen = zip(img_gen, mask_gen)
    valid_gen = zip(valid_img_gen, valid_mask_gen)

    early_stop = EarlyStopping(monitor='val_jaccard_index',
                               min_delta=0,
                               patience=6,
                               mode='max',
                               verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_jaccard_index',
                                  factor=0.1,
                                  patience=2,
                                  min_delta=0.005,
                                  mode='max',
                                  verbose=1)
    model_checkpoint = ModelCheckpoint(MODEL_PATH + "_best",
                                       monitor='val_jaccard_index',
                                       mode='max',
                                       save_best_only=True,
                                       verbose=1)

    history = unet_model.fit(train_gen,
                             validation_data=valid_gen,
                             validation_steps=8100 * VALIDATION_SPLIT // BATCH_SIZE,
                             steps_per_epoch=8100 // BATCH_SIZE // 3,
                             epochs=NUM_EPOCHS,
                             verbose=1,
                             callbacks=[early_stop, reduce_lr, model_checkpoint],
                             )
    unet_model.save(MODEL_PATH)
    observe_history(history)


def segment(model_path, override=False):
    # Evaluate and visualize the performance of a segmentation model
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    s = random.randint(0, 66666) # random seed
    gen = ImageDataGenerator()
    test_set = pd.read_csv('test_set_segmented.csv')
    test_gen = SetGen(test_set['image_id'])
    model.evaluate(test_gen.gen())
    jac, jac_with_inv = evaluate_model(model, test_gen.gen())
    print(f'Mean Jaccard index for test set: {jac}; With inversions: {jac_with_inv}')
    visualize_segmentation(model, test_gen.gen(), 6)
    train_img_gen = gen.flow_from_directory('src/splits/train/',
                                            batch_size=1,
                                            target_size=UNIFORM_IMAGE_SHAPE,
                                            class_mode=None,
                                            seed=s)
    train_mask_gen = gen.flow_from_directory('src/splits/masks/train/',
                                             target_size=UNIFORM_IMAGE_SHAPE,
                                             batch_size=1,
                                             class_mode=None,
                                             seed=s)
    train_gen = zip(train_img_gen, train_mask_gen)
    jac, jac_with_inv = evaluate_model(model, train_gen)
    print(f'Mean Jaccard index for train set: {jac}; With inversions: {jac_with_inv}')
    visualize_segmentation(model, train_gen, 6)
    if override:
        test_set['jaccard_index'] = test_set['image_id'].map(
            lambda img_id: evaluate_record(model, img_id)
        )
        test_set['jaccard_index_improved'] = test_set['image_id'].map(
            lambda img_id: evaluate_record_inverted(model, img_id)
        )
        print(evaluate_model(tta_seg(model), test_gen.gen()))
        save_predictions(model, 'src/test_predictions')


def visualize_segmentation(model, generator, n=1):
    # display a grid of predictions juxtaposed with their respective ground truth and origin for visualization
    fig, axs = plt.subplots(n, 3)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].set_title('Ground Truth')
    axs[0, 2].set_title('Predicted Output')
    for i in range(n):
        gen_img, gen_mask = next(generator)
        original_image = gen_img[0]
        axs[i, 0].imshow(array_to_img(original_image))
        axs[i, 0].axis('off')
        axs[i, 1].imshow(gen_mask[0], plt.cm.binary_r)
        axs[i, 1].axis('off')
        img_pred = model.predict(gen_img)
        axs[i, 2].imshow(img_pred.reshape(img_pred.shape[1:3]), plt.cm.binary_r)
        axs[i, 2].axis('off')
    plt.show()


def evaluate_model(model, set_gen):
    # return the mean jaccard index of the model on the given set (i.e test set).
    jac = []
    jac_with_inv = []
    for image, ground_truth_mask in set_gen:
        prediction = model.predict(image)
        measurement = jaccard_index(prediction, ground_truth_mask)
        jac.append(measurement)
        inv_measurement = jaccard_index(1 - prediction, ground_truth_mask)
        jac_with_inv.append(
            max(measurement, inv_measurement)
        )
        steps_taken = len(jac)
        if steps_taken > 968:
            break
    return np.mean(jac), np.mean(jac_with_inv)


def evaluate_record(model, image_id):
    # evaluate the jaccard index of a record
    orig_image, true_mask = load_img_and_mask(image_id)
    pred_mask = model.predict(orig_image)
    jac = jaccard_index(pred_mask, true_mask)
    return jac.numpy()


def evaluate_record_inverted(model, image_id):
    # evaluate the jaccard index of a record, taking the max between normal and inverted
    orig_image, true_mask = load_img_and_mask(image_id)
    pred_mask = model.predict(orig_image)
    j1 = jaccard_index(pred_mask, true_mask)
    inverted_pred_mask = 1 - pred_mask
    j2 = jaccard_index(inverted_pred_mask, true_mask)

    return max(j1.numpy(), j2.numpy())


def observe_history(history):
    # Analyse the history returned after training a fresh model
    with open(f'{MODEL_PATH}/history.txt', 'w') as f:
        f.write(f'epochs:{history.epoch}\nhistory:{history.history}')
    for idx, metric in enumerate(list(history.history.keys())):
        if 'val_' in metric or metric == 'lr':
            continue
        plt.figure(idx)
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.legend(['train', 'validation'], loc='upper left')
        fig = plt.gcf()
        fig.savefig(MODEL_PATH + f'/{metric} chart')
        fig.show()


def visualize_specific_segmentation(model, img_id):
    # Function to visualize the model's predictions on a specific input image
    fig, axs = plt.subplots(3)
    axs[0].set_title('Original Image')
    axs[1].set_title('Ground Truth')
    axs[2].set_title('Predicted Output')
    orig_image, true_mask = load_img_and_mask(img_id)
    pred_mask = model.predict(orig_image)
    original_image = orig_image[0]
    axs[0].imshow(array_to_img(original_image))
    axs[0].axis('off')
    axs[1].imshow(true_mask[0], plt.cm.binary_r)
    axs[1].axis('off')
    axs[2].imshow(pred_mask.reshape(pred_mask.shape[1:3]), plt.cm.binary_r)
    axs[2].axis('off')
    plt.show()


def save_pred_to_file(model, image_id, _dir):
    # Save a prediction to file
    orig_image, true_mask = load_img_and_mask(image_id)
    pred_mask = model.predict(orig_image)
    plt.imshow(pred_mask.reshape(pred_mask.shape[1:3]), plt.cm.binary_r)
    plt.axis('off')
    plt.savefig(f'{_dir}/{image_id}')


def save_predictions(model, _dir):
    # Save predictions as files
    test_set = pd.read_csv('test_set_segmented.csv')
    for image_id in test_set['image_id']:
        save_pred_to_file(model, image_id, _dir)
