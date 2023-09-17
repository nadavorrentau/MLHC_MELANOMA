import os
import logging
import random
import shutil
import re

import numpy as np
import pandas as pd
import pickle

from PIL import Image

from exceptions import SourceFilesMissing

SOURCE_PATH = "/src"

HAM10000_LINK = 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T#'

UNIFORM_IMAGE_SHAPE = (96, 128)


logging.basicConfig(level=logging.INFO)


def preprocess(from_scratch=False):
    validate_src_files()
    ham10000_df = pd.read_csv("src/HAM10000/HAM10000_metadata.csv")
    # Imputation
    ham10000_df = impute(ham10000_df)
    # One Hot Encoding
    ham10000_df = one_hot_encode(ham10000_df)
    # ham10000_df = normalize(ham10000_df)
    # image_data_dict = preprocess_images(from_scratch=from_scratch)
    # ham10000_df['image_data'] = ham10000_df['image_id'].map(image_data_dict.get)
    return ham10000_df


def impute(df):
    # Only age has null values. Fill null values with mean age for sex:
    mean_age_by_sex = df.groupby('sex')['age'].mean()
    mean_age_males = mean_age_by_sex["male"]
    mean_age_females = mean_age_by_sex["female"]
    mean_all_values = df['age'].mean()

    # We will fill unknowns with the general mean because we think the unknown tag
    # may be biased: it is much younger.than the other group means.

    df['age'] = df.apply(lambda row: mean_age_males if (row['sex'] == 'male' and pd.isna(row['age'])) else (
        mean_age_females if (row['sex'] == 'female' and pd.isna(row['age'])) else row['age']), axis=1)
    df['age'].fillna(mean_all_values, inplace=True)
    return df


def one_hot_encode(df):
    exclude_values = ['unknown']

    # For sex and localization columns: create one-hot encoding columns for all values except "unknown"
    location_one_hot_encoding = pd.get_dummies(df['localization']).drop(exclude_values, axis=1)
    sex_one_hot_encoding = pd.get_dummies(df['sex']).drop(exclude_values, axis=1)
    label_one_hot_encoding = pd.get_dummies(df['dx'])
    df = pd.concat([df, sex_one_hot_encoding, location_one_hot_encoding, label_one_hot_encoding],
                   axis=1)

    df = df.drop(columns=["localization", "sex"])
    return df


def preprocess_images(from_scratch=False):
    images_pickle_path = 'src/pickle/image_data_dict.pkl'
    if from_scratch and os.path.exists(images_pickle_path):
        os.unlink(images_pickle_path)
    # Processed image pickle already exists
    elif os.path.exists(images_pickle_path):
        with open(images_pickle_path, 'rb') as f:
            logging.info('Deserializing image_data_dict pickle')
            image_data_dict = pickle.load(f)  # deserialize using load()
        return image_data_dict
    # Pickle does not exist; proceed to process images:
    image_locations = itemize_images(['src/HAM10000/HAM10000_images_part_1', 'src/HAM10000/HAM10000_images_part_2'])
    image_data_dict = image_ids_to_data(image_locations)
    with open(images_pickle_path, 'wb') as f:
        logging.info('Began Pickling image_data_dict')
        pickle.dump(image_data_dict, f)
        logging.info('Pickled image_data_dict successfully')
    return image_data_dict


def image_ids_to_data(image_locations):
    # retrieves image data given id and resizes to consistent dimensions
    logging.info('Resizing images -- this might take a while (necessary on the first run only)')
    return {
        image_id: np.array(Image.open(image_locations[image_id]).resize(UNIFORM_IMAGE_SHAPE, method='nearest'))
        for image_id in image_locations
    }


def itemize_images(image_locations, suffix='.jpg'):
    itemized_dict = {}
    for loc in image_locations:
        for image in os.listdir(loc):
            itemized_dict[image.replace(suffix, "")] = loc + "/" + image
    return itemized_dict


def validate_src_files():
    src_files_present = all(
        [
            os.path.exists('src'),
            os.path.exists('src/HAM10000'),
            os.path.exists('src/HAM10000/HAM10000_images_part_1'),
            os.path.exists('src/HAM10000/HAM10000_images_part_2'),
            os.path.exists('src/HAM10000/HAM10000_metadata.csv'),
        ]
    )
    if not src_files_present:
        raise SourceFilesMissing("Please download and unzip using this link"
                                 " https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T#"
                                 "into the directory /src/HAM10000")
    return


def preprocess_seg_masks(from_scratch=False):
    masks_pickle_path = 'src/pickle/segmentation_masks.pkl'
    if from_scratch and os.path.exists(masks_pickle_path):
        os.unlink(masks_pickle_path)
    # Masks pickle already exists
    elif os.path.exists(masks_pickle_path):
        with open(masks_pickle_path, 'rb') as f:
            logging.info('Deserializing masks pickle')
            seg_masks = pickle.load(f)  # deserialize using load()
        return seg_masks
    # Pickle does not exist; proceed to process images:
    image_locations = itemize_images(['src/HAM10000/HAM10000_segmentations_lesion_tschandl'],
                                     suffix='_segmentation.png')
    seg_masks = image_ids_to_data(image_locations)
    with open(masks_pickle_path, 'wb') as f:
        logging.info('Began Pickling seg_masks')
        pickle.dump(seg_masks, f)
        logging.info('Pickled seg_masks successfully')
    return seg_masks


def preprocess_segmentation(from_scratch=False):
    if len(os.listdir('src/splits/train/imgs')) > 5 and not from_scratch:
        return
    ham10000_df = preprocess(from_scratch=from_scratch)
    # seg_masks = preprocess_seg_masks(from_scratch=from_scratch)
    # ham10000_df['seg_mask'] = ham10000_df['image_id'].map(seg_masks.get)
    return train_test_validation_split(ham10000_df, segmentation=True)


def train_test_validation_split(df, segmentation=False, from_scratch=False):
    if len(os.listdir('src/splits/train/imgs')) > 5 and not from_scratch:
        return
    if os.path.exists('src/test_ids_preselected.csv'):
        filepaths = pd.read_csv('src/test_ids_preselected.csv')['filepath']

        test_image_ids = [re.search("(ISIC_\d+).jpg", str(fp)).group(1) for fp in filepaths]

    else:
        lesion_ids = df['lesion_id']
        test_lesion_ids = lesion_ids.sample(frac=0.2, random_state=0)
        test_image_ids = list(df[df['lesion_id'].isin(test_lesion_ids)]['image_id'])
    itemized_images = itemize_images(['src/HAM10000/HAM10000_images_part_1', 'src/HAM10000/HAM10000_images_part_2'])
    for image_id in itemized_images:
        source = itemized_images[image_id]
        dest_prefix = 'test' if image_id in test_image_ids else 'train'
        dest = f'src/splits/{dest_prefix}/imgs'
        shutil.copy(source, dest)

    if segmentation:
        itemized_images = itemize_images(['src/HAM10000/HAM10000_segmentations_lesion_tschandl'],
                                         suffix='_segmentation.png')
        for image_id in itemized_images:
            source = itemized_images[image_id]
            dest_prefix = 'test' if image_id in test_image_ids else 'train'
            dest = f'src/splits/masks/{dest_prefix}/imgs'
            shutil.copy(source, dest)
    return


def load_img_and_mask(img_id):
    image_locations = itemize_images(['src/HAM10000/HAM10000_images_part_1', 'src/HAM10000/HAM10000_images_part_2'])
    mask_locations = itemize_images(['src/HAM10000/HAM10000_segmentations_lesion_tschandl'],
                                     suffix='_segmentation.png')
    height, width = UNIFORM_IMAGE_SHAPE
    size = (width, height)
    image_data = np.array(Image.open(image_locations[img_id]).resize(size)).reshape((1, height, width, 3))
    mask_data = np.array(Image.open(mask_locations[img_id]).resize(size)).reshape((1, height, width, 1))
    return image_data, mask_data


class SetGen:
    def __init__(self, data):
        self.data = list(data)

    def gen(self, shuffle=True):
        if shuffle:
            self._shuffle()
        return set_generator(self.data)

    def _shuffle(self):
        np.random.shuffle(self.data)


def set_generator(test_ids):
    for test_id in test_ids:
        yield load_img_and_mask(test_id)

