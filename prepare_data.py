import json
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

from indexes import *


crop_id2name = {1: 'Wheat',
                2: 'Mustard',
                3: 'Lentil',
                4: 'No Crop',
                6: 'Sugarcane',
                8: 'Garlic',
                15: 'Potato',
                5: 'Green pea',
                16: 'Bersem',
                14: 'Coriander',
                13: 'Gram',
                9: 'Maize',
                36: 'Rice'}


def field_crop_extractor(field_paths, label_paths):
    field_crops = {}

    for field_path, label_path in tqdm(zip(field_paths, label_paths)):
        with rasterio.open(field_path) as src:
            field_data = src.read()[0]

        with rasterio.open(label_path) as src:
            crop_data = src.read()[0]
        for x in range(0, crop_data.shape[0]):
            for y in range(0, crop_data.shape[1]):
                field_id = str(field_data[x][y])
                field_crop = crop_data[x][y]
                if field_id not in field_crops:
                    field_crops[field_id] = set()

                field_crops[field_id].add(field_crop)

    field_crop_map = []
    for field_id, field_crop in field_crops.items():
        field_crop_ = list(field_crop)
        assert len(field_crop_) == 1
        field_crop_map.append((field_id, field_crop_[0]))

    field_crop = pd.DataFrame(field_crop_map, columns=['field_id', 'crop_id'])

    return field_crop[field_crop['field_id'] != '0']


def feature_extractor(field_paths, source_dirs, selected_bands):
    X_arrays = []
    field_ids = []

    for field_path, source_dir in tqdm(zip(field_paths, source_dirs)):
        with rasterio.open(field_path) as src:
            field_array = src.read()[0]

        field_ids.append(field_array.flatten())

        bands_array = []
        for band in selected_bands:
            with rasterio.open(f'{source_dir}/{band}.tif') as src:
                band_array = np.expand_dims(src.read()[0].flatten(), axis=1)
            bands_array.append(band_array)

        X_tile = np.hstack(bands_array)

        X_arrays.append(X_tile)

    data = pd.DataFrame(np.concatenate(X_arrays), columns=selected_bands)
    data['field_id'] = np.concatenate(field_ids)

    return data[data['field_id'] != 0]


def prepare_data(bands):
    data_root = 'ref_agrifieldnet_competition_v1'
    source_collection = f'{data_root}_source'
    train_label_collection = f'{data_root}_labels_train'
    test_label_collection = f'{data_root}_labels_test'

    with open(f'{data_root}/{train_label_collection}/collection.json') as f:
        train_json = json.load(f)
    with open(f'{data_root}/{test_label_collection}/collection.json') as f:
        test_json = json.load(f)

    train_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in train_json['links'][4:]]
    test_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in test_json['links'][4:]]

    train_field_paths = [f'{data_root}/{train_label_collection}/{train_label_collection}_{i}/field_ids.tif'
                         for i in train_folder_ids]
    test_field_paths = [f'{data_root}/{test_label_collection}/{test_label_collection}_{i}/field_ids.tif'
                        for i in test_folder_ids]

    train_label_paths = [f'{data_root}/{train_label_collection}/{train_label_collection}_{i}/raster_labels.tif'
                         for i in train_folder_ids]

    train_source_dirs = [f'{data_root}/{source_collection}/{source_collection}_{i}'
                         for i in train_folder_ids]
    test_source_dirs = [f'{data_root}/{source_collection}/{source_collection}_{i}'
                        for i in test_folder_ids]

    field_crop_pair = field_crop_extractor(train_field_paths, train_label_paths)

    train_data = feature_extractor(train_field_paths, train_source_dirs, bands)
    test_data = feature_extractor(test_field_paths, test_source_dirs, bands)

    train_data['NDVI'] = train_data.apply(lambda x: NDVI(x), axis=1)
    train_data['SRI'] = train_data.apply(lambda x: SRI(x), axis=1)
    train_data['RENDVI'] = train_data.apply(lambda x: RENDVI(x), axis=1)
    train_data['ARI'] = train_data.apply(lambda x: ARI(x), axis=1)
    train_data['SAVI'] = train_data.apply(lambda x: SAVI(x), axis=1)
    train_data['MSI'] = train_data.apply(lambda x: MSI(x), axis=1)
    train_data['MCARI'] = train_data.apply(lambda x: MCARI(x), axis=1)
    train_data['MARI'] = train_data.apply(lambda x: MARI(x), axis=1)
    train_data['GNDVI'] = train_data.apply(lambda x: GNDVI(x), axis=1)
    # train_data['EVI'] = train_data.apply(lambda x: EVI(x), axis=1)
    train_data['EVI2'] = train_data.apply(lambda x: EVI2(x), axis=1)
    train_data['NDMI'] = train_data.apply(lambda x: NDMI(x), axis=1)
    train_data['NDWI'] = train_data.apply(lambda x: NDWI(x), axis=1)

    test_data['NDVI'] = test_data.apply(lambda x: NDVI(x), axis=1)
    test_data['SRI'] = test_data.apply(lambda x: SRI(x), axis=1)
    test_data['RENDVI'] = test_data.apply(lambda x: RENDVI(x), axis=1)
    test_data['ARI'] = test_data.apply(lambda x: ARI(x), axis=1)
    test_data['SAVI'] = test_data.apply(lambda x: SAVI(x), axis=1)
    test_data['MSI'] = test_data.apply(lambda x: MSI(x), axis=1)
    test_data['MCARI'] = test_data.apply(lambda x: MCARI(x), axis=1)
    test_data['MARI'] = test_data.apply(lambda x: MARI(x), axis=1)
    test_data['GNDVI'] = test_data.apply(lambda x: GNDVI(x), axis=1)
    # test_data['EVI'] = test_data.apply(lambda x: EVI(x), axis=1)
    test_data['EVI2'] = test_data.apply(lambda x: EVI2(x), axis=1)
    test_data['NDMI'] = test_data.apply(lambda x: NDMI(x), axis=1)
    test_data['NDWI'] = test_data.apply(lambda x: NDWI(x), axis=1)

    return train_data, field_crop_pair, test_data
