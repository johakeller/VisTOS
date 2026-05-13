"""
Module containing the MultiSenGE dataset class for fine-tuning.
Calculation of inverse-frequency class weights and split generation via main() method.

Credits for base dataset: Schmitt et al., 2021, https://zenodo.org/records/6375084
"""

import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import rasterio
import torch
from rasterio import transform as rtx
from rasterio import warp as rwarp
from torch.utils.data import IterableDataset

import params


class MultiSenGEDataset(IterableDataset):
    """
    Dataset class for the MultiSenGE crop segmentation dataset.
    """

    def __init__(
        self,
        split="train",
        shuffle=False,
        seed=params.MULTISENGE_SEED,
        max_seq_len=params.MULTISENGE_MAX_SEQ_LEN,
    ):
        self.seed = seed
        self.rand = random.Random(self.seed)

        # data paths
        self.root_dir = params.MULTISENGE_ROOT_DIR
        self.labels_dir = params.MULTISENGE_LABELS
        self.s1_path = params.MULTISENGE_S1_PATH
        self.s2_path = params.MULTISENGE_S2_PATH
        self.gr_path = params.MULTISENGE_GR_PATH

        # image dimensions
        self.max_seq_len = max_seq_len
        self.img_width = params.MULTISENGE_IMG_WIDTH
        self.num_pixels = self.img_width * self.img_width

        # get number of original pre-training input bands
        self.num_bands = sum(
            values["length"] for values in params.CHANNEL_GROUPS.values()
        )

        # get split samples idx list
        self.sample_idx = self.load_split(split)

        # apply shuffling
        if shuffle:
            self.rand.shuffle(self.sample_idx)

        # number of samples (images) used
        self.num_samples = len(self.sample_idx)
        self.data_length = self.num_samples * self.num_pixels

    def __len__(self):
        """
        Returns the dataset length in samples.
        """

        return self.data_length

    def __iter__(self):
        """
        Iterator method of the MultiSenGE dataset. Iterates through the image
        time series and yields pixel time series as samples.
        """

        for sample_id in self.sample_idx:

            # create orig. sized array with 0s to represent missing bands
            eo_style_array = torch.zeros(
                [
                    self.num_bands,
                    self.max_seq_len,
                    self.img_width,
                    self.img_width,
                ],
                dtype=torch.float32,
            )
            # indicate unavailable channels with 1
            eo_style_mask = torch.ones_like(eo_style_array, dtype=torch.float32)

            # prepare and pad the output data
            eo_style_array, eo_style_mask, eo_style_label, coords, start_month = (
                self.prepare_channel_input(
                    sample_id=sample_id,
                    eo_tensor=eo_style_array,
                    eo_mask=eo_style_mask,
                )
            )

            # iterate over pixels
            for pix_id in range(self.num_pixels):
                # obtain pixel time series
                eo_data_vf = eo_style_array[pix_id].float().to(params.DEVICE)
                eo_mask_vf = eo_style_mask[pix_id].float().to(params.DEVICE)
                eo_label_vf = eo_style_label[pix_id].to(params.DEVICE)

                dw_data_vf = torch.zeros(
                    eo_mask_vf.shape[0], dtype=torch.float32
                ).to(params.DEVICE)
                dw_mask_vf = torch.ones_like(dw_data_vf).to(params.DEVICE)

                coords_data_vf = torch.tensor(
                    coords[pix_id], dtype=torch.float32
                ).to(params.DEVICE)

                # yield input dictionary
                yield self.create_input_dict(
                    eo_data=eo_data_vf,
                    dw_data=dw_data_vf,
                    lat_lon=coords_data_vf,
                    start_months=start_month,
                    eo_mask=eo_mask_vf,
                    eo_label=eo_label_vf,
                    dw_mask=dw_mask_vf,
                )

    def load_split(self, split):
        """
        Loads sample IDs for the given split from the split file.
        """

        # split file path
        split_file = os.path.join(
            self.root_dir,
            f"multisenge_{split}_seed{self.seed}.txt",
        )

        # load split sample ids
        with open(split_file, "r", encoding="utf-8") as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        return sample_ids

    def read_json(self, sample_id):
        """
        Reads the JSON metadata for a sample.
        """

        # JSON label path
        json_path = os.path.join(self.labels_dir, f"{sample_id}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def split_filenames(filenames_str):
        """
        Splits a semicolon-separated filename string.
        """

        return [f.strip() for f in filenames_str.split(";") if f.strip()]

    @staticmethod
    def date_from_filename(filename):
        """
        Extracts the image date from a filename.
        Format: {tile}_{YYYYMMDD}_{sensor}_{x}_{y}.tif
        """

        parts = filename.replace(".tif", "").split("_")
        date_str = parts[1]
        return datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))

    @staticmethod
    def downsample_image_files(image_files, max_len):
        """
        Downsamples the image files to max_len entries.
        """

        if len(image_files) <= max_len:
            return image_files
        indices = np.linspace(0, len(image_files) - 1, max_len, dtype=int)
        return [image_files[i] for i in indices]

    @staticmethod
    def ground_reference_filename(sample_id):
        """
        Builds the ground reference filename from a sample ID.
        """

        parts = sample_id.split("_")
        tile, x, y = parts[0], parts[1], parts[2]
        return f"{tile}_GR_{x}_{y}.tif"

    @staticmethod
    def remap_label(label_data):
        """
        Maps out-of-range labels to the ignored class.
        """

        max_label = params.MULTISENGE_NUM_OUTPUTS - 1
        label_data = label_data.copy()
        label_data[(label_data < 0) | (label_data > max_label)] = 0
        return label_data

    @staticmethod
    def generate_coords(transform, src_crs, dst_crs=params.LATLON_CRS):
        """
        Creates a (2, H, W) matrix with lat/lon coordinates.
        """

        width = params.MULTISENGE_IMG_WIDTH
        height = params.MULTISENGE_IMG_WIDTH
        # get rows and column matrices
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        # get coordinates from affine transform
        x, y = rtx.xy(transform, rows, cols, offset="center")
        x, y = np.asarray(x), np.asarray(y)
        # convert to lat/lon
        transformed_coords = rwarp.transform(
            src_crs=src_crs, dst_crs=dst_crs, xs=x.ravel(), ys=y.ravel()
        )
        lon, lat = transformed_coords[0], transformed_coords[1]
        lat = np.asarray(lat).reshape(height, width)
        lon = np.asarray(lon).reshape(height, width)
        # get lat, lon pairs into one matrix
        return np.stack([lat, lon], axis=0).astype(np.float32)

    def insert_s2(self, path, time_idx, eo_tensor, eo_mask):
        """
        Reads a Sentinel-2 GeoTIFF and inserts the bands into the EO tensor.
        """

        # load Sentinel-2 data
        with rasterio.open(path) as src:
            s2_data = src.read().astype(np.float32)
        # normalize to range [0,1]
        s2_data = torch.from_numpy(s2_data * params.MULTISENGE_S2_NORM)

        # insert Sentinel-2 bands
        for group_name, src_indices in params.MULTISENGE_S2_BAND_MAP.items():
            dst_indices = list(params.CHANNEL_GROUPS[group_name]["idx"])
            for src_idx, dst_idx in zip(src_indices, dst_indices):
                eo_tensor[dst_idx, time_idx] = s2_data[src_idx]
                # update EO mask
                eo_mask[dst_idx, time_idx] = 0

        # NDVI from NIR and red ((NIR-red)/(nir+red))
        nir = s2_data[params.MULTISENGE_S2_NIR_BAND]
        red = s2_data[params.MULTISENGE_S2_RED_BAND]
        denominator = nir + red
        # avoid zero division
        ndvi = torch.where(
            denominator != 0,
            (nir - red) / denominator,
            torch.zeros_like(red),
        )
        # fill in data
        ndvi_idx = params.CHANNEL_GROUPS["NDVI"]["idx"][0]
        eo_tensor[ndvi_idx, time_idx] = ndvi
        # update EO mask
        eo_mask[ndvi_idx, time_idx] = 0

    @staticmethod
    def insert_s1(path, time_idx, eo_tensor, eo_mask):
        """
        Reads a Sentinel-1 GeoTIFF and inserts the bands into the EO tensor.
        """

        # load Sentinel-1 data
        with rasterio.open(path) as src:
            s1_data = src.read().astype(np.float32)
        # normalize to range [0,1]
        s1_data = torch.from_numpy(
            (s1_data + params.MULTISENGE_S1_SHIFT) / params.MULTISENGE_S1_DIV
        )

        # insert Sentinel-1
        s1_idx = list(params.CHANNEL_GROUPS["S1"]["idx"])
        for band_offset, dst_idx in enumerate(s1_idx):
            eo_tensor[dst_idx, time_idx] = s1_data[band_offset]
            # update EO mask
            eo_mask[dst_idx, time_idx] = 0

    def prepare_channel_input(self, sample_id, eo_tensor, eo_mask):
        """
        Loads one patch and returns flattened pixel arrays.
        """

        # get metadata
        data = self.read_json(sample_id)

        # get corresponding S1 and S2 filenames
        s1_files = self.split_filenames(data["corresponding_s1"])
        s2_files = self.split_filenames(data["corresponding_s2"])

        image_files = []

        # collect Sentinel-2 files
        for filename in s2_files:
            path = os.path.join(self.s2_path, filename)
            if not os.path.exists(path):
                continue
            image_files.append(
                {
                    "sensor": "S2",
                    "date": self.date_from_filename(filename),
                    "path": path,
                }
            )

        # collect Sentinel-1 files
        for filename in s1_files:
            path = os.path.join(self.s1_path, filename)
            if not os.path.exists(path):
                continue
            image_files.append(
                {
                    "sensor": "S1",
                    "date": self.date_from_filename(filename),
                    "path": path,
                }
            )

        # sort files chronologically
        image_files = sorted(image_files, key=lambda x: x["date"])

        # get start month
        start_month = image_files[0]["date"].month - 1

        # truncate to desired length
        image_files = self.downsample_image_files(image_files, self.max_seq_len)

        # ground reference path
        gr_filename = self.ground_reference_filename(sample_id)
        gr_full_path = os.path.join(self.gr_path, gr_filename)

        # load label and geo information
        with rasterio.open(gr_full_path) as src:
            label_data = src.read(1)
            gr_transform = src.transform
            gr_crs = src.crs

        # label tensor
        label_data = self.remap_label(label_data)
        label_data = torch.from_numpy(label_data).long()

        # create matrix with coordinates for each pixel
        coords = self.generate_coords(gr_transform, gr_crs)

        # insert images into EO-style tensor
        for time_idx, image_file in enumerate(image_files):
            if image_file["sensor"] == "S2":
                self.insert_s2(
                    path=image_file["path"],
                    time_idx=time_idx,
                    eo_tensor=eo_tensor,
                    eo_mask=eo_mask,
                )
            elif image_file["sensor"] == "S1":
                self.insert_s1(
                    path=image_file["path"],
                    time_idx=time_idx,
                    eo_tensor=eo_tensor,
                    eo_mask=eo_mask,
                )

        # mask values
        eo_tensor = eo_tensor.masked_fill(eo_mask.to(dtype=torch.bool), 0)

        # (c, t, h,w) -> reshape and flatten (h*w,t,c)
        c, t, h, w = eo_tensor.shape
        eo_tensor = eo_tensor.reshape(c, t, h * w).permute(2, 1, 0)
        eo_mask = eo_mask.reshape(c, t, h * w).permute(2, 1, 0)
        # (h,w) -> (h*w)
        region_label = label_data.reshape(h * w)
        # (2,h,w) -> (h*w,2)
        coords = coords.reshape(2, h * w).transpose(1, 0)

        return eo_tensor, eo_mask, region_label, coords, start_month

    def create_input_dict(
        self,
        eo_data,
        dw_data,
        lat_lon,
        start_months,
        eo_mask=None,
        eo_label=None,
        dw_mask=None,
    ):
        """
        Returns a dictionary of channel groups, masks and labels.
        """

        # input dictionary: collect channel groups in dictionary, labels and masks
        input_dict = {}
        input_dict["EO"] = eo_data
        input_dict["DW"] = dw_data
        input_dict["loc"] = lat_lon
        input_dict["EO_label"] = eo_label
        input_dict["EO_mask"] = eo_mask
        input_dict["DW_mask"] = dw_mask
        input_dict["month"] = start_months
        return input_dict


def generate_splits(
    root_dir=params.MULTISENGE_ROOT_DIR,
    labels_dir=params.MULTISENGE_LABELS,
    num_selected=params.MULTISENGE_SELECTED_SAMPLES,
    seed=params.MULTISENGE_SEED,
    split_ratio=params.MULTISENGE_SPLIT_RATIO,
):
    """
    Creates the reduced MultiSenGE splits from the JSON label files.
    """

    print(f"MultiSenGE split generation with seed: {seed}.")
    print(f"Root directory: {root_dir}")
    print(f"Number of selected samples: {num_selected}")

    # get sample ids from JSON label files
    json_files = [f for f in os.listdir(labels_dir) if f.endswith(".json")]
    all_sample_ids = sorted(f.replace(".json", "") for f in json_files)

    # select random subset
    rand = random.Random(seed)
    shuffled_ids = all_sample_ids.copy()
    rand.shuffle(shuffled_ids)
    selected = sorted(shuffled_ids[:num_selected])

    # split selected subset
    rand_split = random.Random(seed)
    split_order = selected.copy()
    rand_split.shuffle(split_order)

    # split indices
    n = len(split_order)
    train_end = int(split_ratio[0] * n)
    val_end = int((split_ratio[0] + split_ratio[1]) * n)

    # split sample ids
    train_ids = sorted(split_order[:train_end])
    val_ids = sorted(split_order[train_end:val_end])
    test_ids = sorted(split_order[val_end:])

    def save_ids(filename, ids):
        # save split file
        path = os.path.join(root_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for sid in ids:
                f.write(f"{sid}\n")
        print(f"Saved {len(ids)} sample IDs to {path}")

    # save split files
    save_ids(f"multisenge_selected_{num_selected}_seed{seed}.txt", selected)
    save_ids(f"multisenge_train_seed{seed}.txt", train_ids)
    save_ids(f"multisenge_validation_seed{seed}.txt", val_ids)
    save_ids(f"multisenge_test_seed{seed}.txt", test_ids)

    print(
        f"\nSplit summary: train={len(train_ids)}, "
        f"validation={len(val_ids)}, test={len(test_ids)}"
    )


def init_class_weights_multisenge():
    """
    Computes inverse-frequency class weights from the train split.
    """

    print(f"\rInitiating MultiSenGE class weights.{params.EOL_SPACE}", end="")

    # train split file
    split_file = os.path.join(
        params.MULTISENGE_ROOT_DIR,
        f"multisenge_train_seed{params.MULTISENGE_SEED}.txt",
    )

    # load train sample ids
    with open(split_file, "r", encoding="utf-8") as f:
        train_ids = [line.strip() for line in f if line.strip()]

    # initialize label counter
    label_count = torch.zeros(
        params.MULTISENGE_NUM_OUTPUTS, dtype=torch.long
    )

    # go through train ground references
    for i, sample_id in enumerate(train_ids):
        if (i + 1) % 100 == 0:
            print(
                f"\rLoad ground reference {i + 1}/{len(train_ids)}."
                f"{params.EOL_SPACE}",
                end="",
            )
        # load region mask
        gr_filename = MultiSenGEDataset.ground_reference_filename(sample_id)
        gr_full_path = os.path.join(params.MULTISENGE_GR_PATH, gr_filename)
        with rasterio.open(gr_full_path) as src:
            labels = src.read(1)
        labels = MultiSenGEDataset.remap_label(labels)
        labels = torch.from_numpy(labels).long().flatten()
        # individual label count
        label_count += torch.bincount(
            labels, minlength=params.MULTISENGE_NUM_OUTPUTS
        )

    print()
    class_weights = {}
    # inverse-frequency weights
    for label_name, label_id in params.MULTISENGE_LABELS_DICT.items():
        count = label_count[label_id].item()
        weight = 1.0 / count
        class_weights[label_name] = weight
        # print weights to console output
        print(f"Label: {label_name}, count: {count}, weight: {weight}")

    # also returns weights as tensor
    weights_tensor = torch.tensor(
        list(class_weights.values()), dtype=torch.float32
    ).to(params.DEVICE)
    return weights_tensor


def main():
    """
    Entry point for split generation and class weight computation.
    Usage:
        python -m dataset.multisenge_dataset splits
        python -m dataset.multisenge_dataset weights
    """

    if len(sys.argv) < 2:
        print(
            "Usage: python -m dataset.multisenge_dataset "
            "[splits|weights]"
        )
        sys.exit(1)

    command = sys.argv[1]

    if command == "splits":
        generate_splits()
    elif command == "weights":
        init_class_weights_multisenge()
    else:
        print(f"Unknown command: {command}")
        print(
            "Usage: python -m dataset.multisenge_dataset "
            "[splits|weights]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
