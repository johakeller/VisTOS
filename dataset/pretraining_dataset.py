"""
Module containing the Dynamic World-based pretraining dataset
class, as applied by Tseng et al. (2024).

Credits: Tseng et al., 2024, https://github.com/nasaharvest/presto
"""

import random

import numpy as np
import torch
import webdataset as wds
from webdataset.pipeline import DataPipeline

import params
import utils
from dataset import masking


class PreTrainingDataset:
    """
    Dataset class of Earth observation and derived data for pre-training
    of VisTOS with MAE (masked autoencoders). The size of visual field
    is defined via the parameter vis_field.
    """

    def __init__(
        self,
        length=None,
        vis_field_size=params.VIS_FIELDS[1],
        shuffle=False,
        seed=123,
        time_steps=params.TIME_STEPS,
        masking_strategies=None,
    ):
        self.shuffle = shuffle
        self.seed = seed
        self.rand = random.Random(self.seed)
        # define visual field size of input
        self.vis_field_size = vis_field_size
        # number of time steps
        self.time_steps = time_steps
        # length user defined or if no argument passed, max number of samples available
        self.length = length
        # input masking via structured masking for MAE-strategy
        self.masking = masking.Masking(strategies=masking_strategies)
        self.eo_num_channels = 17
        super().__init__()

    def _decode_file_from_webdataset_tar(self):
        """
        Adds decoding step of the webdataset to the pipeline.
        """

        return wds.decode()

    def create_pipeline(self, url) -> DataPipeline:
        """
        Defines webdataset pipeline: how to load the webdataset and
        transforms model input into correct format.
        """

        print(f"\rCreating pipeline ...{params.EOL_SPACE}", end="")
        pipeline = [
            wds.SimpleShardList(url),
            wds.tarfile_to_samples(),
        ]
        # crop to length
        if self.length is not None:
            pipeline = pipeline + [wds.slice(0, self.length)]
        # shuffling of tars
        if self.shuffle:
            pipeline.append(wds.shuffle(1000, rng=self.rand))
        # decode files
        pipeline.append(self._decode_file_from_webdataset_tar())
        # create time series model input
        pipeline.append(self.generate_input)
        # add shuffling of samples
        if self.shuffle:
            pipeline.append(wds.shuffle(1000, rng=self.rand))

        return wds.DataPipeline(pipeline)

    @staticmethod
    def truncate_time(data, month):
        """
        Select 12 months of the 24 months input array starting at
        passed month index.
        """

        return data[month : month + params.TIME_STEPS]

    def create_input_dict(
        self,
        eo_data,
        dw_data,
        lat_lon,
        start_month,
        eo_mask=None,
        eo_label=None,
        dw_mask=None,
        dw_label=None,
        strategy=None,
    ):
        """
        Returns a dictionary of channel groups, masks and labels, as well as a start month for
        a pixel time series sample as the input for the VisTOS model.
        """

        # input dictionary
        input_dict = {}
        input_dict["EO"] = torch.tensor(eo_data, dtype=torch.float32).to(params.DEVICE)
        input_dict["DW"] = torch.tensor(dw_data, dtype=torch.float32).to(params.DEVICE)
        input_dict["loc"] = torch.tensor(lat_lon, dtype=torch.float32).to(params.DEVICE)
        input_dict["month"] = start_month
        input_dict["EO_label"] = torch.tensor(eo_label).to(params.DEVICE)
        input_dict["DW_label"] = torch.tensor(dw_label).to(params.DEVICE)
        input_dict["EO_mask"] = torch.tensor(eo_mask).float().to(params.DEVICE)
        input_dict["DW_mask"] = torch.tensor(dw_mask).float().to(params.DEVICE)
        input_dict["strategy"] = strategy
        return input_dict

    def generate_input(self, dataset, vis=False):
        """
        Method to load image time series from webdataset and fill an Earth observation array
        at the corresponding time steps with images of the channels, as well as a label map
        of Dynamic World land cover maps. Has the option to visualize multiple channels and
        time steps for a given input image time series. Iterates through the pixel time series
        of the image time series and applies the MAE-like structured masking procedure. Yields
        pixel time series as samples.
        """

        # iterate through samples (25x25 pixel tensors)
        data_list = [
            "s1_s2_era5_srtm_2020_2021.npy",
            "dynamicworldmonthly2020_2021.npy",
            "latlon.npy",
        ]

        for i, sample in enumerate(dataset):
            # get satellite data -> check if no data is available -> skip sample
            if all(key not in sample for key in data_list):
                print(
                    f"\rDataloading: s1_s2_era5_srtm_2020_2021.npy not available in .tar, sample {i} skipped.{params.EOL_SPACE}",
                    end="",
                )
                continue

            # earth observation data
            if "s1_s2_era5_srtm_2020_2021.npy" in sample:
                # check if correct number of pixels
                eo_num_pixels = sample["s1_s2_era5_srtm_2020_2021.npy"].shape[0]
                if eo_num_pixels == params.NUM_PIXELS:
                    eo_data = sample["s1_s2_era5_srtm_2020_2021.npy"]
                    # normalization
                    eo_data = (
                        (eo_data + params.BANDS_SHIFT) / params.BANDS_DIV
                    ).astype(np.float32)
                    # remove band B9 (column index 10)
                    eo_data = np.delete(eo_data, 10, axis=2)
                    # initiate as unmasked
                    eo_mask = np.zeros_like(eo_data)
                # too few pixels
                else:
                    eo_data = np.zeros(
                        (
                            params.IMG_WIDTH * params.IMG_WIDTH,
                            params.MAX_SEQ_LEN,
                            self.eo_num_channels,
                        ),
                        dtype=np.float32,
                    )
                    # initiate as masked
                    eo_mask = np.ones_like(eo_data)
            # no EO data available
            else:
                eo_data = np.zeros(
                    (
                        params.IMG_WIDTH * params.IMG_WIDTH,
                        params.MAX_SEQ_LEN,
                        self.eo_num_channels,
                    ),
                    dtype=np.float32,
                )
                # initiate as masked
                eo_mask = np.ones_like(eo_data)

            # latitude, longitude
            if "latlon.npy" in sample:
                # check if correct number of pixels
                coords_num_pixels = sample["latlon.npy"].shape[0]
                if coords_num_pixels == params.NUM_PIXELS:
                    lat_lon = sample["latlon.npy"]
                # too few pixels
                else:
                    lat_lon = np.zeros(
                        (params.IMG_WIDTH * params.IMG_WIDTH, 2), dtype=np.float32
                    )
            # no coordinates available
            else:
                lat_lon = np.zeros(
                    (params.IMG_WIDTH * params.IMG_WIDTH, 2), dtype=np.float32
                )

            # Dynamic World labels
            if "dynamicworldmonthly2020_2021.npy" in sample:
                # check if correct number of pixels
                dw_num_pixels = sample["dynamicworldmonthly2020_2021.npy"].shape[0]
                if dw_num_pixels == params.NUM_PIXELS:
                    dw_data = sample["dynamicworldmonthly2020_2021.npy"]
                    # initiate as unmasked
                    dw_mask = np.zeros_like(dw_data)
                # too few pixels
                else:
                    dw_data = np.zeros(
                        (params.IMG_WIDTH * params.IMG_WIDTH, params.MAX_SEQ_LEN),
                        dtype=np.float32,
                    )
                    # initiate as masked
                    dw_mask = np.ones_like(dw_data)
            # no DW data available
            else:
                dw_data = np.zeros(
                    (params.IMG_WIDTH * params.IMG_WIDTH, params.MAX_SEQ_LEN),
                    dtype=np.float32,
                )
                # initiate as masked
                dw_mask = np.ones_like(dw_data)

            # visualization
            if vis:
                utils.visualize_images_pretraining(eo_data)

            # iterate through pixels of flattened input image
            for pix_id in range(params.NUM_PIXELS):
                # select one random month from 24 months interval
                start_month = self.rand.choice(list(range(self.time_steps)))
                # truncate to 12 months starting from start_month
                eo_data_tr = self.truncate_time(eo_data[pix_id], start_month)
                eo_mask_tr = self.truncate_time(eo_mask[pix_id], start_month)
                dw_data_tr = self.truncate_time(dw_data[pix_id], start_month)
                dw_mask_tr = self.truncate_time(dw_mask[pix_id], start_month)
                # coordinates have no time dimension
                coord_out = lat_lon[pix_id]

                # applied randomized MAE-masking
                (
                    eo_mask_msk,
                    dw_mask_msk,
                    eo_data_msk,
                    eo_label_msk,
                    dw_data_msk,
                    dw_label_msk,
                    strategy,
                ) = self.masking.mask_input(eo_data_tr, dw_data_tr)

                # logical or to include previous padding in mask
                eo_mask_msk = np.logical_or(eo_mask_tr, eo_mask_msk)
                dw_mask_msk = np.logical_or(dw_mask_tr, dw_mask_msk)

                # yield dictionary of model input
                yield self.create_input_dict(
                    eo_data=eo_data_msk,
                    eo_mask=eo_mask_msk,
                    eo_label=eo_label_msk,
                    dw_data=dw_data_msk,
                    dw_mask=dw_mask_msk,
                    dw_label=dw_label_msk,
                    lat_lon=coord_out,
                    start_month=start_month,
                    strategy=strategy,
                )
