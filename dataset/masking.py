"""
Defines the input masking strategy for the pretraining according to
the MAE-based structured masking scheme by Tseng et al. (2024).
"""

import random

import numpy as np

import params


class Masking:
    """
    Class to produce masks on EO and Dynamic World data according to
    MAE-like structured masking strategy. Is initialized with a couple
    of strategies available for masking. A masking object is instantiated
    by the PreTrainingDataset class.
    """

    def __init__(
        self,
        strategies=params.MASKING_STRATEGIES,
        ratio=params.MASKING_RATIO,
        time_steps=params.TIME_STEPS,
    ):
        self.strategies = strategies
        self.ratio = ratio
        self.time_steps = time_steps
        self.len_all_chan_groups = (
            len(params.CHANNEL_GROUPS) + 1
        )  # dynamic world included

    def random_masking(self, eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id):
        """
        Creates masks for Dynamic World data and Earth observation data with
        num_masked_tokens randomly masked time indices.
        """
        # handle srtm extra (one value only required -> only one timestep)
        random_srtm = (
            random.random() <= self.ratio
        )  # select value smaller equals ratio (0.5)
        if random_srtm:
            # only if not yet set True
            if not srtm_mask:
                srtm_mask = True
                num_masked_tokens -= 1

        # ensure num_masked_tokens is not negative
        num_masked_tokens = max(0, num_masked_tokens)
        # handle dynamic world and earth observation data
        eo_mask[:, srtm_id] = True  # remove SRTM from mask
        all_mask = np.concatenate([np.expand_dims(dw_mask, axis=1), eo_mask], axis=1)

        # select tokens to mask randomly
        valid_indices = np.argwhere(~all_mask)
        # not more than valid_indices
        num_masked_tokens = min(num_masked_tokens, len(valid_indices))
        chosen_indices = valid_indices[
            np.random.choice(len(valid_indices), num_masked_tokens, replace=False)
        ]
        all_mask[tuple(chosen_indices.T)] = True

        dw_mask = all_mask[:, 0]
        eo_mask = all_mask[:, 1:]
        return eo_mask, dw_mask, srtm_mask

    def channel_groups_masking(
        self, eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
    ):
        """
        Creates a mask to mask entire channel groups over all available time
        steps at random.
        """

        # handle srtm extra (one value only required -> only one timestep)
        random_srtm = (
            random.random() <= self.ratio
        )  # select value smaller equals ratio (0.5)
        if random_srtm:
            srtm_mask = True
            num_masked_tokens -= 1

        # ensure num_masked_tokens is not negative
        num_masked_tokens = max(0, num_masked_tokens)
        # get the number of channel groups to mask according to ratio
        num_masked_bands = int(num_masked_tokens / self.time_steps)
        # convert column ids of channels to list, remove srtm and dynamic world, they are handled separately
        chan_idx = [
            value["id"]
            for chan, value in params.CHANNEL_GROUPS.items()
            if (chan != "SRTM")
        ]
        chan_idx.append("DW")
        # randomly sample channel indices
        masked_chans_idx = random.sample(chan_idx, num_masked_bands)

        # mask the indicated columns:
        for idx in masked_chans_idx:
            if idx == "DW":
                dw_mask[:] = True
            else:
                eo_mask[:, idx] = True

        # get the rest of tokens to mask according to ratio
        num_masked_tokens -= num_masked_bands * self.time_steps
        # mask the rest
        return self.random_masking(
            eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
        )

    def timesteps_masking(
        self, eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
    ):
        """
        Creates a mask to mask randomly selected time steps across all channel groups.
        """

        # handle srtm extra (one value only required -> only one timestep)
        random_srtm = (
            random.random() <= self.ratio
        )  # select value smaller equals ratio (0.5)
        if random_srtm:
            srtm_mask = True
            num_masked_tokens -= 1

        # ensure num_masked_tokens is not negative
        num_masked_tokens = max(0, num_masked_tokens)
        # get the number of timesteps to mask according to ratio (dw included, -1: exclude SRTM)
        num_masked_steps = int(num_masked_tokens / (self.len_all_chan_groups - 1))
        # not longer than time steps
        num_masked_steps = min(num_masked_tokens, self.time_steps)
        # randomly sample time indices
        masked_steps_idx = random.sample(list(range(self.time_steps)), num_masked_steps)
        # print(masked_steps_idx)

        # mask the indicated rows:
        for idx in masked_steps_idx:
            dw_mask[idx] = True
            eo_mask[idx, :] = True

        # get the rest of tokens to mask according to ratio
        num_masked_tokens -= num_masked_steps * (self.len_all_chan_groups - 1)
        # mask the rest
        return self.random_masking(
            eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
        )

    def cont_timesteps_masking(
        self, eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
    ):
        """
        Creates a mask to mask contiguous time steps across all channel groups.
        """
        # handle srtm extra (one value only required -> only one timestep)
        random_srtm = (
            random.random() <= self.ratio
        )  # select value smaller equals ratio (0.5)
        if random_srtm:
            srtm_mask = True
            num_masked_tokens -= 1

        # ensure num_masked_tokens is not negative
        num_masked_tokens = max(0, num_masked_tokens)
        # get the number of timesteps to mask according to ratio (dw included, -1: exclude SRTM)
        num_masked_steps = int(num_masked_tokens / (self.len_all_chan_groups - 1))
        # not longer than time steps
        num_masked_steps = min(num_masked_tokens, self.time_steps)
        # randomly sample a month index
        start_month = random.choice(list(range(self.time_steps)))
        masked_month_idx = [
            (start_month + i) % (self.time_steps) for i in range(num_masked_steps)
        ]
        # print(masked_month_idx)

        # mask the indicated rows:
        for idx in masked_month_idx:
            dw_mask[idx] = True
            eo_mask[idx, :] = True

        # get the rest of tokens to mask according to ratio
        num_masked_tokens -= num_masked_steps * (self.len_all_chan_groups - 1)
        # mask the rest
        return self.random_masking(
            eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
        )

    def create_mask(self, strategy):
        """
        Creates the mask according to the chosen strategy.
        """

        eo_mask = np.full((self.time_steps, self.len_all_chan_groups - 1), False)
        dw_mask = np.full(self.time_steps, False)
        srtm_mask = False
        num_masked_tokens = int(
            (self.time_steps * self.len_all_chan_groups) * self.ratio
        )  # include dynamic world here
        srtm_id = int(
            params.CHANNEL_GROUPS["SRTM"]["id"]
        )  # exclude SRTM by setting values True

        # MASKING_STRATEGIES=('random','channel_groups', 'contiguous_timesteps', 'timesteps')
        if strategy == "random":
            eo_mask, dw_mask, srtm_mask = self.random_masking(
                eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
            )

        # mask entire channel groups at once
        elif strategy == "channel_groups":
            eo_mask, dw_mask, srtm_mask = self.channel_groups_masking(
                eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
            )

        # masks continuous timestep across all channels
        elif strategy == "contiguous_timesteps":
            eo_mask, dw_mask, srtm_mask = self.cont_timesteps_masking(
                eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
            )

        elif strategy == "timesteps":
            eo_mask, dw_mask, srtm_mask = self.timesteps_masking(
                eo_mask, dw_mask, srtm_mask, num_masked_tokens, srtm_id
            )

        else:
            raise ValueError(f"Masking strategy {strategy} not implemented.")

        # fit srtm mask in
        eo_mask[:, srtm_id] = srtm_mask
        eo_mask = np.repeat(
            eo_mask,
            [props["length"] for props in params.CHANNEL_GROUPS.values()],
            axis=1,
        )  # repeat each mask for the number of encoded channels (columns) per channel group
        return eo_mask, dw_mask

    def mask_input(self, eo_data, dw_data):
        """
        Masks single input (np.ndarray) according to masking strategy. 1=masked, 0=unmasked.
        Returns Earth observation mask, Dynamic World mask, Earth observation input,
        Earth observation label, Dynamic World input and Dynamic World mask.
        """

        # check if strategy available
        for strategy in self.strategies:
            assert (
                strategy in params.MASKING_STRATEGIES
            ), f"Masking strategy {strategy} not implemented."
        # select a random strategy
        strategy = random.choice(self.strategies)
        # apply masking
        eo_mask, dw_mask = self.create_mask(strategy)

        x = eo_data * ~eo_mask
        y = np.zeros(eo_data.shape).astype(np.float32)
        y[eo_mask] = eo_data[eo_mask]

        masked_dw_tokens = (
            np.ones_like(dw_data) * params.DW_CLASSES
        )  # label masked data as 9, labels 0-8 for dynamic world normal data
        x_dw = np.where(dw_mask, masked_dw_tokens, dw_data)
        y_dw = np.zeros(x_dw.shape).astype(np.int16)
        y_dw[dw_mask] = dw_data[dw_mask]

        return eo_mask, dw_mask, x, y, x_dw, y_dw, strategy
