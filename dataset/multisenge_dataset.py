import os

from torch.utils.data import IterableDataset
import random
import torch

import params

class MultiSenGE(IterableDataset):

    def __init__(
        self,
        split: str='train',
        batch_size: int = params.MULTISENGE_BATCH_SIZE,
        shuffle: bool = False,
        seed: int = params.SEED,
        max_seq_len=params.MULTISENGE_MAX_SEQ_LEN,
    ):
        self.seed = seed
        self.rand=random.Random(self.seed)
        self.batch_size = batch_size

        self.root_dir = params.MULTISENGE_ROOT_DIR
        self.raw_samples = params.MULTISENGE_LABELS
        self.s1_path = params.MULTISENGE_S1_PATH
        self.s2_path = params.MULTISENGE_S2_PATH
        self.labels_path = params.MULTISENGE_GR_PATH

        self.max_seq_len = max_seq_len
        self.img_width = params.MULTISENGE_IMG_WIDTH
        self.num_pixels = self.img_width * self.img_width

        self.num_bands = sum(values['length'] for values in params.CHANNEL_GROUPS.values())
        # apply the split
        self.sample_idx = self.load_split(split)

        if shuffle:
            self.rand.shuffle(self.sample_idx)

        self.num_samples = len(self.sample_idx)
        self.data_len = self.num_samples * self.num_pixels


    def __len__(self):
        return self.num_samples
    
    def __iter__(self):
        for idx, sample_id in enumerate(self.sample_idx):

            eo_style_array = torch.zeros(
                [
                    self.num_bands, 
                    self.max_seq_len, 
                    self.img_width, 
                    self.img_width
                ], 
                dtype=torch.float32
            )
            
            # 1 means missing, 0 means available
            eo_style_mask = torch.ones_like(eo_style_array, dtype=torch.float32)

            # process input
            eo_style_array, eo_style_mask, eo_label, coords, start_month = (
                self.prepare_channel_input(
                    sample_id=sample_id,
                    eo_tensor=eo_style_array,
                    eo_mask=eo_style_mask,
                )
            ) 
        
            # get the sample
            for pixel_id in range(self.num_pixels):
                eo_data_vf= eo_style_array[pixel_id].float().to(params.DEVICE)
                eo_mask_vf = eo_style_mask[pixel_id].float().to(params.DEVICE)
                eo_label_vf = eo_label.float().to(params.DEVICE)

                # MultiSenGE has no Dynamic World
                dw_data_vf = torch.zeros(eo_mask_vf.shape[0], dtype=torch.float32).to(params.DEVICE)
                dw_mask_vf = torch.ones_like(dw_data_vf, dtype=torch.float32).to(params.DEVICE)

                choords_data_vf = torch.tensor(coords[pixel_id], dtype=torch.float32).to(params.DEVICE)

                yield self.create_input_dict(
                    eo_data= eo_data_vf,
                    eo_mask=eo_mask_vf,
                    eo_label=eo_label_vf,
                    dw_data=dw_data_vf,
                    dw_mask=dw_mask_vf,
                    lat_lon=choords_data_vf,
                    start_month=start_month,
                )

            

    def load_split(self, split):
        json_files = [f for f in os.listdir(self.raw_samples) if f.endswith('.json')]
        sample_ids = sorted(f.replace('.json', '') for f in json_files)

        # get the same randomization for each split
        rand = random.Random(self.seed)
        rand.shuffle(sample_ids)

        num_samples = len(sample_ids)
        train_end = int(0.7 * num_samples)
        val_end = int(0.85 * num_samples)

        if split == 'train':
            return sample_ids[:train_end]
        elif split == 'validation':
            return sample_ids[train_end:val_end]
        elif split == 'test':
            return sample_ids[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.") 

    def prepare_channel_input(self, sample_id, eo_tensor, eo_mask):

        data = self._read_json(sample_id)

        s1_files = self._split_filenames(data.get("corresponding_s1", ""))
        s2_files = self._split_filenames(data.get("corresponding_s2", ""))

        acquisitions = []

        if self.use_s2:
            for filename in s2_files:
                path = os.path.join(self.s2_path, filename)
                if os.path.exists(path):
                    acquisitions.append(
                        {
                            "sensor": "S2",
                            "date": self._date_from_filename(filename),
                            "path": path,
                            "filename": filename,
                        }
                    )
                elif self.strict:
                    raise FileNotFoundError(path)

        if self.use_s1:
            for filename in s1_files:
                path = os.path.join(self.s1_path, filename)
                if os.path.exists(path):
                    acquisitions.append(
                        {
                            "sensor": "S1",
                            "date": self._date_from_filename(filename),
                            "path": path,
                            "filename": filename,
                        }
                    )
                elif self.strict:
                    raise FileNotFoundError(path)

        acquisitions = sorted(acquisitions, key=lambda x: x["date"])

        if len(acquisitions) == 0:
            raise RuntimeError(f"No usable S1/S2 acquisitions for {sample_id}")

        start_month = acquisitions[0]["date"].month - 1

        acquisitions = self._downsample_acquisitions(acquisitions, self.max_seq_len)

        gr_filename = self._ground_reference_filename(sample_id)
        gr_path = os.path.join(self.labels_path, gr_filename)

        with rasterio.open(gr_path) as src:
            label_data = src.read(1)
            gr_transform = src.transform
            gr_crs = src.crs

        label_data = self._remap_label(label_data)
        label_data = torch.from_numpy(label_data).long()

        coords = self.generate_coords(gr_transform, gr_crs)

        for time_idx, acq in enumerate(acquisitions):
            if acq["sensor"] == "S2":
                self._insert_s2(
                    path=acq["path"],
                    time_idx=time_idx,
                    eo_tensor=eo_tensor,
                    eo_mask=eo_mask,
                )
            elif acq["sensor"] == "S1":
                self._insert_s1(
                    path=acq["path"],
                    time_idx=time_idx,
                    eo_tensor=eo_tensor,
                    eo_mask=eo_mask,
                )

        eo_tensor = eo_tensor.masked_fill(eo_mask.to(dtype=torch.bool), 0)

        c, t, h, w = eo_tensor.shape

        eo_tensor = eo_tensor.reshape(c, t, h * w).permute(2, 1, 0)
        eo_mask = eo_mask.reshape(c, t, h * w).permute(2, 1, 0)

        region_label = label_data.reshape(h * w)
        coords = coords.reshape(2, h * w).transpose(1, 0)

        return eo_tensor, eo_mask, region_label, coords, start_month

    def create_input_dict(
        self,
        eo_data,
        eo_mask,
        eo_label,
        dw_data,
        dw_mask,
        lat_lon,
        start_month,
    ):
        input_dict = {}
        input_dict['EO'] = eo_data
        input_dict['EO_mask'] = eo_mask
        input_dict['EO_label'] = eo_label
        input_dict['DW'] = dw_data
        input_dict['DW_mask'] = dw_mask
        input_dict['loc'] = lat_lon
        input_dict['month'] = start_month


        return input_dict
