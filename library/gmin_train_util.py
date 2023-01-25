import torch
from pathlib import Path
import library.train_util as train_util
import csv
import numpy as np
import os
from tqdm import tqdm
import argparse

class GminDataset(train_util.BaseDataset):
    def read_tag_dump_file(self, path):
        def is_int(s):
            try:
                int(s)
                return True
            except:
                return False
        
        with open(path, newline='') as f:
            reader = csv.reader(f)
            self.tags = dict()
            for e in reader:
                if e[0] == 'id': continue
                if int(e[3]) == 0: continue
                if int(e[2]) == 1: continue
                if int(e[2]) == 3: continue
                if int(e[2]) == 4: continue
                if int(e[2]) == 6: continue
                tag_name = e[1].replace('(', '').replace(')', '').split('_')
                tag_name = " ".join(tag_name)
                if "sound" in tag_name:
                    continue
                if "pokemon" in tag_name:
                    continue
                if tag_name in ["hi res", "absurd res", "digital media artwork"]:
                    continue
                _tag_test = tag_name.split(":")
                if len(_tag_test) == 2:
                    if is_int(_tag_test[0]) and is_int(_tag_test[1]):
                        continue
                if is_int(tag_name):
                    continue

                self.tags[int(e[0])] = tag_name

    def tag_ids_to_caption(self, tag_ids):
        return ", ".join([self.tags[_t] for _t in tag_ids if _t in self.tags])
    
    def __init__(self, batch_size, tag_dump_file, train_data_dir, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, resolution, enable_bucket, min_bucket_reso, max_bucket_reso, flip_aug, color_aug, face_crop_aug_range, random_crop, dataset_repeats, debug_dataset) -> None:
        super().__init__(tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens,
                     resolution, flip_aug, color_aug, face_crop_aug_range, random_crop, debug_dataset)
        
        if os.path.exists(tag_dump_file) and os.path.exists(train_data_dir):
            self.read_tag_dump_file(tag_dump_file)
            print(f"tag dump ok")
        else:
            raise ValueError(f"no tag dump file or dataset")

        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        
        # load all npz to get total length!
        files = [f for f in Path(train_data_dir).glob("*.npz")]
        # files = files[:2] # test only
        
        for f in tqdm(files):
            file = np.load(f)
            for file_in_np in file.files:
                if "reso" in file_in_np or "tags" in file_in_np:
                    continue
                # load respective files
                # TODO: check none?
                latent = file[file_in_np]
                tag_ids = file[f'{file_in_np}_tags']
                reso = file[f'{file_in_np}_reso']
                
                caption = self.tag_ids_to_caption(tag_ids)
                info = train_util.ImageInfo(file_in_np, 1, caption, False, "")
                info.image_size = (int(reso[0]), int(reso[1])) # check w, h or h, w
                info.bucket_reso = info.image_size
                info.latents = torch.FloatTensor(latent)
                
                self.register_image(info)
                
        self.num_train_images = len(self.image_data)
        self._length = len(self.image_data)
        self.num_reg_images = 0
                
        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            sizes.add(image_info.image_size[0])
            sizes.add(image_info.image_size[1])
            resos.add(tuple(image_info.image_size))
        
        self.enable_bucket = True
        self.bucket_resos = list(resos)
        self.bucket_resos.sort()
        self.bucket_aspect_ratios = [w / h for w, h in self.bucket_resos]

        self.min_bucket_reso = min([min(reso) for reso in resos])
        self.max_bucket_reso = max([max(reso) for reso in resos])
        
        self.make_buckets()
        

def add_dataset_arguments(parser: argparse.ArgumentParser):
    # dataset common
    parser.add_argument("--tag_dump_file", type=str, default=None, help="tag dump file for gmin dataset")