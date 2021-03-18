# -*- coding: utf-8 -*-

from src.Common import parse_cmd_args
from src.datasets.sample_datasets import OnlyFood
from src.models.sample_models import SampleModel

import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "paris".lower().replace(" ", "") if args.c is None else args.c
stage = 3 if args.s is None else args.s

seed = 100 if args.sd is None else args.sd
l_rate = 5e-4 if args.lr is None else args.lr
n_epochs = 4000 if args.ep is None else args.ep
b_size = 1024 if args.bs is None else args.bs
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))

# DATASETS #############################################################################################################

data_cfg = {"city": city, "data_path": "/media/HDD/pperez/TripAdvisor/"+city+"_data/"}
dts = OnlyFood(data_cfg)

# MODELS ###############################################################################################################

model_cfg = {"model": {"learning_rate": 1e-4, "epochs": 1000, "batch_size": 1024, "seed": seed},
             "data_filter": {"min_imgs_per_rest": 5, "min_usrs_per_rest": 5},
             "session": {"gpu": gpu, "in_md5": False}}

mdl = SampleModel(model_cfg, dts)

if stage == 0:  # GridSearch
    mdl.train(dev=True, save_model=True)
elif stage == 1:  # Train
    mdl.train(dev=False, save_model=True)
