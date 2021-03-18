import nvgpu

from src.Common import parse_cmd_args
from src.datasets.sample_datasets.OnlyFood import *
from src.models.sample_models.SampleModel import *

########################################################################################################################

args = parse_cmd_args()

city = "paris".lower().replace(" ", "") if args.c is None else args.c
seed = 100
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
l_rate = 5e-4 if args.lr is None else args.lr
n_epochs = 4000
b_size = 1024 if args.bs is None else args.bs
stage = 3 if args.s is None else args.s

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
