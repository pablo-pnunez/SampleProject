# -*- coding: utf-8 -*-

from src.models import KerasModelClass
from src.sequences import BaseImageSequence
from src.sequences.Common import read_and_normalize_images

import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

########################################################################################################################


class SampleModel(KerasModelClass):
    """ Ejemplo de modelo de imágenes. No funciona, solo ejemplifica el uso de las secuencias, train etc..."""

    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def train(self, dev=False, save_model=False):

        train_seq = self.SampleModelSequence(self)

        if not dev:
            self.__train_model__(train_sequence=train_seq, dev_sequence=None, save_model=save_model)

    ####################################################################################################################

    class SampleModelSequence(BaseImageSequence):

        def __init__(self, model, augmentation):
            BaseImageSequence.__init__(self, parent_model=model, data_aug=augmentation)

            self.N_RESTAURANTS = len(self.MODEL.DATASET.DATA["RST_ADY"])
            self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_USR"])))

        def init_data(self):

            n_usrs = self.MODEL.DATASET.DATA["N_USR"]

            x = []
            y = []

            for id_r, rows in tqdm(self.MODEL.DATASET.DATA['TRAIN'].groupby("id_restaurant"), desc="USRS DATA"):
                rst_imgs = self.MODEL.DATASET.DATA["TRAIN_RST_IMG"].loc[id_r]

                # Obtener usuarios
                rltd_usrs = rows.id_user.unique()
                rltd_usrs = rltd_usrs[np.argwhere(rltd_usrs < n_usrs).flatten()]

                x.extend(rst_imgs)
                y.extend([list(rltd_usrs)] * len(rst_imgs))

            ############################################################################################################

            ret = pd.DataFrame(list(zip(x, y)), columns=["id_img", "output"]).sample(frac=1)

            ############################################################################################################

            return ret

        def preprocess_input(self, batch_data):
            return read_and_normalize_images(batch_data.path.values, base_path=self.DATA["img_path"], img_shape=self.DATA["img_shape"][:-1], augmenter=self.AUGMENTER)

        def preprocess_output(self, batch_data):
            return self.KHOT.fit_transform(batch_data.output.values)

    ####################################################################################################################
