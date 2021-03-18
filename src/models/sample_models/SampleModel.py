# -*- coding: utf-8 -*-

from src.Metrics import precision, recall, f1
from src.models import KerasModelClass
from src.sequences import BaseSequence

import tqdm
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from sklearn.preprocessing import MultiLabelBinarizer

########################################################################################################################


class SampleModel(KerasModelClass):
    """ Ejemplo de modelo. No funciona, solo ejemplifica el uso de las secuencias, train etc..."""

    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        self.DATASET.DATA["N_USR"]= int( self.DATASET.DATA["N_USR"] * self.CONFIG["model"]["pctg_usrs"])

        model_input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(model_input)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu", name="img_emb")(x)
        x = BatchNormalization()(x)
        output = Dense(self.DATASET.DATA["N_USR"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[model_input], outputs=[output])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[precision, recall, f1, "accuracy"])

        return model

    def train(self, dev=False, save_model=False):

        train_seq = self.SampleModelSequence(self)

        if not dev:
            self.__train_model__(train_sequence=train_seq, dev_sequence=None, save_model=save_model)

    ####################################################################################################################

    class SampleModelSequence(BaseSequence):

        def __init__(self, model):
            BaseSequence.__init__(self, parent_model=model)

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
            return self.MODEL.DATASET.DATA['IMG_VEC'][batch_data.id_img.values]

        def preprocess_output(self, batch_data):
            return self.KHOT.fit_transform(batch_data.output.values)

    ####################################################################################################################
