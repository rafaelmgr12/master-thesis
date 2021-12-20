import sys, os
home = os.getenv("HOME")
sys.path.append(home+"/Projetos/PHOTOzxcorr/functions/") # user here the path where we download the folder PHTOzxcorr

import numpy as np
import pandas as pd
import ml_algorithims as ml
from astropy.table import Table
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

vipers = Table.read("/home/rafael/Projetos/PHOTOzxcorr/data/vipers.fits").to_pandas()
feat = ['MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z','MAG_AUTO_Y',
        'MAG_AUTO_G_DERED','MAG_AUTO_R_DERED','MAG_AUTO_I_DERED','MAG_AUTO_Z_DERED','MAG_AUTO_Y_DERED',
        "WAVG_MAG_PSF_G","WAVG_MAG_PSF_R","WAVG_MAG_PSF_I","WAVG_MAG_PSF_Z","WAVG_MAG_PSF_Y"
       ,'WAVG_MAG_PSF_G_DERED','WAVG_MAG_PSF_R_DERED','WAVG_MAG_PSF_I_DERED','WAVG_MAG_PSF_Z_DERED','WAVG_MAG_PSF_Y_DERED']

vipers.loc[vipers[feat[0]]==99,feat[0]] = vipers[vipers[feat[0]]!=99][feat[0]].max()
vipers.loc[vipers[feat[1]]==99,feat[1]] = vipers[vipers[feat[1]]!=99][feat[1]].max()
vipers.loc[vipers[feat[2]]==99,feat[2]] = vipers[vipers[feat[2]]!=99][feat[2]].max()
vipers.loc[vipers[feat[3]]==99,feat[3]] = vipers[vipers[feat[3]]!=99][feat[3]].max()
vipers.loc[vipers[feat[4]]==99,feat[4]] = vipers[vipers[feat[4]]!=99][feat[4]].max()
vipers.loc[vipers[feat[5]]>90,feat[5]] = vipers[vipers[feat[5]]<90][feat[5]].max()
vipers.loc[vipers[feat[6]]>90,feat[6]] = vipers[vipers[feat[6]]<90][feat[6]].max()
vipers.loc[vipers[feat[7]]>90,feat[7]] = vipers[vipers[feat[7]]<90][feat[7]].max()
vipers.loc[vipers[feat[8]]>90,feat[8]] = vipers[vipers[feat[8]]<90][feat[8]].max()
vipers.loc[vipers[feat[9]]>90,feat[9]] = vipers[vipers[feat[9]]<90][feat[9]].max()
vipers.loc[vipers[feat[10]]>90,feat[10]] = vipers[vipers[feat[10]]<90][feat[10]].max()
vipers.loc[vipers[feat[11]]>90,feat[11]] = vipers[vipers[feat[11]]<90][feat[11]].max()
vipers.loc[vipers[feat[12]]>90,feat[12]] = vipers[vipers[feat[12]]<90][feat[12]].max()
vipers.loc[vipers[feat[13]]>90,feat[13]] = vipers[vipers[feat[13]]<90][feat[13]].max()
vipers.loc[vipers[feat[14]]>90,feat[14]] = vipers[vipers[feat[14]]<90][feat[14]].max()
vipers.loc[vipers[feat[15]]>90,feat[15]] = vipers[vipers[feat[15]]<90][feat[15]].max()
vipers.loc[vipers[feat[16]]>90,feat[16]] = vipers[vipers[feat[16]]<90][feat[16]].max()
vipers.loc[vipers[feat[17]]>90,feat[17]] = vipers[vipers[feat[17]]<90][feat[17]].max()
vipers.loc[vipers[feat[18]]>90,feat[18]] = vipers[vipers[feat[18]]<90][feat[18]].max()
vipers.loc[vipers[feat[19]]>90,feat[19]] = vipers[vipers[feat[19]]<90][feat[19]].max()


X,y = ml.get_features_targets_des2(vipers)

for i in range(300):
    X_train, X_test, y_train, y_test = ml.tts_split(X, y, 0.3, 5)
    model = ml.build_nn(5,X_train.shape[1:],len(X_train),"tanh")
    EarlyStop = EarlyStopping(monitor='rmse_ann3', mode='auto', patience=10)
    history = model.fit(X_train, y_train, epochs=256,
                        batch_size=128, validation_split=0.2, callbacks=[EarlyStop])
    test_predictions = model.predict(X_test).flatten()
    redshift = pd.DataFrame()
    redshift["z_phot"] = test_predictions
    redshift["z_spec"] = y_test
    redshift.to_csv("data/redshift"+str(i)+".csv",index = False)
        


