import numpy as np
import pandas as pd

from astropy.table import Table

import sys,os
home = os.getenv("HOME")
sys.path.append(home+"/Projetos/master-thesis/functions/") # user here the path where we download the folder PHTOzxcorr

import wrangle
import keras
import talos
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import confusion_matrix
import keras as ks
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import ml_algorithims as ml
print("Reading the data in'\n")

path = "/media/new-drive/optical-data/DESzxcorr/pycode/DR2-match.fits"
data = Table.read("/home/rafael/Projetos/master-thesis/data/vipers.fits").to_pandas()

print(path,"\n")

#vipers = data[data["source"]==b'VIPERS'].copy()
#data = vipers

feat = ['MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z','MAG_AUTO_Y',
        'MAG_AUTO_G_DERED','MAG_AUTO_R_DERED','MAG_AUTO_I_DERED','MAG_AUTO_Z_DERED','MAG_AUTO_Y_DERED',
        "WAVG_MAG_PSF_G","WAVG_MAG_PSF_R","WAVG_MAG_PSF_I","WAVG_MAG_PSF_Z","WAVG_MAG_PSF_Y"
       ,'WAVG_MAG_PSF_G_DERED','WAVG_MAG_PSF_R_DERED','WAVG_MAG_PSF_I_DERED','WAVG_MAG_PSF_Z_DERED','WAVG_MAG_PSF_Y_DERED']

print("Preprocessing the data\n")

X,y = ml.get_features_targets_des2(data)
y = y.reshape(-1,1)

from sklearn.preprocessing import KBinsDiscretizer
kbins = KBinsDiscretizer(200,encode = "onehot",strategy = "uniform")
kbins.fit(y.reshape(-1,1))
y_bins = kbins.transform(y.reshape(-1,1))


from scipy.sparse import hstack,vstack
y_total = hstack([y_bins,y])
y_total.shape

y_total = y_total.toarray()

X = np.concatenate((X,data[['MAG_AUTO_G_DERED','MAG_AUTO_R_DERED','MAG_AUTO_I_DERED','MAG_AUTO_Z_DERED','MAG_AUTO_Y_DERED',]].values),axis = 1 )

x_train, y1_train, x_val, y1_val = wrangle.array_split(X[:,:5], y_total[:,:200], 0.3)
x_train, y2_train, x_val, y2_val = wrangle.array_split(X[:,:5], y_total[:,200], 0.3)


def telco_churn(x_train, y_train, x_val, y_val, params):

    # the second side of the network
    input_layer = keras.layers.Input(shape=(5,))
    hidden_layer1 = Dense(params['neurons'], activation=params['activation'])(input_layer)
    hidden_layer2 = Dense(params['neurons'], activation=params['activation'])(hidden_layer1)
    hidden_layer3 = Dense(params['neurons'], activation=params['activation'])(hidden_layer2)

    # creating the outputs
    output1 = Dense(200,  activation='softmax', name='pdf')(hidden_layer3)
    output2 = Dense(1, activation='linear', name='reg')(hidden_layer3)

    losses = {"pdf": keras.losses.CategoricalCrossentropy(),
              "reg": "mean_absolute_error"}

    loss_weights = {"pdf": 0.9, "reg": 0.1}

    # put the model together, compile and fit
    model = keras.Model(inputs=input_layer, outputs=[output1, output2])

    model.compile(params["compile"], loss=losses, loss_weights=loss_weights,
                  metrics={'pdf': "acc",
                      'reg': "mse"})

    out = model.fit(x=x_train,
                    y=y_train,
                    #validation_data=[x_val, y_val],
                    validation_split=0.2,
                    epochs=150,
                    batch_size=params['batch_size'],
                    verbose=0)


    return out, model

p = {'activation':['relu', 'elu','selu',"tanh"],
     'neurons': [10,20,30,40,50],
     'batch_size': [32,64,128,256],
     'compile' : ["adam","nadam",keras.optimizers.RMSprop(),keras.optimizers.Adamax(),
                  keras.optimizers.Adagrad(),keras.optimizers.SGD(),keras.optimizers.Ftrl()]}  

print("Begin the Hyperparametrization\n")

scan_object = talos.Scan(x=x_train,
                         y={"pdf":y1_train, "reg":y2_train},
                         x_val=x_val,
                         y_val=[y1_val, y2_val],
                         params=p,
                         model=telco_churn, experiment_name="name")

model = scan_object.best_model(metric='reg_mse', asc=False)

print("\n\n\n")
print(model.summary())

a = scan_object.data

a.to_csv("statistics.csv",index=False)
