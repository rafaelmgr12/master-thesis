######################################################################################################################################################################################
# This scripts, compute the photo-z for DES DR2 X VIPERS
######################################################################################################################################################################################
from scipy.sparse import hstack, vstack

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import GPz
import astroFunctions as astro
from sklearn.preprocessing import KBinsDiscretizer
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
import keras as ks
from sklearn.metrics import confusion_matrix
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from astropy.table import Table, QTable
import matplotlib.pyplot as plt
import sys
import os
home = os.getenv("HOME")
# user here the path where we download the folder PHTOzxcorr
sys.path.append(home+"/master-thesis/functions/")
import ml_algorithims as ml

# Neural Network Libs


print("Reading the data")
print("/n/n")
path = "/home/rafael/master-thesis/data/vipers.fits"
data = Table.read(path).to_pandas()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


###################################################################################################################################################################
###################### Data Preprocessing #########################################################################################################################
feat = ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'MAG_AUTO_Y',
        'MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED', 'MAG_AUTO_I_DERED', 'MAG_AUTO_Z_DERED', 'MAG_AUTO_Y_DERED',
        "WAVG_MAG_PSF_G", "WAVG_MAG_PSF_R", "WAVG_MAG_PSF_I", "WAVG_MAG_PSF_Z", "WAVG_MAG_PSF_Y", 'WAVG_MAG_PSF_G_DERED', 'WAVG_MAG_PSF_R_DERED', 'WAVG_MAG_PSF_I_DERED', 'WAVG_MAG_PSF_Z_DERED', 'WAVG_MAG_PSF_Y_DERED']

data.loc[data[feat[0]] == 99, feat[0]
         ] = data[data[feat[0]] != 99][feat[0]].max()
data.loc[data[feat[1]] == 99, feat[1]
         ] = data[data[feat[1]] != 99][feat[1]].max()
data.loc[data[feat[2]] == 99, feat[2]
         ] = data[data[feat[2]] != 99][feat[2]].max()
data.loc[data[feat[3]] == 99, feat[3]
         ] = data[data[feat[3]] != 99][feat[3]].max()
data.loc[data[feat[4]] == 99, feat[4]
         ] = data[data[feat[4]] != 99][feat[4]].max()
data.loc[data[feat[5]] > 90, feat[5]] = data[data[feat[5]] < 90][feat[5]].max()
data.loc[data[feat[6]] > 90, feat[6]] = data[data[feat[6]] < 90][feat[6]].max()
data.loc[data[feat[7]] > 90, feat[7]] = data[data[feat[7]] < 90][feat[7]].max()
data.loc[data[feat[8]] > 90, feat[8]] = data[data[feat[8]] < 90][feat[8]].max()
data.loc[data[feat[9]] > 90, feat[9]] = data[data[feat[9]] < 90][feat[9]].max()
data.loc[data[feat[10]] > 90, feat[10]
         ] = data[data[feat[10]] < 90][feat[10]].max()
data.loc[data[feat[11]] > 90, feat[11]
         ] = data[data[feat[11]] < 90][feat[11]].max()
data.loc[data[feat[12]] > 90, feat[12]
         ] = data[data[feat[12]] < 90][feat[12]].max()
data.loc[data[feat[13]] > 90, feat[13]
         ] = data[data[feat[13]] < 90][feat[13]].max()
data.loc[data[feat[14]] > 90, feat[14]
         ] = data[data[feat[14]] < 90][feat[14]].max()
data.loc[data[feat[15]] > 90, feat[15]
         ] = data[data[feat[15]] < 90][feat[15]].max()
data.loc[data[feat[16]] > 90, feat[16]
         ] = data[data[feat[16]] < 90][feat[16]].max()
data.loc[data[feat[17]] > 90, feat[17]
         ] = data[data[feat[17]] < 90][feat[17]].max()
data.loc[data[feat[18]] > 90, feat[18]
         ] = data[data[feat[18]] < 90][feat[18]].max()
data.loc[data[feat[19]] > 90, feat[19]
         ] = data[data[feat[19]] < 90][feat[19]].max()

########################################################################################################################################################################
####################### Get features for training ######################################################################################################################
print("Get features and Setting traning/n/n")

X, y = ml.get_features_targets_des2(data)
y = y.reshape(-1, 1)
kbins = KBinsDiscretizer(200, encode="onehot", strategy="uniform")
kbins.fit(y.reshape(-1, 1))
y_bins = kbins.transform(y.reshape(-1, 1))

y_total = hstack([y_bins, y])
y_total = y_total.toarray()
X = np.concatenate((X, data[['MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED',
                   'MAG_AUTO_I_DERED', 'MAG_AUTO_Z_DERED', 'MAG_AUTO_Y_DERED', ]].values), axis=1)

#X_train, X_test, y_train, y_test = ml.tts_split(X, y_total, 0.3, 5)

########################################################################################################################################################################
####################### Declaring the NN ######################################################################################################################

EarlyStop = EarlyStopping(monitor='reg_mse', mode='min', patience=25)
BATCH_SIZE = 64
STEPS_PER_EPOCH = len(data)//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.0001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False)
inputs = keras.layers.Input(5)
x = BatchNormalization()(inputs)
x = Dense(25, kernel_initializer='normal',  kernel_constraint=max_norm(2.) ,activation='tanh',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)) (x)
x = BatchNormalization()(x)
x = Dense(15, kernel_initializer='normal',  kernel_constraint=max_norm(2.) ,activation='tanh',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)) (x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
x = Dense(10, kernel_initializer='normal',  kernel_constraint=max_norm(2.) ,activation='elu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)) (x)
output1 = Dense(1,activation = "linear",name = "reg") (x)
output2 = Dense(200,activation = "softmax",name ="pdf")(x)
model = keras.Model(inputs=inputs, outputs=[output1,output2], name="ann-custom")
model.compile(
    loss={'reg': 'mean_absolute_error',
          'pdf': keras.losses.CategoricalCrossentropy()}, loss_weights=[0.1, 0.9],
    optimizer=ks.optimizers.Nadam(),
    metrics={'pdf': "acc",
             'reg': "mse"})
history = model.fit(X[:, :5], {
                    'pdf': y[:, :200], 'reg': y[:, 200]}, batch_size=128, epochs=256, validation_split=0.2)

#########################################################################################################################################################################################
####################### Createe the photometric z the NN ######################################################################################################################


path = "/media/new-drive/optical-data/DES_64_Pixels"
filename = os.listdir(path)
path_new = "/media/new-drive/optical-data/DES-photz"

for i in filename:
    name = i[:17] + ".fits"
    if not astro.ver_file(path_new, name):
        data = Table.read(os.path.join(path, i)).to_pandas()
        data.loc[data[feat[0]] == 99, feat[0]
                 ] = data[data[feat[0]] != 99][feat[0]].max()
        data.loc[data[feat[1]] == 99, feat[1]
                 ] = data[data[feat[1]] != 99][feat[1]].max()
        data.loc[data[feat[2]] == 99, feat[2]
                 ] = data[data[feat[2]] != 99][feat[2]].max()
        data.loc[data[feat[3]] == 99, feat[3]
                 ] = data[data[feat[3]] != 99][feat[3]].max()
        data.loc[data[feat[4]] == 99, feat[4]
                 ] = data[data[feat[4]] != 99][feat[4]].max()
        data.loc[data[feat[5]] > 90, feat[5]
                 ] = data[data[feat[5]] < 90][feat[5]].max()
        data.loc[data[feat[6]] > 90, feat[6]
                 ] = data[data[feat[6]] < 90][feat[6]].max()
        data.loc[data[feat[7]] > 90, feat[7]
                 ] = data[data[feat[7]] < 90][feat[7]].max()
        data.loc[data[feat[8]] > 90, feat[8]
                 ] = data[data[feat[8]] < 90][feat[8]].max()
        data.loc[data[feat[9]] > 90, feat[9]
                 ] = data[data[feat[9]] < 90][feat[9]].max()
        data.loc[data[feat[10]] > 90, feat[10]
                 ] = data[data[feat[10]] < 90][feat[10]].max()
        data.loc[data[feat[11]] > 90, feat[11]
                 ] = data[data[feat[11]] < 90][feat[11]].max()
        data.loc[data[feat[12]] > 90, feat[12]
                 ] = data[data[feat[12]] < 90][feat[12]].max()
        data.loc[data[feat[13]] > 90, feat[13]
                 ] = data[data[feat[13]] < 90][feat[13]].max()
        data.loc[data[feat[14]] > 90, feat[14]
                 ] = data[data[feat[14]] < 90][feat[14]].max()
        data.loc[data[feat[15]] > 90, feat[15]
                 ] = data[data[feat[15]] < 90][feat[15]].max()
        data.loc[data[feat[16]] > 90, feat[16]
                 ] = data[data[feat[16]] < 90][feat[16]].max()
        data.loc[data[feat[17]] > 90, feat[17]
                 ] = data[data[feat[17]] < 90][feat[17]].max()
        data.loc[data[feat[18]] > 90, feat[18]
                 ] = data[data[feat[18]] < 90][feat[18]].max()
        data.loc[data[feat[19]] > 90, feat[19]
                 ] = data[data[feat[19]] < 90][feat[19]].max()
        X = ml.get_features_targets_des3(data)

        predictions = model.predict(X)
        zphot = predictions[0].flatten()
        data["Keras:zphot"] = zphot
        DF = QTable.from_pandas(data)
        DF.write("/media/new-drive/optical-data/DES-photz/"+i[:17]+".fits",overwrite = True)
        # data.to_csv("/media/new-drive/optical-data/DES-photz/"+i[:17]+".csv",index = False)

#######################################################################################################################################################################
##### GPz photo-z #########################################################################################################

maxIter = 500                  # maximum number of iterations [default=200]
# maximum iterations to attempt if there is no progress on the validation set [default=infinity]
maxAttempts = 50
trainSplit = 0.8               # percentage of data to use for training
validSplit = 0.2               # percentage of data to use for validation
testSplit = 0              # percentage of data to use for testing

#data = vipers

X = data[['MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED', 'MAG_AUTO_I_DERED',
          'MAG_AUTO_Z_DERED', 'MAG_AUTO_Y_DERED']].values
Y = data["z"].values.reshape(-1, 1)

err = data[["MAGERR_AUTO_G", "MAGERR_AUTO_R",
            "MAGERR_AUTO_I", "MAGERR_AUTO_Z", "MAGERR_AUTO_Y"]].values

X = np.concatenate((X, err), axis=1)
print(X.shape, "\n", Y.shape)


########### Model options ###############

# select method, options = GL, VL, GD, VD, GC and VC [required]
method = 'VC'
#
m = 25                      # number of basis functions to use [required]
#
# jointly learn a prior linear mean function [default=true]
joint = True
#
# learn a heteroscedastic noise process, set to false interested only in point estimates
heteroscedastic = True
#
# cost-sensitive learning option: [default='normal']
csl_method = 'normal'
#       'balanced':     to weigh rare samples more heavly during train
#       'normalized':   assigns an error cost for each sample = 1/(z+1)
#       'normal':       no weights assigned, all samples are equally important
#
# the width of the bin for 'balanced' cost-sensitive learning [default=range(z_spec)/100]
binWidth = 0.1

decorrelate = True          # preprocess the data using PCA [default=False]


n, d = X.shape

filters = int(d/2)

# log the uncertainties of the magnitudes, any additional preprocessing should be placed here
X[:, filters:] = np.log(X[:, filters:])

# sample training, validation and testing sets from the data
training, validation, testing = GPz.sample(
    n, trainSplit, validSplit, testSplit)

# you can also select the size of each sample
# training,validation,testing = GPz.sample(n,10000,10000,10000)

# get the weights for cost-sensitive learning
omega = GPz.getOmega(Y, method=csl_method)


# initialize the initial model
model = GPz.GP(m, method=method, joint=joint,
               heteroscedastic=heteroscedastic, decorrelate=decorrelate)

# train the model
model.train(X.copy(), Y.copy(), omega=omega, training=training,
            validation=validation, maxIter=maxIter, maxAttempts=maxAttempts)

########### NOTE ###########
# you can train the model gain, eve using different data, by executing:
# model.train(model,X,Y,options)


###################################################################################################################################################################
##################################################Create the GPz photo-z###################################################################################
filename = os.listdir(path_new)
for i in filename:
    data = Table.read(os.path.join(path_new, i)).to_pandas()
    data.loc[data[feat[0]] == 99, feat[0]
             ] = data[data[feat[0]] != 99][feat[0]].max()
    data.loc[data[feat[1]] == 99, feat[1]
             ] = data[data[feat[1]] != 99][feat[1]].max()
    data.loc[data[feat[2]] == 99, feat[2]
             ] = data[data[feat[2]] != 99][feat[2]].max()
    data.loc[data[feat[3]] == 99, feat[3]
             ] = data[data[feat[3]] != 99][feat[3]].max()
    data.loc[data[feat[4]] == 99, feat[4]
             ] = data[data[feat[4]] != 99][feat[4]].max()
    data.loc[data[feat[5]] > 90, feat[5]
             ] = data[data[feat[5]] < 90][feat[5]].max()
    data.loc[data[feat[6]] > 90, feat[6]
             ] = data[data[feat[6]] < 90][feat[6]].max()
    data.loc[data[feat[7]] > 90, feat[7]
             ] = data[data[feat[7]] < 90][feat[7]].max()
    data.loc[data[feat[8]] > 90, feat[8]
             ] = data[data[feat[8]] < 90][feat[8]].max()
    data.loc[data[feat[9]] > 90, feat[9]
             ] = data[data[feat[9]] < 90][feat[9]].max()
    data.loc[data[feat[10]] > 90, feat[10]
             ] = data[data[feat[10]] < 90][feat[10]].max()
    data.loc[data[feat[11]] > 90, feat[11]
             ] = data[data[feat[11]] < 90][feat[11]].max()
    data.loc[data[feat[12]] > 90, feat[12]
             ] = data[data[feat[12]] < 90][feat[12]].max()
    data.loc[data[feat[13]] > 90, feat[13]
             ] = data[data[feat[13]] < 90][feat[13]].max()
    data.loc[data[feat[14]] > 90, feat[14]
             ] = data[data[feat[14]] < 90][feat[14]].max()
    data.loc[data[feat[15]] > 90, feat[15]
             ] = data[data[feat[15]] < 90][feat[15]].max()
    data.loc[data[feat[16]] > 90, feat[16]
             ] = data[data[feat[16]] < 90][feat[16]].max()
    data.loc[data[feat[17]] > 90, feat[17]
             ] = data[data[feat[17]] < 90][feat[17]].max()
    data.loc[data[feat[18]] > 90, feat[18]
             ] = data[data[feat[18]] < 90][feat[18]].max()
    data.loc[data[feat[19]] > 90, feat[19]
             ] = data[data[feat[19]] < 90][feat[19]].max()
    X = data[['MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED', 'MAG_AUTO_I_DERED',
              'MAG_AUTO_Z_DERED', 'MAG_AUTO_Y_DERED']].values
    err = data[["MAGERR_AUTO_G", "MAGERR_AUTO_R",
                "MAGERR_AUTO_I", "MAGERR_AUTO_Z", "MAGERR_AUTO_Y"]].values

    X = np.concatenate((X, err), axis=1)
    if len(X) > 0:

        mu, sigma, modelV, noiseV, PHI = model.predict(X.copy())

        zphot = mu.flatten()
        data["GPz:zphot"] = zphot
        data["GPz:sigma"] = sigma.flatten()
        DF = QTable.from_pandas(data)
        DF.write("/media/new-drive/optical-data/DES-photz/" +
                 i[:17]+".fits")
        # data.to_csv("/media/new-drive/optical-data/DES-photz/"+i[:17]+".csv",index = False)
