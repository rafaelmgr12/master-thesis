import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regression Libraries
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
# Other libraries

from sklearn.model_selection import cross_val_score, KFold, train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import norm, skew  # for some statistics

# Neural Network Libs
import keras
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




def random_lightGBM(len_dataset):

    b_type = ["gbdt", "dart", "goss"]
    l_rate = 10.0 ** -np.arange(0, 20)
    n_estimator = list(range(200, 1000, 100))
    num_leaves = list(range(31, 200, 10))

    typee = random.choice(b_type)
    rate = random.choice(l_rate)
    nest = random.choice(n_estimator)
    leaves = random.choice(num_leaves)
    alpha = random.choice(l_rate)
    lambd = random.choice(l_rate)

    model = LGBMRegressor(boosting_type=typee, num_leaves=leaves,
                          learning_rate=rate, n_estimators=nest, reg_alpha=alpha, reg_lambda=lambd)
    return model


def random_DTR(len_train_features, alg="Boost"):
    """ Define a Boosted/Bagging  and Gradient Decision tree random
    """
    # DTR parameters
    min_samples_s = list(range(2, 100, 2))
    min_samples_leaf = list(range(2, 100, 2))
    crit = ['mse', 'friedman_mse', 'mae']
    split = ["best", "random"]
    max_feat = ["auto", "sqrt", "log2"]
    # Boost parmeter
    n_estimator = list(range(200, 1000, 100))
    l_rate = 10.0 ** -np.arange(0, 20)
    loss = ['linear', 'square', 'exponential']
    # Bagging parameters
    max_samp = list(range(2, len_train_features, 1))
    max_feat_b = list(range(2, len_train_features, 1))
    boots = [False, True]
    boots_feat = [False, True]
    oob = [False, True]
   # warm = [False, True]
    # Gradient parameters
    loss_g = ['ls', 'lad', 'huber', 'quantile']
    crit_g = ['mse', 'friedman_mse', 'mae']
    max_depth = list(range(3, 40, 1))
    # Hist GradientBoostingRegresso

    loss_h = ["least_squares", "least_absolute_deviation", "poisson"]
    m_iter = list(range(200, 1000, 100))

    mss = random.choice(min_samples_s)
    msl = random.choice(min_samples_leaf)
    crt = random.choice(crit)
    spt = random.choice(split)
    mft = random.choice(max_feat)
    nest = random.choice(n_estimator)
    rate = random.choice(l_rate)
    closs = random.choice(loss)
    msb = random.choice(max_samp)
    mfb = random.choice(max_feat_b)
    bts = random.choice(boots)
    btf = random.choice(boots_feat)
    oobs = random.choice(oob)
    #wrms = random.choice(warm)
    lsg = random.choice(loss_g)
    mdt = random.choice(max_depth)
    l = random.choice(loss_h)
    m = random.choice(m_iter)

    tree = DecisionTreeRegressor(
        criterion=crt, splitter=spt, min_samples_split=mss, min_samples_leaf=msl, max_features=mft)
    if alg == "Boost":
        bdt = AdaBoostRegressor(base_estimator=tree, n_estimators=nest,
                                learning_rate=rate, loss=closs)
        return bdt
    if alg == "Bagging":
        bdt = BaggingRegressor(base_estimator=tree, n_estimators=nest, max_samples=msb,
                               max_features=mfb)  # warm_start=wrms)
        return bdt
    if alg == "Gradient":
        gbt = GradientBoostingRegressor(loss=lsg, learning_rate=rate, n_estimators=nest,
                                        criterion=crt, min_samples_split=mss, min_samples_leaf=msl, max_depth=mdt)
        return gbt
    if alg == "Hist":
        hist = HistGradientBoostingRegressor(
            loss=l, learning_rate=rate, max_iter=m, max_depth=mdt, l2_regularization=1e-5, min_samples_leaf=msl)
        return hist

def clean_tab(tab, col, val, type_s=0):
    '''Function to clean the
 droping the row of values. Using 0 for equal values,-1 less than and +1 for greater than.'''
    if type_s == 0:
        tab.drop(tab[tab[col] == val].index, inplace=True)
    if type_s == -1:
        tab.drop(tab[tab[col] < val].index, inplace=True)
    if type_s == 1:
        tab.drop(tab[tab[col] > val].index, inplace=True)

def tts_split2(X, y, size, splits):
    '''Split the data in Train and
     test using the Shuffle split with random_state fix'''

    rs = ShuffleSplit(n_splits=splits, test_size=size,random_state=101)

    rs.get_n_splits(X)

    for train_index, test_index in rs.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def tts_split(X, y, size, splits):
    '''Split the data in Train and
     test using the Shuffle split'''

    rs = ShuffleSplit(n_splits=splits, test_size=size)

    rs.get_n_splits(X)

    for train_index, test_index in rs.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def rmsle_cv(model, X_train, y_train):
    """Root mean Square using cross validation"""
    kf = KFold(3, shuffle=True, random_state=None).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train.ravel(),
                                    scoring="neg_mean_squared_error", cv=kf))
    return(rmse)


def rmsle(y, y_pred):
    '''Root mean Square'''
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse_ann(y_true, y_pred):
    '''root mean square erro for using as metric for the ANNs'''
    return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true), axis=-1))


def rmse_ann2(y_true, y_pred):
    return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true), axis=-1))


def rmse_ann3(y_true, y_pred):
    return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true), axis=-1))


def plot_history(history):
    '''Plot the graphics for the train_error and val_loss'''
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
# Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def get_features_targets_des1(data):
    '''Extract the colors using the mag.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['MAG_AUTO_G'].values - data['MAG_AUTO_R'].values
    features[:, 1] = data['MAG_AUTO_R'].values - data['MAG_AUTO_I'].values
    features[:, 2] = data['MAG_AUTO_I'].values - data['MAG_AUTO_Z'].values
    features[:, 3] = data['MAG_AUTO_Z'].values - data['MAG_AUTO_Y'].values
    # features[:, 4] = data['WAVG_MAG_PSF_G'] - data['WAVG_MAG_PSF_R']
    # features[:, 5] = data['WAVG_MAG_PSF_R'] - data['WAVG_MAG_PSF_I']
    # features[:, 6] = data['WAVG_MAG_PSF_I'] - data['WAVG_MAG_PSF_Z']
    # features[:, 7] = data['WAVG_MAG_PSF_Z'] - data['WAVG_MAG_PSF_Y']
    features[:, 4] = data["MAG_AUTO_I"].values

    targets = data['z'].values
    return features, targets


def get_features_targets_des2(data):
    '''Extract the colors using the mag_derred.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['MAG_AUTO_G_DERED'].values - \
        data['MAG_AUTO_R_DERED'].values
    features[:, 1] = data['MAG_AUTO_R_DERED'].values - \
        data['MAG_AUTO_I_DERED'].values
    features[:, 2] = data['MAG_AUTO_I_DERED'].values - \
        data['MAG_AUTO_Z_DERED'].values
    features[:, 3] = data['MAG_AUTO_Z_DERED'].values - \
        data['MAG_AUTO_Y_DERED'].values
    #features[:, 4] = data['WAVG_MAG_PSF_G_DERED'].values - \
    #   data['WAVG_MAG_PSF_R_DERED'].values
    #features[:, 5] = data['WAVG_MAG_PSF_R_DERED'].values - \
    #    data['WAVG_MAG_PSF_I_DERED'].values
    #features[:, 6] = data['WAVG_MAG_PSF_I_DERED'].values - \
    #   data['WAVG_MAG_PSF_Z_DERED'].values
    #features[:, 7] = data['WAVG_MAG_PSF_Z_DERED'].values - \
    #  data['WAVG_MAG_PSF_Y_DERED'].values

    features[:, 4] = data["MAG_AUTO_I_DERED"].values
    #features[:, 9] = data["WAVG_MAG_PSF_I_DERED"].values

    targets = data['z'].values
    return features, targets

def get_features_targets_des3(data):
    '''Extract the colors using the mag_derred.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['MAG_AUTO_G_DERED'].values - \
        data['MAG_AUTO_R_DERED'].values
    features[:, 1] = data['MAG_AUTO_R_DERED'].values - \
        data['MAG_AUTO_I_DERED'].values
    features[:, 2] = data['MAG_AUTO_I_DERED'].values - \
        data['MAG_AUTO_Z_DERED'].values
    features[:, 3] = data['MAG_AUTO_Z_DERED'].values - \
        data['MAG_AUTO_Y_DERED'].values
    #features[:, 4] = data['WAVG_MAG_PSF_G_DERED'].values - \
    #   data['WAVG_MAG_PSF_R_DERED'].values
    #features[:, 5] = data['WAVG_MAG_PSF_R_DERED'].values - \
    #    data['WAVG_MAG_PSF_I_DERED'].values
    #features[:, 6] = data['WAVG_MAG_PSF_I_DERED'].values - \
    #   data['WAVG_MAG_PSF_Z_DERED'].values
    #features[:, 7] = data['WAVG_MAG_PSF_Z_DERED'].values - \
    #  data['WAVG_MAG_PSF_Y_DERED'].values

    features[:, 4] = data["MAG_AUTO_I_DERED"].values
    #features[:, 9] = data["WAVG_MAG_PSF_I_DERED"].values

   
    return features

def get_features_targets_gama1(data):
    '''Extract the colors using the mag_derred.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['gKronMag'].values - \
        data['rKronMag'].values
    features[:, 1] = data['rKronMag'].values - \
        data['iKronMag'].values
    features[:, 2] = data['iKronMag'].values - \
        data['zKronMag'].values
    features[:, 3] = data['zKronMag'].values - \
        data['yKronMag'].values
    # features[:, 4] = data['WAVG_MAG_PSF_G_DERED'] - data['WAVG_MAG_PSF_R_DERED']
    # features[:, 5] = data['WAVG_MAG_PSF_R_DERED'] - data['WAVG_MAG_PSF_I_DERED']
    # features[:, 6] = data['WAVG_MAG_PSF_I_DERED'] - data['WAVG_MAG_PSF_Z_DERED']
    # features[:, 7] = data['WAVG_MAG_PSF_Z_DERED'] - data['WAVG_MAG_PSF_Y_DERED']
    features[:, 4] = data["iKronMag"].values

    targets = data['Z'].values
    return features, targets


def smote(X, y, n, k):
    '''Add values for the dataset using the smote technique'''

    if n == 0:
        return X, y

    knn = KNeighborsRegressor(k, "distance").fit(X, y)
    # choose random neighbors of random points
    ix = np.random.choice(len(X), n)
    nn = knn.kneighbors(X[ix], return_distance=False)
    newY = knn.predict(X[ix])
    nni = np.random.choice(k, n)
    ix2 = np.array([n[i] for n, i in zip(nn, nni)])

    # synthetically generate mid-point between each point and a neighbor
    dif = X[ix] - X[ix2]
    gap = np.random.rand(n, 1)
    newX = X[ix] + dif*gap
    return np.r_[X, newX], np.r_[y, newY]


def gaussian_noise(X, y, sigma, n):
    """
    Add gaussian noise to the dataset
    """
    _X = X.copy()
    _y = y.copy()
    for _ in range(n):
        X = np.r_[X, _X + np.random.randn(*_X.shape)*sigma]
        y = np.r_[y, _y]
    return X, y


def plot_model(model):
    return SVG(keras.utils.vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def rmse_loss_keras(y_true, y_pred):
    diff = keras.backend.square(
        (y_pred - y_true) / (keras.backend.abs(y_true) + 1))
    return keras.backend.sqrt(keras.backend.mean(diff))


def build_nn(n_inputs, shape, len_dataset, activations):
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = (len_dataset)//BATCH_SIZE
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.0001,
        decay_steps=STEPS_PER_EPOCH*1000,
        decay_rate=1,
        staircase=False)
    n_units = list(range(20, n_inputs + 70, 1))

    ann_model = Sequential([Dense(n_inputs, input_shape=shape, kernel_initializer=tf.keras.initializers.Ones(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
                            Dense(random.choice(n_units), kernel_initializer='normal',  kernel_constraint=max_norm(2.5), activation=activations, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
                            BatchNormalization(),
                            Dense(random.choice(n_units), kernel_initializer='normal', kernel_constraint=max_norm(2.5), activation=activations, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
                            BatchNormalization(),
                            Dense(random.choice(n_units), kernel_initializer='normal', kernel_constraint=max_norm(2.5), activation=activations, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
                            Dense(1, activation=None, name="output")
                            ])
    list_opt = [ks.optimizers.Adam(lr_schedule), ks.optimizers.RMSprop(lr_schedule)]
    opt = random.choice(list_opt)
    #opt = tf.keras.optimizers.RMSprop(0.001)
    ann_model.compile(optimizer=opt, loss="mse",
                      metrics=['mse', 'mae', rmse_ann3])

    return ann_model


def model_nn(len_dataset, input_dim, n_hidden_layers, dropout=0, batch_normalization=False,
             activation='tanh', neurons_decay=0, starting_power=1, l2=0,
             compile_model=True, trainable=True, schedule=True):
    """Define an ANN with tanh activation"""

    # Define a optmitzer
    if schedule == True:
        BATCH_SIZE = 64
        STEPS_PER_EPOCH = (len_dataset)//BATCH_SIZE
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False)
        opt = keras.optimizers.Adam(lr_schedule)
    if schedule == False:
        opt = keras.optimizers.Adam()

    assert dropout >= 0 and dropout < 1
    assert batch_normalization in {True, False}
    model = keras.models.Sequential()

    for layer in range(n_hidden_layers):
        n_units = input_dim + 3
        # n_units = 2**(int(np.log2(input_dim)) +
        #              starting_power - layer*neurons_decay)
        # if n_units < 8:
        #    n_units = 8
        if layer == 0:
            model.add(Dense(units=n_units, input_dim=input_dim, name='Dense_' + str(layer + 1),
                            kernel_regularizer=keras.regularizers.l2(l2)))
        else:
            model.add(Dense(units=n_units, name='Dense_' + str(layer + 1),
                            kernel_regularizer=keras.regularizers.l2(l2)))
        if batch_normalization:
            model.add(BatchNormalization(
                name='BatchNormalization_' + str(layer + 1)))
        model.add(Activation('tanh', name='Activation_' + str(layer + 1)))
        if dropout > 0:
            model.add(Dropout(dropout, name='Dropout_' + str(layer + 1)))

    model.add(Dense(units=1, name='Dense_' + str(n_hidden_layers+1),
                    kernel_regularizer=keras.regularizers.l2(l2)))
    model.trainable = trainable
    if compile_model:
        model.compile(loss=rmse_loss_keras, optimizer=opt,
                      metrics=['mse', 'mae', rmse_loss_keras])

    return model


def create_random_nn(input_shape, N_inputs):
    """
    Creates a CNN, based on random layer size.
    Idea is to generate similar CNN models per function call.

    Args:
        input_shape: the input_shape of the model

    Returns: a keras CNN model
    """
    weight_decay = 1e-4
    num_classes = 10
    model = Sequential()
    model.add(Dense(N_inputs,
                    kernel_regularizer=regularizers.l2(weight_decay),
                    input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))

    model.add(Dense(1, name_output, activation='softmax'))
    return model
