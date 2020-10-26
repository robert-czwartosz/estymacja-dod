from config import config
cfg = config()
import numpy as np
import pickle
import math
from sklearn.preprocessing import StandardScaler

fileOut = open('.\\'+cfg.OutputDirPrior+"\\Out.pkl",'rb')
Out = pickle.load(fileOut)
fileOut.close()
#Out = Out[-100:]


fileIn = open('.\\'+cfg.InputDirPrior+"\\In.pkl",'rb')
In = pickle.load(fileIn)
fileIn.close()
#In = In[-100:]

InShape = np.shape(In)
OutShape = np.shape(Out)
Nsamples = int(InShape[0])
N = cfg.N
n = cfg.n
m = cfg.m
print(np.shape(In))
print(np.shape(Out))
print(N,n,m)
import matplotlib.pyplot as plt
import statsmodels.api as sm


from sklearn.model_selection import train_test_split

# =======================
# PRZYGOTOWANIE DANYCH
# ======================

delta = cfg.delta
Output = np.zeros((Nsamples*(N-delta),m))
Input = np.zeros((Nsamples*(N-delta),n,n,delta+1))

for sample in range(Nsamples):
    for i in range(delta, N):
        Input[sample*(N-delta) + i-delta] = In[sample, :, :, i-delta:i+1]
        Output[sample*(N-delta) + i-delta] = Out[sample, i]


train_X = Input[:int(0.7*Nsamples)*(N-delta)]
validation_X = Input[int(0.7*Nsamples)*(N-delta):int(0.9*Nsamples)*(N-delta)]
test_X = Input[int(0.9*Nsamples)*(N-delta):]

train_Y = Output[:int(0.7*Nsamples)*(N-delta)]
validation_Y = Output[int(0.7*Nsamples)*(N-delta):int(0.9*Nsamples)*(N-delta)]
test_Y = Output[int(0.9*Nsamples)*(N-delta):]

train_X = train_X.astype('float32')
validation_X  = validation_X.astype('float32')
test_X = test_X.astype('float32')
train_Y = train_Y.astype('float32')
validation_Y  = validation_Y.astype('float32')
test_Y = test_Y.astype('float32')

scalerX = StandardScaler()
scalerX.fit(train_X.reshape(train_X.shape[0],-1))
scalerY = StandardScaler()
scalerY.fit(train_Y.reshape(train_Y.shape[0],-1))
if cfg.scaling:
    train_X = scalerX.transform(train_X.reshape(train_X.shape[0],-1)).reshape(train_X.shape)
    validation_X  = scalerX.transform(validation_X.reshape(validation_X.shape[0],-1)).reshape(validation_X.shape)
    test_X = scalerX.transform(test_X.reshape(test_X.shape[0],-1)).reshape(test_X.shape)
    train_Y = scalerY.transform(train_Y.reshape(train_Y.shape[0],-1)).reshape(train_Y.shape)
    validation_Y  = scalerY.transform(validation_Y.reshape(validation_Y.shape[0],-1)).reshape(validation_Y.shape)
    test_Y = scalerY.transform(test_Y.reshape(test_Y.shape[0],-1)).reshape(test_Y.shape)

print('Train dataset shape:', train_X.shape, 
      '\tValidation dataset shape:', validation_X.shape)

print(train_X[0].shape)
print(train_Y[0].shape)
"""
N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
#x = test_Y[:(N-delta)].flatten()
x = test_Y.flatten()
print(np.shape(x))
# We can set the number of bins with the `bins` kwarg
plt.hist(x, bins=n_bins)
#axs[1].hist(x, bins=n_bins)
plt.show()
"""
#print(train_X[0])
#print(train_Y[:10])

# ===========================
# MODEL W TENSORFLOW 1.14
# =================
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Layer, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.compat.v1.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import EarlyStopping

from saaf import DenseSAAF
import tensorflow as tf


"""
x = tf.ones((3, 7))
linear_layer = DenseSAAF(4, 15, 2)
y = linear_layer(x)
print(y)
#exit()
"""
#==========================================
# STRUKTURA SIECI CNN
#=======================================
batch_size = cfg.batch_size
epochs = cfg.epochs
input_shape = (n, n, delta+1)
input_shape = np.shape(train_X[0])
tf.keras.backend.set_epsilon(1e-6)
# early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=cfg.early_stopping)
numCNNFilters = cfg.numCNNFilters
model = Sequential()
for i,numFilters in enumerate(numCNNFilters):
    if i == 0:
        model.add(Conv2D(numFilters, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer=RandomNormal(mean=0.0, stddev=math.sqrt(2/(n*n*9*(delta+1)*numFilters)), seed=None),
        bias_initializer='zeros',
        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    else:
        model.add(Conv2D(numFilters, kernel_size=(3, 3), activation='relu', 
            padding='same', 
            kernel_initializer=RandomNormal(mean=0.0, stddev=math.sqrt(2/(numCNNFilters[i-1]*int(n-i)*int(n-i)*9*numFilters)), seed=None),
            bias_initializer='zeros'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

if len(numCNNFilters) > 0:
    model.add(Flatten())
    model.add(DenseSAAF(m))
else:
    model.add(Dense(5000, activation='relu', input_shape=input_shape))
    model.add(DenseSAAF(m))
#model.add(Dense(m, activation='linear'))

adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss=tf.keras.losses.Huber(),#loss='mae',
              optimizer=adam,
              metrics=['mse', 'mae'])

model.summary()



# =================
# TRENING
# ==================

history = model.fit(x=train_X, y=train_Y,
                    validation_data=(validation_X, validation_Y),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, callbacks=[es])

model.save('Assigner.h5')


# =================
# WIZUALIZACJA
# =================
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ==========================================
# SPRAWDZENIE NA DANYCH TESTOWYCH
# =========================================

test_preds = model.predict(
    test_X,
    batch_size=None,
    verbose=0,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)

def MAEgt5(Yreal,Y):
    return np.average(np.abs(Yreal - Y)[Yreal > 5])
def MAE(Yreal,Y):
    return np.average(np.abs(Yreal - Y))
def MSE(Yreal,Y):
    return np.average((Yreal - Y)*(Yreal - Y))

def MAPEgt5(Yreal,Y):
    return 100*np.average((np.abs(Yreal - Y)/(np.maximum(Yreal,5)))[Yreal > 5])
def MAPE(Yreal,Y):
    return 100*np.average(np.abs(Yreal - Y)/(np.maximum(Yreal,1)))

def R2(Y_real, Y_pred):
    return 1 - np.sum((Y_pred - Y_real)**2)/np.sum((Y_real - np.average(Y_real))**2)

def ToStr(Th, Tmin):
    if Tmin < 10:
        return str(Th)+'.0'+str(Tmin)
    return str(Th)+':'+str(Tmin)

dT = str(cfg.dT)
for i in range(N-delta):
    T0 = 7 * 60 + (delta+i)*int(dT)
    T0h = int(T0/60)
    T0min = int(T0%60)
    
    T1 = T0 + int(dT)
    T1h = int(T1/60)
    T1min = int(T1%60)
    
    X = test_preds[i::(N-delta)]
    Y = test_Y[i::(N-delta)]
    
    if cfg.scaling:
        X = scalerY.inverse_transform(X.reshape(X.shape[0],-1)).reshape(X.shape)
        Y = scalerY.inverse_transform(Y.reshape(Y.shape[0],-1)).reshape(Y.shape)
    X = X.flatten()
    Y = Y.flatten()
    results = sm.OLS(Y,sm.add_constant(X)).fit()

    print(results.summary())

    plt.scatter(X,Y)

    plt.plot(X, results.fittedvalues,color='orange')
    plt.plot(results.fittedvalues, results.fittedvalues,color='green')


    plt.title('Comparisons of the assigned and observed flows of the monitored links\n\
    and turning movements during TI'+str(delta+i+1)+'('+ToStr(T0h,T0min)+' AM ~ '+ToStr(T1h,T1min)+' AM)')
    plt.xlabel('Assigned Link Flows (veh/'+dT+' min)')
    plt.ylabel('Observed Link Flows (veh/'+dT+' min)')

    a = '1.00'
    b = '0.00'
    s = 'asd'
    if len(results.params) == 2:
        a = str(round(results.params[1], 4))
        if results.params[0] < 0:
            b = '- '+str(round(abs(results.params[0]), 4))
        else:
            b = '+ '+str(round(abs(results.params[0]), 4))
    else:
        a = str(results.params[0])
    s = 'y = '+a+'x '+b+'\n'+\
            r'$R^2$ = '+str(round(results.rsquared, 4))
    plt.text(0.4*np.max(X), 0.9*np.max(Y), s, multialignment="center")
    print('MAE: ',MAE(Y,X),' ('+str(MAEgt5(Y,X))+')')
    print('MAPE: ',MAPE(Y,X),'% ('+str(MAPEgt5(Y,X))+'%)')
    print('R^2: ', R2(Y, X))
    plt.show()


X = test_preds
Y = test_Y
if cfg.scaling:
    X = scalerY.inverse_transform(X.reshape(X.shape[0],-1)).reshape(X.shape)
    Y = scalerY.inverse_transform(Y.reshape(Y.shape[0],-1)).reshape(Y.shape)

print('MAE: ',MAE(Y,X),' ('+str(MAEgt5(Y,X))+')')
print('MAPE: ',MAPE(Y,X),'% ('+str(MAPEgt5(Y,X))+'%)')
print('MSE: ',MSE(Y,X))
print('R^2: ', R2(Y, X))