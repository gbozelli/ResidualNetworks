# importação das bibliotecas necessárias

import numpy as np
import pandas as pd #leitura dos dados
from matplotlib import pyplot as plt #plotagem dos gráficos
from sklearn import preprocessing #pre-processamento dos dados
from sklearn import neural_network #importação da rede neural
from sklearn.metrics import mean_squared_error as loss #importação da função de perda
import pickle
import time
from sklearn.utils import shuffle

#matrizes de dados. Os números correspondem à distância de transmissão, 'real' corresponde
#aos dados recebidos e 'exp' aos dados desejados
real_data_135, real_data_175, exp_data_135, exp_data_175 = [], [], [], []
real_data_120 = []
exp_data_120 = []
real_data_110 = []
exp_data_110 = []
real_data_100 = []
exp_data_100 = []
real_data_90 = []
exp_data_90 = []
real_data_80 = []
exp_data_80 = []

for i in range(16):
  file_175km = 'data/DP_RealConstellationDiagram_'+str(i)+'dbm_175km.csv' #175km
  file_exp_175km = 'data/DP_IdealConstellationDiagram_0dbm_175km.csv'
  file_135km = 'data/DP_RealConstellationDiagram_'+str(i)+'dbm.csv' #135km
  data_175km = np.transpose(pd.read_csv(file_175km, sep=',', skiprows=5).values.tolist())
  exp_175km = np.transpose(pd.read_csv(file_exp_175km, sep=',', skiprows=5).values.tolist())
  if i < 13: 
    data_X120_exp = np.transpose(pd.read_csv('dados_1_canal_ref_X_pol_canal1.csv', sep=',', skiprows=6).values.tolist())
    data_Y120_exp = np.transpose(pd.read_csv('dados_1_canal_ref_X_pol_canal1.csv', sep=',', skiprows=6).values.tolist())
    file_X120km = 'dados_120km_canal1/dados_1_canal_canal_1_120km_X_pol_'+str(i)+'dBm_por_canal.csv'
    data_X120km = np.transpose(pd.read_csv(file_X120km, sep=',', skiprows=6).values.tolist())
    file_Y120km = 'dados_120km_canal1/dados_1_canal_canal_1_120km_Y_pol_'+str(i)+'dBm_por_canal.csv'
    data_Y120km = np.transpose(pd.read_csv(file_Y120km, sep=',', skiprows=6).values.tolist())
    exp_data_120.append([data_X120_exp[0], data_X120_exp[1], data_Y120_exp[0], data_Y120_exp[1]])
    real_data_120.append([data_X120km[0], data_X120km[1],data_Y120km[0], data_Y120km[1]])
    
    data_X110_exp = np.transpose(pd.read_csv('dados_X_ref.csv', sep=',', skiprows=6).values.tolist())
    data_Y110_exp = np.transpose(pd.read_csv('dados_Y_ref.csv', sep=',', skiprows=6).values.tolist())
    file_X110km = 'dados_110km/dados_110km_'+str(i)+'dBm/dados_X_110km_'+str(i)+'dBm.csv'
    data_X110km = np.transpose(pd.read_csv(file_X120km, sep=',', skiprows=6).values.tolist())
    file_Y110km = 'dados_110km/dados_120km_'+str(i)+'dBm/dados_Y_110km_'+str(i)+'dBm.csv'
    data_Y110km = np.transpose(pd.read_csv(file_Y120km, sep=',', skiprows=6).values.tolist())
    exp_data_110.append([data_X110_exp[0], data_X110_exp[1], data_Y110_exp[0], data_Y110_exp[1]])
    real_data_110.append([data_X110km[0], data_X110km[1],data_Y110km[0], data_Y110km[1]])
    
    data_X100_exp = np.transpose(pd.read_csv('dados_X_ref.csv', sep=',', skiprows=6).values.tolist())
    data_Y100_exp = np.transpose(pd.read_csv('dados_Y_ref.csv', sep=',', skiprows=6).values.tolist())
    file_X100km = 'dados_100km/dados_100km_'+str(i)+'dBm/dados_X_100km_'+str(i)+'dBm.csv'
    data_X100km = np.transpose(pd.read_csv(file_X100km, sep=',', skiprows=6).values.tolist())
    file_Y100km = 'dados_100km/dados_100km_'+str(i)+'dBm/dados_Y_100km_'+str(i)+'dBm.csv'
    data_Y100km = np.transpose(pd.read_csv(file_Y100km, sep=',', skiprows=6).values.tolist())
    exp_data_100.append([data_X100_exp[0], data_X100_exp[1], data_Y100_exp[0], data_Y100_exp[1]])
    real_data_100.append([data_X100km[0], data_X100km[1],data_Y100km[0], data_Y100km[1]])

    data_X90_exp = np.transpose(pd.read_csv('dados_X_ref.csv', sep=',', skiprows=6).values.tolist())
    data_Y90_exp = np.transpose(pd.read_csv('dados_Y_ref.csv', sep=',', skiprows=6).values.tolist())
    file_X90km = 'dados_90km/dados_90km_'+str(i)+'dBm/dados_X_90km_'+str(i)+'dBm.csv'
    data_X90km = np.transpose(pd.read_csv(file_X90km, sep=',', skiprows=6).values.tolist())
    file_Y90km = 'dados_90km/dados_90km_'+str(i)+'dBm/dados_Y_90km_'+str(i)+'dBm.csv'
    data_Y90km = np.transpose(pd.read_csv(file_Y90km, sep=',', skiprows=6).values.tolist())
    exp_data_90.append([data_X90_exp[0], data_X90_exp[1], data_Y90_exp[0], data_Y90_exp[1]])
    real_data_90.append([data_X90km[0], data_X90km[1],data_Y90km[0], data_Y90km[1]])

    data_X80_exp = np.transpose(pd.read_csv('dados_X_ref.csv', sep=',', skiprows=6).values.tolist())
    data_Y80_exp = np.transpose(pd.read_csv('dados_Y_ref.csv', sep=',', skiprows=6).values.tolist())
    file_X80km = 'dados_80km/dados_80km_'+str(i)+'dBm/dados_X_'+str(i)+'dBm.csv'
    data_X80km = np.transpose(pd.read_csv(file_X80km, sep=',', skiprows=6).values.tolist())
    file_Y80km = 'dados_80km/dados_80km_'+str(i)+'dBm/dados_Y_'+str(i)+'dBm.csv'
    data_Y80km = np.transpose(pd.read_csv(file_Y80km, sep=',', skiprows=6).values.tolist())
    exp_data_80.append([data_X80_exp[0], data_X80_exp[1], data_Y80_exp[0], data_Y80_exp[1]])
    real_data_80.append([data_X80km[0], data_X80km[1],data_Y80km[0], data_Y80km[1]])


  if i < 13:
    data_135km = np.transpose(pd.read_csv(file_135km, sep=',', skiprows=5).values.tolist())

    #adiciona as 4 componentes correspondentes à potência
    real_data_135.append([data_135km[4], data_135km[5], data_135km[6], data_135km[7]])
    exp_data_135.append([data_135km[0], data_135km[1], data_135km[2], data_135km[3]])
  real_data_175.append([data_175km[0], data_175km[1], data_175km[2], data_175km[3]])
  exp_data_175.append([exp_175km[0], exp_175km[1], exp_175km[2], exp_175km[3]])

def split(data, n):
  #separa os dados para treino e para teste
  t_data = np.transpose(data)
  train, test = t_data[0:n], t_data[n:]
  train, test = np.transpose(train), np.transpose(test)
  return train, test

n_1 = int(0.8*262143)
n_2 = int(0.8*262138)
real_data_150_10dbm = np.transpose(pd.read_csv('data/DP_RealConstellationDiagram_10dbm_150km_14e9Bd.csv', sep=',', skiprows=5).values.tolist())
real_data_150_9dbm = np.transpose(pd.read_csv('data/DP_RealConstellationDiagram_9dbm_150km_14e9Bd.csv', sep=',', skiprows=5).values.tolist())
real_data_150_8dbm = np.transpose(pd.read_csv('data/DP_RealConstellationDiagram_8dbm_150km_14e9Bd.csv', sep=',', skiprows=5).values.tolist())
real_data_150_7dbm = np.transpose(pd.read_csv('data/DP_RealConstellationDiagram_7dbm_150km_14e9Bd.csv', sep=',', skiprows=5).values.tolist())
real_data_150 = np.array([real_data_150_10dbm, real_data_150_9dbm, real_data_150_8dbm, real_data_150_7dbm])
x_train_175km, x_test_175km = split(real_data_175, n_1)
y_train_175km, y_test_175km = split(exp_data_175, n_1)
x_train_135km, x_test_135km = split(real_data_135, n_2)
x_train_120km, x_test_120km = split(real_data_120, n_1)
y_train_120km, y_test_120km = split(exp_data_120, n_1)
y_train_135km, y_test_135km = split(exp_data_135, n_2)

def BER(X_test, Y_test, clf):
  Y_test_hat = clf.predict(X_test)
  aux = 4*np.clip(np.round((Y_test[:,0]-1)/2)+2,0,3)+np.clip(np.round((Y_test[:,1]-1)/2)+2,0,3)
  sym_X_test = aux.astype(int)
  # Y polarization
  aux = 4*np.clip(np.round((Y_test[:,2]-1)/2)+2,0,3)+np.clip(np.round((Y_test[:,3]-1)/2)+2,0,3)
  sym_Y_test = aux.astype(int)
  # Detection of the real symbols with multiple symbol
  # X polarization
  aux = 4*np.clip(np.round((Y_test_hat[:,0]-1)/2)+2,0,3)+np.clip(np.round((Y_test_hat[:,1]-1)/2)+2,0,3)
  sym_X_test_Multisym = aux.astype(int)
  # Y polarization
  aux = 4*np.clip(np.round((Y_test_hat[:,2]-1)/2)+2,0,3)+np.clip(np.round((Y_test_hat[:,3]-1)/2)+2,0,3)
  sym_Y_test_Multisym = aux.astype(int)
  # Corrected to calculate BER instead of SER
  BER = (sum(sym_X_test!=sym_X_test_Multisym)/len(sym_X_test)+sum(sym_Y_test!=sym_Y_test_Multisym)/len(sym_Y_test))/8
  return BER

import math

def Windowing(x, y, n):
  x_mult = []
  n_elem = len(x[0])-n
  for i in range(0,n):
    x_mult.append(x[0][i:n_elem+i])
    x_mult.append(x[1][i:n_elem+i])
    x_mult.append(x[2][i:n_elem+i])
    x_mult.append(x[3][i:n_elem+i])
  y = y[:,int(n/2):n_elem+int(n/2)]
  return np.array(x_mult), np.array(y)

import tensorflow as tf
from tensorflow.keras import layers, Model

class WindowingLayer(layers.Layer):
    def __init__(self, n, **kwargs):
        super(WindowingLayer, self).__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        # inputs shape: (4, length)
        # We need to slice and stack according to your logic
        n_elem = tf.shape(inputs)[1] - self.n
        
        x_mult = []
        for i in range(self.n):
            # Slicing the 4 channels
            slice_i = inputs[:, i : n_elem + i] # Shape: (4, n_elem)
            x_mult.append(tf.transpose(slice_i)) # Transpose to (n_elem, 4)
            
        # Concatenate all windows along the feature axis or sample axis 
        # based on your specific goal. 
        # For a standard ResNet MLP, we stack them:
        return tf.concat(x_mult, axis=0) 
    def compute_output_shape(self, input_shape):
        # This tells the Dense layer that the last dimension is 4
        return (4*self.n,None)

import numpy as np
from tensorflow.keras import layers, Model
from sklearn import preprocessing

x = np.array(real_data_120[8]) 
y = np.array(exp_data_120[8])

scaler = preprocessing.StandardScaler()

x = scaler.fit_transform(x.T)
y = scaler.transform(y.T)

split_idx = 180000
X_train, X_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

inputs = layers.Input(shape=(4,)) 
mlp_layer = layers.Dense(64, activation='relu')(inputs) 
mlp_output = layers.Dense(4, activation='linear')(mlp_layer)

model = Model(inputs=inputs, outputs=mlp_output, name="Simple_MLP")
model.compile(optimizer='adam', loss='mse')


model.fit(X_train, y_train, epochs=5)

print("Calculating BER...")

print(BER(X_test, y_test, model))

inputs = layers.Input(shape=(None,), batch_size=4)
windowing = WindowingLayer(n=11)(inputs)
mlp_layer = layers.Dense(64, activation='relu')(windowing)
mlp_output = layers.Dense(4, activation='linear')(mlp_layer)
outputs = layers.Add()([mlp_output, inputs])
model = Model(inputs=inputs, outputs=outputs, name="Simple_ResNet_MLP")
model.summary()

#Simple ResNet model using data manipulation

import numpy as np
from tensorflow.keras import layers, Model
from sklearn import preprocessing

def BER(X_test, Y_test, x_test_c, clf):
  Y_test_hat = clf.predict(X_test)
  Y_test_hat +=  x_test_c
  aux = 4*np.clip(np.round((Y_test[:,0]-1)/2)+2,0,3)+np.clip(np.round((Y_test[:,1]-1)/2)+2,0,3)
  sym_X_test = aux.astype(int)
  # Y polarization
  aux = 4*np.clip(np.round((Y_test[:,2]-1)/2)+2,0,3)+np.clip(np.round((Y_test[:,3]-1)/2)+2,0,3)
  sym_Y_test = aux.astype(int)
  # Detection of the real symbols with multiple symbol
  # X polarization
  aux = 4*np.clip(np.round((Y_test_hat[:,0]-1)/2)+2,0,3)+np.clip(np.round((Y_test_hat[:,1]-1)/2)+2,0,3)
  sym_X_test_Multisym = aux.astype(int)
  # Y polarization
  aux = 4*np.clip(np.round((Y_test_hat[:,2]-1)/2)+2,0,3)+np.clip(np.round((Y_test_hat[:,3]-1)/2)+2,0,3)
  sym_Y_test_Multisym = aux.astype(int)
  # Corrected to calculate BER instead of SER
  BER = (sum(sym_X_test!=sym_X_test_Multisym)/len(sym_X_test)+sum(sym_Y_test!=sym_Y_test_Multisym)/len(sym_Y_test))/8
  return BER

def Windowing(x, y, n):
  x_mult = []
  n_elem = len(x[0])-n
  for i in range(0,n):
    x_mult.append(x[0][i:n_elem+i])
    x_mult.append(x[1][i:n_elem+i])
    x_mult.append(x[2][i:n_elem+i])
    x_mult.append(x[3][i:n_elem+i])
  y = y[:,int(n/2):n_elem+int(n/2)]
  x_cent = x[:,int(n/2):n_elem+int(n/2)]
  return np.array(x_cent), np.array(x_mult), np.array(y)

x = np.array(real_data_120[8]) 
y = np.array(exp_data_120[8])

scaler = preprocessing.StandardScaler()

x = scaler.fit_transform(x.T).T

x_cent, x_w, y_cent = Windowing(x, y, 3)

y_residual = (y_cent - x_cent).T

x_w = x_w.T
y = y.T
y_cent = y_cent.T
x_cent = x_cent.T

split_idx = 180000
X_train, X_test = x_w[:split_idx], x_w[split_idx:]
y_train, y_test = y_residual[:split_idx], y_residual[split_idx:]
y_train_s, y_test_s = y_cent[:split_idx], y_cent[split_idx:]
x_train_s, x_test_s = x_cent[:split_idx], x_cent[split_idx:]

inputs = layers.Input(shape=(4*3,)) 
mlp_layer = layers.Dense(64, activation='relu')(inputs) 
mlp_output = layers.Dense(4, activation='linear')(mlp_layer)

model = Model(inputs=inputs, outputs=mlp_output, name="Simple_Resnet_MLP")
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=5)

print("Calculating BER...")
print('X_test:', np.shape(X_test))
print('y_test_s:', np.shape(y_test_s))
print('BER:', BER(X_test, y_test_s, x_test_s, model))