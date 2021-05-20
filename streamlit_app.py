#INSTALAR LOS PAQUETES NECESARIOS
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import streamlit as st
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam

print('###############PREDICCION SOLAR###############')

st.title('PREDICCIÓN GENERACION PLANTA SOLAR')

#IMPORTAR EL TRAINING SET
dataset_train = pd.read_csv('DATOS-SOLAR-LA-GUAJIRA.csv')

st.write('Datos históricos La Guajira: IDEAM')
st.dataframe(pd.DataFrame(dataset_train))

#EXTRAER FECHAS
datelist_train = list(dataset_train['Fecha'])
datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M').date() for date in datelist_train]

#TOMAR LAS COLUMNAS NECESARIAS
cols = list(dataset_train)[1:6]

#CONVERTIR LOS TITULOS DEL DATASET A TIPO STRING
dataset_train = dataset_train[cols].astype(str)
#QUITAR LAS COMAS
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

print(dataset_train)

#CONVERTIR EL DATASET A TIPO FLOAT
dataset_train = dataset_train.astype(float)
#CONVERTIR EL DATASET A NUMPY
training_set = dataset_train.to_numpy()

#ESCALAR DATOS PARA QUE ENTRE -1 Y 1
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])

# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

#HORAS FUTURAS A PREDECIR
n_future = 24
#HORAS PASADAS PARA PREDECIR
n_past = 504

for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#CREAR EL MODELO
model = Sequential()
#PRIMERA CAPA
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))
#SEGUNDA CAPA
model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.2))
#CAPA DE SALIDA
model.add(Dense(units=1, activation='linear'))
#COMPILAR EL MODELO
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

#ENTRENAR EL MODELO
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')

history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)

#MOSTRAR ENTRENAMIENTO
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.legend(['loss','val_loss'])
#plt.grid()
#plt.show()

datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1H').tolist()

datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

predictions_future = model.predict(X_train[-n_future:])

predictions_train = model.predict(X_train[n_past:])

def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d %H%M'), '%Y%m%d %H%M')

y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Generacion']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Generacion']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

st.write('Prediccion de generación para las proximas '+str(n_future)+' horas')
st.dataframe(pd.DataFrame(PREDICTIONS_FUTURE))

hora = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

#plt.plot(hora,PREDICTIONS_FUTURE)
#plt.title('Generacion al día siguiente')
#plt.xlabel('Hora')
#plt.ylabel('Potencia [MW]')
#plt.grid()
#plt.show()

char_data = pd.DataFrame(PREDICTIONS_FUTURE)
st.line_chart(char_data)
