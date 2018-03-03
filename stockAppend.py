

# coding: utf-8

# In[69]:


# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing


# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET 
dataset = pd.read_csv('COSMOFILMS.csv', usecols=[1,2,3,4,5,6,7,8])
dataset = dataset.reindex(index = dataset.index[::-1])
dataset = dataset.fillna(0)


# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)


close_val = dataset[['Last']].mean(axis = 1)
def net(close_val,step):
	nextValues = []
	for i in range(step):

		# PREPARATION OF TIME SERIES DATASE

		OHLC_avg = np.reshape(close_val.values, (len(close_val),1)) 
		scaler = MinMaxScaler(feature_range=(0, 1))
		OHLC_avg = scaler.fit_transform(OHLC_avg)

		# TRAIN-TEST SPLIT
		train_OHLC = int(len(OHLC_avg) * .99)
		test_OHLC = len(OHLC_avg) - train_OHLC
		train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]


		# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
		trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
		testX, testY = preprocessing.new_dataset(test_OHLC, 1)


		# RESHAPING TRAIN AND TEST DATA
		trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
		testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
		step_size = 1


		# LSTM MODEL
		model = Sequential()
		model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
		model.add(LSTM(16))
		model.add(Dense(1))
		model.add(Activation('linear'))



		# MODEL COMPILING AND TRAINING
		model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
		model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

		# PREDICTION
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)



		# DE-NORMALIZING FOR PLOTTING
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform([trainY])
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([testY])


		# TRAINING RMSE
		trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
		print('Train RMSE: %.2f' % (trainScore))

		# TEST RMSE
		testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
		print('Test RMSE: %.2f' % (testScore))


		# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
		trainPredictPlot = np.empty_like(OHLC_avg)
		trainPredictPlot[:, :] = np.nan
		trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict
		OHLC_avg


		# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
		testPredictPlot = np.empty_like(OHLC_avg)
		testPredictPlot[:, :] = np.nan
		testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict


		# DE-NORMALIZING MAIN DATASET 
		OHLC_avg = scaler.inverse_transform(OHLC_avg)


		# PREDICT FUTURE VALUES
		
		last_val= pd.Series(testPredict[-1])

		close_val.append(last_val)

		nextValues.append(last_val)


	return nextValues
# NUMBER OF VALUES TO PREDICT
step = 5


series = net(close_val,step)
print (series)
