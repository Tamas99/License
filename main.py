import pandas as pd
from preprocessing_data.data_manipulation import mergeData
from data_analysis.boxplots import boxplots
from data_analysis.subplots import subplotActions
from data_analysis.histograms import sessionHistograms
from generate_datasets.bezier_dataset import generateMouseMovementsBezier
from generate_datasets.synthesized_dataset import generateMouseMovementsSynth
from generate_datasets.autoencoder_dataset import generateMouseMovementsAE, generateMouseMovementsAE2
from feature_extraction.extract_features import extractFeatures
from classifications.user_classification import trainTest, classification
from classifications.binary_classification import binaryClassification
from plotting.x_directional_dynamics import plotXDirectionalDynamics
from feature_extraction.extract_features import dataToDiffs
from plotting.trajectories import plotTrajDxDy, plotTrajectoriesOnSingleDiagram
from plotting.trajectories import plotSingleActionOnSingleDiagram
from data_analysis.histograms import NumOfActionsToString
from data_analysis.histograms import countActionsPerUser
from plotting.trajectories import plotBezierTrajDxDy

# Autoencoder
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://www.kaggle.com/robinteuwens/anomaly-detection-with-auto-encoders/data#Train/Validate/Test-split
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# https://keras.io/examples/vision/autoencoder/

def main():
	# mode = 0
	# users = range(1,11)
	# trainPath = 'D:/Diplomadolgozat/Features/sapimouse_3min_strong.csv'
	# testPath = 'D:/Diplomadolgozat/Features/sapimouse_1min_strong.csv'
	# if mode == 1:
	# 	trainPath = 'D:/Diplomadolgozat/Features/sapimouse_3min.csv'
	# 	testPath = 'D:/Diplomadolgozat/Features/sapimouse_1min.csv'
	# train = pd.read_csv(trainPath)
	# test = pd.read_csv(testPath)
	# classification(test, users, mode)
	# trainTest(train, test, users, mode)
	
	# mergeData()
	# subplotActions(1)
	# generateMouseMovementsBezier()
	# generateMouseMovementsSynth()
	# generateMouseMovementsAE2()
	# generateMouseMovementsAE()
	# extractFeatures(2)
	# binaryClassification(3)
	# boxplots()
	# plotXDirectionalDynamics(0)
	# dataToDiffs()
	# plotTrajectoriesOnSingleDiagram(10, 30)
	# sessionHistograms()
	# plotSingleActionOnSingleDiagram(1,22,1)
	# printNumOfActionsPerUser()
	# plotBezierTrajDxDy()
	plotTrajDxDy()
	# NumOfActionsToString()

def main1():
	mode = 0
	users = range(1,11)
	trainPath = 'D:/Diplomadolgozat/Features/sapimouse_3min_strong.csv'
	testPath = 'D:/Diplomadolgozat/Features/sapimouse_1min_strong.csv'
	if mode == 1:
		trainPath = 'D:/Diplomadolgozat/Features/sapimouse_3min.csv'
		testPath = 'D:/Diplomadolgozat/Features/sapimouse_1min.csv'
	train = pd.read_csv(trainPath)
	test = pd.read_csv(testPath)
	# classification(test, users, mode)
	trainTest(train, test, users, mode)


if __name__ == '__main__':
	main()