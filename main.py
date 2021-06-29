import functions as f
import pandas as pd

# ActivitySim
# https://activitysim.github.io/populationsim/application_configuration.html

# Autoencoder
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://www.kaggle.com/robinteuwens/anomaly-detection-with-auto-encoders/data#Train/Validate/Test-split
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# https://keras.io/examples/vision/autoencoder/

# conv1 = Conv1D(filters = fcn_filters, kernel_size=8, padding='same', activation='relu')(input_layer)
# conv3 = Conv1DTranspose( filters = fcn_filters, kernel_size=3, padding='same', activation='relu')( h )

# X = X.reshape(-1, input_size, input_dim)
# input_size = 128
# input_dim = 2

# X.shape --> 256 x 1
# reshape utan
# X.shape -->128 x 2

def main():
	# f.mergeData()
	# f.subplotActions(1)
	# f.generateMouseMovementsSynth()
	# f.generateMouseMovementsAE2()
	# f.generateMouseMovementsAE()
	# f.extractFeatures(2)
	# f.binaryClassification(3)
	# f.boxplots()
	# f.plotXDirectionalDynamics(0)
	# f.dataToDiffs()
	# f.plotTrajectories(10, 30)
	# f.sessionHistograms()
	# f.plotSingleAction(1,22,1)
	# f.printNumOfActionsPerUser()
	f.plotBezierForUI()

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
	# f.classification(test, users, mode)

	f.trainTest(train, test, users, mode)
	print('\nDone.')

if __name__ == '__main__':
	main()