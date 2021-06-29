from __future__ import annotations, division
from keras import layers
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Conv1D, Conv1DTranspose
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

# pd.set_option('display.max_colwidth', 500)
np.set_printoptions(threshold=sys.maxsize)
def generateMouseMovementsAE():
    ## Paths
    diffsFolder_1min = 'D:/Diplomadolgozat/Diffs/diffs_1min.csv'
    diffsFolder_3min = 'D:/Diplomadolgozat/Diffs/diffs_3min.csv'
    diffsSavePath = 'D:/Diplomadolgozat/DiffsGenAEConv/'

    diffs_1min = pd.read_csv(diffsFolder_1min)
    diffs_3min = pd.read_csv(diffsFolder_3min)

    Path(diffsSavePath).mkdir(parents=True, exist_ok=True)

    # df = selectUsers(diffs_3min, [1])
    # df2 = selectUsers(diffs_1min, [1])
    X_train = diffs_3min.iloc[:2, :128]
    X_test = diffs_1min.iloc[:2, :128]

    df = similarActionsWithAEConv(X_train, X_test)

def generateMouseMovementsAE2():
    ## Paths
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'
    dataSavePath = 'D:/Diplomadolgozat/UsersGenAEOneByOne/'
    actionsSavePath = 'D:/Diplomadolgozat/ActionsGenAEOneByOne/'

    Path(dataSavePath).mkdir(parents=True, exist_ok=True)
    Path(actionsSavePath).mkdir(parents=True, exist_ok=True)

    ## Initializing
    count_users = 0
    folders = 0

    # Iterating through user folders
    for dirname, _, filenames in os.walk(path_for_data):
        if count_users <= folders - 1:
            count_users += 1
            continue

        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue

        Path(dataSavePath + user + '/').mkdir(parents=True, exist_ok=True)
        Path(actionsSavePath + user.capitalize() + '/').mkdir(parents=True, exist_ok=True)

        read_data = {
            '1min' : pd.DataFrame({}),
            '3min' : pd.DataFrame({})
        }

        read_actions = {
            '1min' : pd.DataFrame({}),
            '3min' : pd.DataFrame({})
        }

        # Iterating through files in user's folder
        for filename in filenames:
            ## Show progress percentage
            # progress = (count_users * 100)/120
            # progress = "{:.2f}".format(progress)
            # print('\rProgress: ' + str(progress) + '%', end='')
            print(dirname, filename)

            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            dataPath = os.path.join(dirname, filename)
            read_data[minute] = pd.read_csv(dataPath)

            actionsPath = path_for_actions + user.capitalize() + '/' + minute + '.csv'
            read_actions[minute] = pd.read_csv(actionsPath)

        # After both files were read into DataFrames
        train = read_data['3min']
        test = read_data['1min']

        train = train.drop(columns=['button', 'state'])
        test = test.drop(columns=['button', 'state'])
        df = pd.DataFrame({})

        for column in train.columns:
            x_train = train[column]
            x_test = test[column]
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

            generated_data, loss = similarActionsWithAE(x_train, x_test)

            while (loss > 0.001):
                print('############################################')
                print('############ Loss was too high #############')
                print(loss, '--->', 0.001)
                print('############################################')
                generated_data, loss = similarActionsWithAE(x_train, x_test)

            df[column] = generated_data[:,0]

        # t = df['client timestamp']
        # t = np.sort(t)
        # x = df['x']
        # # y = df.iloc[:,2]

        # # Gen t
        # dt = np.diff(t)

        # # Real t
        # rt = test['client timestamp']
        # drt = np.diff(rt)

        # plt.figure()
        # plt.plot(drt)

        # plt.figure()
        # plt.plot(dt)

        # dx = np.diff(x)
        # rx = test['x']
        # drx = np.diff(rx)

        # plt.figure()
        # plt.plot(drx)

        # plt.figure()
        # plt.plot(dx)


        # plt.show()

        # return
        
        ## Save generated raw data and actions
        df = df.astype(int)
        df.to_csv(dataSavePath + user + '/1min.csv', index = False)
        read_actions['1min'].to_csv(actionsSavePath + user.capitalize()
            + '/1min.csv', index = False)
        
        return

def similarActionsWithAEConv(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    # scaler = MinMaxScaler(feature_range=(0,1))
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    # print(x_test)
    # plt.figure()
    # plt.plot(x_test[0])
    # plt.show()
    # return
    # x_train = x_train[0]
    # x_test = x_test[0]
    # x_train = np.reshape(x_train, (2, 128))
    # x_test = np.reshape(x_test, (2, 128))
    # x_train = x_train.reshape(-1, 2)
    # x_test = x_test.reshape(-1, 2)

    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # print(x_train.shape)
    # print(x_test.shape)

    rows, _ = x_train.shape
    # BATCH_SIZE = int(rows/50)
    BATCH_SIZE = 1
    EPOCHS = 25
    # input_shape = keras.Input(shape=(256, 1), batch_size=(256))
    input_shape = keras.Input(shape=(128, 1))
    fcn_filters = 128
    conv1 = Conv1D(filters = fcn_filters, kernel_size=3, padding='same', activation='relu')(input_shape)
    conv2 = Conv1D(filters = 2*fcn_filters, kernel_size=3, padding='same', activation='relu')(conv1)
    conv3 = Conv1D(filters = fcn_filters, kernel_size=3, padding='same', activation='relu')(conv2)

    conv3 = Conv1DTranspose( filters = fcn_filters, kernel_size=3, padding='same', activation='relu')( conv3 )
    conv2 = Conv1DTranspose( filters = 2*fcn_filters, kernel_size=3, padding='same', activation='relu')( conv3 )
    conv1 = Conv1DTranspose( filters = fcn_filters, kernel_size=3, padding='same', activation='relu')( conv2 )
    # decoded = Conv1D(filters=1, kernel_size=4, padding='same', activation='relu')(conv1)

    autoencoder = keras.Model(input_shape, conv1)
    autoencoder.summary()
    autoencoder.compile(optimizer='rmsprop', loss=keras.losses.mean_absolute_error)

    autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    # validation_data=(x_test, x_test),
                    batch_size=BATCH_SIZE,
                    shuffle=True
                    # callbacks=[TensorBoard(log_dir='/tmp/convAutoencoder')]
                    )

    decoded_data = autoencoder.predict(x_test)
    # decoded_data = np.array(decoded_data[0,:])
    # decoded_data = decoded_data.reshape(1,-1)
    # print(decoded_data)
    print(x_test.shape)
    print(x_test[0].shape)
    print(decoded_data.shape)
    print(decoded_data[:,:,0].shape)
    # decoded_data = scaler.inverse_transform(decoded_data)
    # x_test = x_test.reshape(-1, 1)
    # plt.figure()
    # plt.plot(x_test)
    # plt.figure()
    # plt.plot(decoded_data[0])
    # plt.show()
    decoded_data = decoded_data[:,:,0]
    print(decoded_data[0,:])


def similarActionsWithAE(x_train, x_test):
    standardScaler = MinMaxScaler(feature_range=(0,1))
    x_train = standardScaler.fit_transform(x_train)
    x_test = standardScaler.fit_transform(x_test)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    # BATCH_SIZE = 128
    rows, _ = x_train.shape
    BATCH_SIZE = int(rows/50)
    print('BATCH_SIZE: ', BATCH_SIZE)
    EPOCHS = 10
    # This is the size of our encoded representations
    encoding_dim = 4
    # The data dimension is 1, because 1 row * 1 col
    data_dim = 1
    # This is our input image
    input_img = keras.Input(shape=(data_dim,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # encoded = layers.Dense(16, activation='relu')(encoded)
    # encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    # decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    # decoded = layers.Dense(16, activation='relu')(decoded)
    decoded = layers.Dense(data_dim, activation='relu')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.summary()
    history = autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    plot_history(history)

    loss = autoencoder.evaluate(x_train, x_train,
                    batch_size=BATCH_SIZE)

    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    decoded_imgs = standardScaler.inverse_transform(decoded_imgs)
    return decoded_imgs, loss

def plot_history( history ):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # plt.savefig(stt.TRAINING_CURVES_PATH+'/' + model_name +  '.png', format='png')

# def similarActionsWithAE2(train, test):
#     VALIDATE_SIZE = 0.2
#     RANDOM_SEED = 42
#     EPOCHS = 100
#     BATCH_SIZE = 256

#     # setting random seeds for libraries to ensure reproducibility
#     np.random.seed(RANDOM_SEED)
#     rn.seed(RANDOM_SEED)
#     tf.compat.v1.set_random_seed(RANDOM_SEED)

#     X_train = train.drop(columns=['button', 'state'])
#     X_test = test.drop(columns=['button', 'state'])

#     X_train, X_validate = train_test_split(X_train, 
#                                        test_size=VALIDATE_SIZE, 
#                                        random_state=RANDOM_SEED)



#     # configure our pipeline
#     pipeline = Pipeline([('normalizer', Normalizer()),
#                         ('scaler', MinMaxScaler())])
    
#     # get normalization parameters by fitting to the training data
#     pipeline.fit(X_train)

#     # transform the training and validation data with these parameters
#     X_train_transformed = pipeline.transform(X_train)
#     X_validate_transformed = pipeline.transform(X_validate)

#     input_dim = 3

#     autoencoder = tf.keras.models.Sequential([
    
#         # deconstruct / encode
#         tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )), 
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
        
#         # reconstruction / decode
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(input_dim, activation='elu')
        
#     ])

#     # https://keras.io/api/models/model_training_apis/
#     autoencoder.compile(optimizer="adam", 
#                         loss="mse",
#                         metrics=["acc"])

#     history = autoencoder.fit(
#         X_train_transformed, X_train_transformed,
#         shuffle=False,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(X_validate_transformed, X_validate_transformed)
#     )

#     # transform the test set with the pipeline fitted to the training set
#     X_test_transformed = pipeline.transform(X_test)

#     # pass the transformed test set through the autoencoder to get the reconstructed result
#     reconstructions = autoencoder.predict(X_test_transformed)
#     return reconstructions
