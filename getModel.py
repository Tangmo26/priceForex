import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, LeakyReLU, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, f1_score, auc
import joblib

# Load your data
Curency = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF', 'USDCAD']
textTime = ['5m', '15m', '30m', '1h', '2h', '4h']
State = ['buy','sell']
path = ['modelBuy', 'modelSell']
pathScaler = ['scalerBuy', 'scalerSell']


os.makedirs(path[0], exist_ok=True)
os.makedirs(path[1], exist_ok=True)
os.makedirs(pathScaler[0], exist_ok=True)
os.makedirs(pathScaler[1], exist_ok=True)


for i in range(len(State)) :
    for currency in Curency :
        for time in textTime :
            df = pd.read_csv(f'{State[i]}Data/{State[i]}_{currency}{time}.csv')
            # Preprocess the data
            df['result'] = df['result'].replace({'win': 1, 'loss': 0})
            df = df.drop(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'EMA50', 'EMA200', 'cross'])
            df = df.astype('float32')
            
            # Filter rows with 'win' label for x_test
            x_test_win = df[df['result'] == 1].tail(40)
            y_test_win = x_test_win['result']

            # Filter rows with 'loss' label for x_test
            x_test_loss = df[df['result'] == 0].tail(60)
            y_test_loss = x_test_loss['result']

            # Concatenate the win and loss test sets
            x_test = pd.concat([x_test_win, x_test_loss])
            y_test = pd.concat([y_test_win, y_test_loss])

            # Drop the rows used for testing from the original DataFrame
            df = df.drop(index=x_test.index)

            # Shuffle the test sets if needed
            x_new_test = x_test.sample(frac=1, random_state=42)
            x_new_test = x_new_test.drop(columns=['result'])
            y_new_test = y_test.sample(frac=1, random_state=42)

            x = df.drop(columns=['result'])
            y = df['result']

            # Split the data into training and testing sets
            x_train = x
            y_train = y
            
            print(y_new_test.value_counts())
            

            # Standardize the features
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)

            # Reshape the input data for LSTM
            x_train_reshaped = np.reshape(x_train_scaled, (x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))

            x_new_test_scaled = scaler.transform(x_new_test)
            x_new_test_reshaped = np.reshape(x_new_test_scaled, (x_new_test_scaled.shape[0], 1, x_new_test_scaled.shape[1]))
            
            joblib.dump(scaler, f'{pathScaler[i]}/scaler_{currency}{time}.joblib')

            for k in range(1000000) :
               # Build the model
                print(y_train.value_counts())
                model = Sequential()
                model.add(LSTM(64, input_shape=(x_train_reshaped.shape[1], x_train_reshaped.shape[2]), kernel_regularizer=l2(0.01)))
                # model.add(Dropout(0.5))
                model.add(Dense(128, kernel_regularizer=l2(0.01)))
                model.add(LeakyReLU(alpha=0.01))
                model.add(Dropout(0.3))  # Increased dropout rate
                model.add(Dense(1, activation='sigmoid'))

                # Batch Normalization
                model.add(BatchNormalization())

                # Compile the model
                optimizer = Adam(clipvalue=0.5)  # Adjust the learning rate as needed
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                # Callbacks
                early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
                checkpoint = ModelCheckpoint('best_weights.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=0)

                def lr_scheduler(epoch, lr):
                    if epoch % 10 == 0 and epoch > 0:
                        return lr * 0.9
                    return lr

                lr_schedule = LearningRateScheduler(lr_scheduler)

                # Train the model
                model.fit(x_train_reshaped, y_train, verbose = 0,epochs=100, batch_size=32, 
                        validation_data=(x_new_test_reshaped, y_new_test), 
                        callbacks=[early_stop, reduce_lr, checkpoint, lr_schedule])

                # Load the best weights
                model.load_weights('best_weights.h5')

                # Evaluate the model
                accuracy = model.evaluate(x_new_test_reshaped, y_new_test)[1]
                print(f"Accuracy: {accuracy * 100:.2f}%")
                # Predict probabilities for the positive class
                y_pred_prob = model.predict(x_new_test_reshaped)

                # Compute ROC curve and ROC area for each class
                fpr, tpr, thresholds = roc_curve(y_new_test, y_pred_prob)
                best_threshold = thresholds[np.argmax(tpr - fpr)]
                y_predict_datatest = np.squeeze(y_pred_prob)
                y_binary_datatest = (y_predict_datatest >= best_threshold).astype(int)

                datatest_win_correct = np.sum(y_new_test[y_new_test == 1] == (y_binary_datatest[y_new_test == 1]))
                datatest_win_wrong = np.sum(y_new_test[y_new_test == 1] != (y_binary_datatest[y_new_test == 1]))

                datatest_loss_correct = np.sum(y_new_test[y_new_test == 0] == (y_binary_datatest[y_new_test == 0]))
                datatest_loss_wrong = np.sum(y_new_test[y_new_test == 0] != (y_binary_datatest[y_new_test == 0]))

                try:
                    win_rate_test = datatest_win_correct/(datatest_win_correct + datatest_loss_wrong)
                    print("winrate :",win_rate_test , " total trade :", datatest_win_correct + datatest_loss_wrong, " for", f"{State[i]}Data/{State[i]}_{currency}{time}")
                
                    if win_rate_test >= 0.7 and (datatest_win_correct + datatest_loss_wrong) >= int(y_new_test.value_counts()[1]*0.3) :
                        print('% win in data test :', win_rate_test)
                        print("total trade test :",datatest_win_correct + datatest_loss_wrong)
                        print("win trade test :",datatest_win_correct)
                        print("loss trade test :",datatest_loss_wrong, '\n')
                        model.save(f'{path[i]}/{State[i]}_{currency}{time}.h5')
                        break
                except print(0):
                    pass