import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as sk
from keras.models import Sequential
from keras.layers import Dense, Masking, LSTM, Bidirectional

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
import os.path
import matplotlib.pyplot as pyplot

def evaluate(model, x_test, y_test, week_type, feature_type, course, percentile=0.4, current_timestamp=0, model_name=None, y_pred=None):
    scores={}
    if y_pred is None:
        y_pred = model.predict(x_test)
        y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]

    scores['acc'] = accuracy_score(y_test, y_pred)
    scores['bac'] = balanced_accuracy_score(y_test, y_pred)
    scores['prec'] = precision_score(y_test, y_pred)
    scores['rec'] = recall_score(y_test, y_pred)
    scores['f1'] = f1_score(y_test, y_pred)
    scores['auc'] = roc_auc_score(y_test, y_pred)
    scores['feature_type'] = feature_type
    scores['week_type'] = week_type
    scores['course'] = course

    if model_name == None:
        scores['model_name'] = type(model).__name__
    else:
        scores['model_name'] = model_name

    scores['timestamp'] = current_timestamp
    scores['percentile'] = percentile

    scores['data_balance'] = sum(y_test)/len(y_test)

    return scores

def bidirectional_lstm_64(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, num_epochs=10):
    dim_max = int(np.max(x_train[:, :]) + 1)
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    look_back = 3
    
    # LSTM
    # define model
    lstm = Sequential()
    lstm.add(Masking(mask_value=-1., input_shape=(n_dims, n_weeks, n_features)))
    lstm.add(Bidirectional(LSTM(64)))
    lstm.add(Dense(1, activation='sigmoid'))
    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-64-' + current_timestamp

    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_pred, model_name="TF-LSTM-bi-64" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_val_pred, model_name="TF-LSTM-bi-64" , model_params=model_params)
    lstm.save(checkpoint_filepath + '_final_e')
    return history, scores, val_scores, lstm

def bidirectional_lstm_32_32(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, num_epochs=10):
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    
    # LSTM
    # define model
    lstm = Sequential()
    lstm.add(Masking(mask_value=-1., input_shape=(n_dims, n_weeks, n_features)))
    lstm.add(Bidirectional(LSTM(32, return_sequences=True)))
    lstm.add(Bidirectional(LSTM(32)))
    lstm.add(Dense(1, activation='sigmoid'))
    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-32-32-' + current_timestamp

    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_pred, model_name="TF-LSTM-bi-32-32" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_val_pred, model_name="TF-LSTM-bi-32-32" , model_params=model_params)
    lstm.save(checkpoint_filepath + '_final_e')
    return history, scores, val_scores, lstm

def bidirectional_lstm_32(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, num_epochs=10):
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    
    # LSTM
    # define model
    lstm = Sequential()
    lstm.add(Masking(mask_value=-1., input_shape=(n_dims, n_weeks, n_features)))
    lstm.add(Bidirectional(LSTM(32)))
    lstm.add(Dense(1, activation='sigmoid'))
    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-32-' + current_timestamp

    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_pred, model_name="TF-LSTM-bi-32" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_val_pred, model_name="TF-LSTM-bi-32" , model_params=model_params)
    lstm.save(checkpoint_filepath + '_final_e')
    return history, scores, val_scores, lstm


def bidirectional_lstm_128(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, num_epochs=10):
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    
    # LSTM
    # define model
    lstm = Sequential()
    lstm.add(Masking(mask_value=-1., input_shape=(n_dims, n_weeks, n_features)))
    lstm.add(Bidirectional(LSTM(128)))
    lstm.add(Dense(1, activation='sigmoid'))
    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    
    checkpoint_filepath = 'checkpoints/lstm-bi-128-' + current_timestamp
    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_pred, model_name="TF-LSTM-bi-128" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, percentile, current_timestamp, y_pred=y_val_pred, model_name="TF-LSTM-bi-128" , model_params=model_params)
    lstm.save(checkpoint_filepath + '_final_e')
    return history, scores, val_scores, lstm

def plot_history(history, file_name, counter):
    # plot loss during training
    pyplot.figure(counter*2)
    pyplot.title('Loss ' + file_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig(file_name + "_loss.png")
    # plot accuracy during training
    pyplot.figure(counter*2+1)
    pyplot.title('Accuracy ' + file_name)
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.savefig(file_name + "_acc.png")