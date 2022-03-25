#!/usr/bin/env python
# coding: utf-8

# # Baseline Experiments
# models: SVM, RandomForest, Logistic Regression, MLP, Simple 2 layer NN

# In[59]:


import numpy as np
import pandas as pd
import tensorflow as tf
from math import floor, ceil
import sklearn as sk
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, LSTM, SimpleRNN, GRU, Masking, Bidirectional, Dropout, TimeDistributed, Flatten
from keras import Input

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os.path
import matplotlib.pyplot as pyplot
import tensorflow_hub as hub

import time


def evaluate(model, x_test, y_test, week_type, feature_type, course, model_name=None, model_params=None, y_pred=None):
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

def bidirectional_lstm_64(x_train, y_train, x_test, y_test, x_val, y_val, x_meta_train, x_meta_test, x_meta_val, week_type, feature_types, course, num_epochs=10, experiment=''):
    dim_max = int(np.max(x_train[:, :]) + 1)
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    look_back = 3
    
    # LSTM
    # define model
    bilstm = Bidirectional(LSTM(64))
    firstdense = Dense(73, activation='sigmoid')
    metadense1 = Dense(64, activation='sigmoid', input_shape=(n_weeks, x_meta_train.shape[2]))
    metadense2 = Dense(1, activation='sigmoid')
    pooling = Flatten()

    # define model
    lstm = Sequential([
        bilstm,
        firstdense,
        metadense1,
        metadense2,
        pooling
    ])

    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    meta_model = Sequential([
        metadense1,
        metadense2,
        pooling
    ])    
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-128-'+ experiment + current_timestamp
    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    meta_model.fit(x_meta_train, y_train, validation_data=(x_meta_val, y_val), epochs=num_epochs, batch_size=8, verbose=1)
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, y_pred=y_pred, model_name="TF-LSTM-bi-64" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, y_pred=y_val_pred, model_name="TF-LSTM-bi-64" , model_params=model_params)
    lstm.save(checkpoint_filepath + experiment + '_final_e')
    return history, scores, val_scores, lstm

def bidirectional_lstm_32_32(x_train, y_train, x_test, y_test, x_val, y_val, x_meta_train, x_meta_test, x_meta_val, week_type, feature_types, course, num_epochs=10, experiment=''):
    dim_max = int(np.max(x_train[:, :]) + 1)
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    look_back = 3
    
    # LSTM
    # define model
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
#     lstm.add(Embedding(input_dim=dim_max, output_dim=64))
    # Add a LSTM layer with 128 internal units.
    # lstm.add(Masking(mask_value=-1., input_shape=(n_dims, n_weeks, n_features)))
    # lstm.add(LSTM(128, input_shape=(n_weeks, look_back), return_sequences=True))
    bilstm = Bidirectional(LSTM(32, return_sequences=True))
    bilstm2 = Bidirectional(LSTM(32))
    firstdense = Dense(73, activation='sigmoid')
    metadense1 = Dense(64, activation='sigmoid', input_shape=(n_weeks, x_meta_train.shape[2]))
    metadense2 = Dense(1, activation='sigmoid')
    pooling = Flatten()

    # define model
    lstm = Sequential([
        bilstm,
        bilstm2,
        firstdense,
        metadense1,
        metadense2,
        pooling
    ])

    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    meta_model = Sequential([
        metadense1,
        metadense2,
        pooling
    ])    
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-32-32-'+ experiment + current_timestamp
    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    meta_model.fit(x_meta_train, y_train, validation_data=(x_meta_val, y_val), epochs=num_epochs, batch_size=8, verbose=1)
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, y_pred=y_pred, model_name="TF-LSTM-bi-32-32" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, y_pred=y_val_pred, model_name="TF-LSTM-bi-32-32" , model_params=model_params)
    lstm.save(checkpoint_filepath + experiment + '_final_e')
    return history, scores, val_scores, lstm

def bidirectional_lstm_32(x_train, y_train, x_test, y_test, x_val, y_val, x_meta_train, x_meta_test, x_meta_val, week_type, feature_types, course, num_epochs=10, experiment=''):
    dim_max = int(np.max(x_train[:, :]) + 1)
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    look_back = 3
    
    # LSTM
    # define model
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
#     lstm.add(Embedding(input_dim=dim_max, output_dim=64))
    # Add a LSTM layer with 128 internal units.
    # lstm.add(Masking(mask_value=-1., input_shape=(n_dims, n_weeks, n_features)))
    # lstm.add(LSTM(128, input_shape=(n_weeks, look_back), return_sequences=True))
    bilstm = Bidirectional(LSTM(32))
    firstdense = Dense(73, activation='sigmoid')
    metadense1 = Dense(64, activation='sigmoid', input_shape=(n_weeks, x_meta_train.shape[2]))
    metadense2 = Dense(1, activation='sigmoid')
    pooling = Flatten()

    # define model
    lstm = Sequential([
        bilstm,
        firstdense,
        metadense1,
        metadense2,
        pooling
    ])

    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    meta_model = Sequential([
        metadense1,
        metadense2,
        pooling
    ])    
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-32-'+ experiment + current_timestamp
    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # fit the model
    meta_model.fit(x_meta_train, y_train, validation_data=(x_meta_val, y_val), epochs=num_epochs, batch_size=8, verbose=1)
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, y_pred=y_pred, model_name="TF-LSTM-bi-32" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, y_pred=y_val_pred, model_name="TF-LSTM-bi-32" , model_params=model_params)
    lstm.save(checkpoint_filepath + experiment + '_final_e')
    return history, scores, val_scores, lstm


def bidirectional_lstm_128(x_train, y_train, x_test, y_test, x_val, y_val, x_meta_train, x_meta_test, x_meta_val, week_type, feature_types, course, num_epochs=10, experiment=''):
    dim_max = int(np.max(x_train[:, :]) + 1)
    n_dims = x_train.shape[0]
    n_weeks = x_train.shape[1]
    n_features = x_train.shape[2]
    look_back = 3
    
    # LSTM
    bilstm = Bidirectional(LSTM(128))
    firstdense = Dense(73, activation='sigmoid')
    metadense1 = Dense(64, activation='sigmoid', input_shape=(n_weeks, x_meta_train.shape[2]))
    metadense2 = Dense(1, activation='sigmoid')
    pooling = Flatten()

    # define model
    lstm = Sequential([
        bilstm,
        firstdense,
        metadense1,
        metadense2,
        pooling
    ])

    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    meta_model = Sequential([
        metadense1,
        metadense2,
        pooling
    ])    
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint_filepath = 'checkpoints/lstm-bi-128-'+ experiment + current_timestamp
    os.mkdir(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])
    meta_model.fit(x_meta_train, y_train, validation_data=(x_meta_val, y_val), epochs=num_epochs, batch_size=8, verbose=1)
    lstm = tf.keras.models.load_model(checkpoint_filepath)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 64, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, y_pred=y_pred, model_name="TF-LSTM-bi-128" , model_params=model_params)

    y_val_pred = lstm.predict(x_val)
    y_val_pred = [1 if y[0] >= 0.5 else 0 for y in y_val_pred]
    val_scores = evaluate(None, x_val, y_val, week_type, feature_types, course, y_pred=y_val_pred, model_name="TF-LSTM-bi-128" , model_params=model_params)
    lstm.save(checkpoint_filepath + experiment + '_final_e')
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

def load_meta_model(course_list):
    from gensim.models import FastText
    model = FastText(vector_size=60, window=3, min_count=1, sentences=course_list, epochs=15)
    return model

def meta_feature(course_name, meta_model):
    encoded_course = meta_model.wv[[course_name]]
    print(encoded_course.shape)
    return np.array(encoded_course)

def get_meta_info(course):
    adding_info_about_course = pd.get_dummies(metadata[['weeks', 'language', 'level', 'topic']])[metadata['course_id'] == course.replace('_', '-')]
    print(adding_info_about_course.values.shape)
    return adding_info_about_course.values

def predict_on_transfer(best_model, exp_type, percentile, name):
    week_type = 'eq_week'
    feature_types = [ "lalle_conati", "boroujeni_et_al", "chen_cui", "marras_et_al"]
    courses = ['dsp_002', 'villesafricaines_001', 'structures_001', 'progfun_002', 'geomatique_003', 'venture_001']
    metadata = pd.read_csv('new_data/metadata_augmented.csv')
    
    path = '../data/result/easy-fail/'
    experiment_scores = pd.DataFrame(columns=[ 'experiment_type','acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])    
    counter = 0
    for course in courses:
        print(course)
        feature_type = "boroujeni_et_al"
        filepath = path + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
        labels = pd.read_csv(filepath)['label-pass-fail']
        total_weeks = list(metadata[metadata['course_id'] == course.replace('_', '-')]['weeks'])[0]
        
        num_weeks = int(np.round(total_weeks * percentile))
        feature_list=[]
        for feature_type in feature_types:
            filepath = path + week_type + '-' + feature_type + '-' + course
            feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
            feature_current = np.nan_to_num(feature_current, nan=-1)
            feature_current = feature_current[:, :num_weeks, :]
            feature_current = np.pad(feature_current, pad_width=((0, 0), (0, int(15*percentile)-feature_current.shape[1]), (0, 0)), mode='constant', constant_values=0)
            feature_norm = feature_current.reshape(labels.shape[0], -1)
            feature_norm = normalize(feature_norm)
            feature_current = feature_norm.reshape(feature_current.shape)
            feature_list.append(feature_current)
        course_features = np.concatenate(feature_list, axis=2)

        description = list(metadata[metadata['course_id'] == course.replace('_', '-')]['short_description'])[0]
        adding_meta_each_week = np.repeat(meta_feature(course + description, meta_model)[:, np.newaxis, :], course_features.shape[1], axis=1).reshape((course_features.shape[1], 60))
        adding_meta_each_week = np.repeat(adding_meta_each_week[np.newaxis, :, :], course_features.shape[0], axis=0)
        print(adding_meta_each_week.shape)
        
        meta_features = np.concatenate([adding_meta_each_week, course_features], axis=2)
        
        adding_info_about_course = np.repeat(get_meta_info(course)[:, np.newaxis, :], meta_features.shape[1], axis=1).reshape((meta_features.shape[1], 13))
        print(adding_info_about_course.shape)
        adding_info_about_course = np.repeat(adding_info_about_course[np.newaxis, :, :], meta_features.shape[0], axis=0)
        print(adding_info_about_course.shape)
        
        meta_features = np.concatenate([adding_info_about_course, meta_features], axis=2)
        print(meta_features.shape)
        features = course_features
        scores1 = evaluate(best_model, features, labels, week_type, feature_types, course, y_pred=None, model_name=name)
        scores1['experiment_type'] = exp_type
        experiment_scores.loc[counter] = scores1

        counter += 1
    return experiment_scores    


rnn_mode = True
path = '../data/result/easy-fail/'
week_type = 'eq_week'
feature_types = [ "lalle_conati", "boroujeni_et_al", "chen_cui", "marras_et_al"]
courses = ['analysenumerique_001', 'analysenumerique_002', 'analysenumerique_003', 'cpp_fr_001', 'dsp_001', 'dsp_004', 'dsp_005', 'dsp_006','hwts_001', 'hwts_002','initprogcpp_001', 'microcontroleurs_003', 'microcontroleurs_004', 'microcontroleurs_005', 'microcontroleurs_006', 'progfun_003', 'structures_002', 'structures_003', 'villesafricaines_002', 'villesafricaines_003']
# courses = ['analysenumerique_001', 'analysenumerique_002', 'analysenumerique_003', 'cpp_fr_001', 'dsp_001', 'dsp_004', 'dsp_005', 'dsp_006', 'initprogcpp_001', 'initprogjava_001', 'microcontroleurs_003', 'microcontroleurs_004', 'microcontroleurs_005', 'microcontroleurs_006', 'hwts_001', 'villesafricaines_001', 'villesafricaines_002', 'villesafricaines_003']
# courses = ['progfun_002', 'progfun_003', 'progfun_004']
    # # of layers: [(64, 32), (128, 64, 32), (128, 64), (128, 128, 64, 32), (256, 128, 64)]
# rnn_models = [bidirectional_lstm_128_128_64_32, bidirectional_lstm_256_128_64, bidirectional_lstm_128, bidirectional_lstm_32_32]
rnn_models = [bidirectional_lstm_32, bidirectional_lstm_32_32, bidirectional_lstm_64, bidirectional_lstm_128]

experiment = 'dense_layers_fasttext_desc_add_info'
exp_type = experiment
save_name = 'run_history/' + experiment + '_20' + week_type + '_bilstm'
save_stats = save_name + ".csv"
save_val_stats = save_name + "val.csv"

train_size = 0.8
test_size = 0.1
val_size = 0.1

counter = 0
experiment_scores = pd.DataFrame(columns=['acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])
val_exp_scores = pd.DataFrame(columns=['acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])
transfer_experiment_scores = pd.DataFrame(columns=[ 'experiment_type','experiment','acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])

metadata = pd.read_csv('new_data/metadata_augmented.csv')
meta_model = load_meta_model(list(metadata['title']))
early_predict = [0.4, 0.6]
epochs = 100

for percentile in early_predict:
    x_train = []
    x_test = []
    x_val = []
    x_meta_train = []
    x_meta_test = []
    x_meta_val = []
    y_train = []
    y_test = []
    y_val = []
    for course in courses:
        feature_list = []
        feature_type = "boroujeni_et_al"
        filepath = path + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
        total_weeks = list(metadata[metadata['course_id'] == course.replace('_', '-')]['weeks'])[0]
        num_weeks = int(np.round(total_weeks * percentile))
        labels = pd.read_csv(filepath)['label-pass-fail']
        for feature_type in feature_types:
            filepath = path + week_type + '-' + feature_type + '-' + course
            print(filepath)

            feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
            feature_current = np.nan_to_num(feature_current, nan=-1)
            feature_current = feature_current[:, :num_weeks, :]
            feature_current = np.pad(feature_current, pad_width=((0, 0), (0, int(15*percentile)-feature_current.shape[1]), (0, 0)), mode='constant', constant_values=0)
            if rnn_mode:
                feature_norm = feature_current.reshape(labels.shape[0], -1)
                feature_norm = normalize(feature_norm)
                feature_current = feature_norm.reshape(feature_current.shape)
            else:
                feature_current = feature_current.reshape(labels.shape[0], -1)
                feature_current = normalize(feature_current)
            
            feature_list.append(feature_current)

        course_features = np.concatenate(feature_list, axis=2)
        print(course_features.shape)
        
        # add meta_features
        description = list(metadata[metadata['course_id'] == course.replace('_', '-')]['short_description'])[0]
        adding_meta_each_week = np.repeat(meta_feature(course + description, meta_model)[:, np.newaxis, :], course_features.shape[1], axis=1).reshape((course_features.shape[1], 60))
        adding_meta_each_week = np.repeat(adding_meta_each_week[np.newaxis, :, :], course_features.shape[0], axis=0)
        print(adding_meta_each_week.shape)
        
        course_features = np.concatenate([adding_meta_each_week, course_features], axis=2)
        print(course_features.shape)
        
        adding_info_about_course = np.repeat(get_meta_info(course)[:, np.newaxis, :], course_features.shape[1], axis=1).reshape((course_features.shape[1], 13))
        print(adding_info_about_course.shape)
        adding_info_about_course = np.repeat(adding_info_about_course[np.newaxis, :, :], course_features.shape[0], axis=0)
        print(adding_info_about_course.shape)
        
        course_features = np.concatenate([adding_info_about_course, course_features], axis=2)
        print(course_features.shape)

        print(course, total_weeks, num_weeks, course_features.shape, percentile)
        x_train_c, x_test_v_c, y_train_c, y_test_v_c = train_test_split(course_features, labels.values, test_size=test_size + val_size, random_state=0, stratify=labels)
        x_test_c, x_val_c, y_test_c, y_val_c = train_test_split(x_test_v_c, y_test_v_c, test_size=val_size, random_state=0, stratify=y_test_v_c)
        x_train.append(x_train_c[:, :, 73:])
        x_test.append(x_test_c[:, :, 73:])
        x_val.append(x_val_c[:, :, 73:])
        y_train.append(y_train_c)
        y_test.append(y_test_c)
        y_val.append(y_val_c)
        
        x_meta_train.append(x_train_c[:, :, :73])
        x_meta_test.append(x_test_c[:, :, :73])
        x_meta_val.append(x_val_c[:, :, :73])
        print(x_train_c[:, :, 73:].shape, x_train_c[:, :, :73].shape)

        print('course: ', course)
        print('week_type: ', week_type)
        print('feature_type: ', feature_types)


    # ### train-test split
    # In[26]:

    x_train, x_test, x_val = np.concatenate(x_train), np.concatenate(x_test), np.concatenate(x_val)
    x_meta_train, x_meta_test, x_meta_val = np.concatenate(x_meta_train), np.concatenate(x_meta_test), np.concatenate(x_meta_val)
    y_train, y_test, y_val = np.concatenate(y_train), np.concatenate(y_test), np.concatenate(y_val)

    best_models = []
    for model in rnn_models:
        print(model.__name__)
        current_timestamp = str(time.time())[:-2]
        history, scores, val_scores, best_model = model(x_train, y_train, x_test, y_test, x_val, y_val, x_meta_train, x_meta_test, x_meta_val, week_type, feature_types, course, num_epochs=epochs, experiment=experiment)
        experiment_scores.loc[counter] = scores
        val_exp_scores.loc[counter] = val_scores
        counter += 1

        run_name = exp_type + model.__name__  + "_ep" + str(percentile) + "_" + current_timestamp

        plot_history(history, 'run_history/' + run_name, counter)
        numpy_loss_history = np.array(history.history['loss'])
        np.savetxt('run_history/' + run_name + "_loss_history.txt", numpy_loss_history, delimiter=",")
        
        experiment_scores.to_csv(save_stats)
        val_exp_scores.to_csv(save_val_stats)
        
        print("Running transfer experiments.")
        
        # run transfer experiments
        transfer_experiment_scores = pd.concat([transfer_experiment_scores, predict_on_transfer(best_model, exp_type, percentile, run_name)])
        transfer_experiment_scores.to_csv(save_name + "_transfer_results.csv")

experiment_scores.to_csv(save_stats)
val_exp_scores.to_csv(save_val_stats)
print(experiment_scores)

transfer_experiment_scores.to_csv(save_name + "_transfer_results.csv")



