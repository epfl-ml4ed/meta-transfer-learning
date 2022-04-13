#!/usr/bin/env python
# coding: utf-8

# Experiment: Train on 20 courses, predict on one course
# RQs: 1, 2
# Code: BO N-1 Diff
# Author: vinitra


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from rnn_models import *

import time

def predict_on_transfer(best_model, exp_type, percentile, name):
    week_type = 'eq_week'
    feature_types = [ "lalle_conati", "boroujeni_et_al", "chen_cui", "marras_et_al"]
    courses = ['dsp_002', 'villesafricaines_001', 'structures_001', 'progfun_002', 'geomatique_003', 'venture_001']
    metadata = pd.read_csv('metadata.csv')
    
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

        features = course_features

        scores1 = evaluate(best_model, features, labels, week_type, feature_types, course, percentile, current_timestamp, y_pred=None, model_name=name)
        scores1['experiment_type'] = exp_type
        experiment_scores.loc[counter] = scores1

        counter += 1
    return experiment_scores    


rnn_mode = True
path = '../data/result/easy-fail/'
exp_type = 'baseline'
week_type = 'eq_week'
feature_types = [ "lalle_conati", "boroujeni_et_al", "chen_cui", "marras_et_al"]
courses = ['analysenumerique_001', 'analysenumerique_002', 'analysenumerique_003', 'cpp_fr_001', 'dsp_001', 'dsp_004', 'dsp_005', 'dsp_006','hwts_001', 'hwts_002','initprogcpp_001', 'microcontroleurs_003', 'microcontroleurs_004', 'microcontroleurs_005', 'microcontroleurs_006', 'progfun_003', 'structures_002', 'structures_003', 'villesafricaines_002', 'villesafricaines_003']
rnn_models = [bidirectional_lstm_32, bidirectional_lstm_32_32, bidirectional_lstm_64, bidirectional_lstm_128]

save_name = 'run_history/baseline' + '_20_' + week_type + '_bilstm'
save_stats = save_name + ".csv"
save_val_stats = save_name + "val.csv"

train_size = 0.8
test_size = 0.1
val_size = 0.1

counter = 0
experiment_scores = pd.DataFrame(columns=['acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])
val_exp_scores = pd.DataFrame(columns=['acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])
transfer_experiment_scores = pd.DataFrame(columns=[ 'experiment_type','experiment','acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])

metadata = pd.read_csv('metadata.csv')
early_predict = [0.4, 0.6]
epochs = 100

for percentile in early_predict:
    x_train = []
    x_test = []
    x_val = []
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

        print(course, total_weeks, num_weeks, course_features.shape, percentile)
        x_train_c, x_test_v_c, y_train_c, y_test_v_c = train_test_split(course_features, labels.values, test_size=test_size + val_size, random_state=0, stratify=labels)
        x_test_c, x_val_c, y_test_c, y_val_c = train_test_split(x_test_v_c, y_test_v_c, test_size=val_size, random_state=0, stratify=y_test_v_c)
        x_train.append(x_train_c)
        x_test.append(x_test_c)
        x_val.append(x_val_c)
        y_train.append(y_train_c)
        y_test.append(y_test_c)
        y_val.append(y_val_c)

        print('course: ', course)
        print('week_type: ', week_type)
        print('feature_type: ', feature_types)


    # ### train-test split

    x_train, x_test, x_val = np.concatenate(x_train), np.concatenate(x_test), np.concatenate(x_val)
    y_train, y_test, y_val = np.concatenate(y_train), np.concatenate(y_test), np.concatenate(y_val)

    best_models = []
    for model in rnn_models:
        print(model.__name__)
        current_timestamp = str(time.time())[:-2]
        history, scores, val_scores, best_model = model(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course, current_timestamp, num_epochs=epochs)
        experiment_scores.loc[counter] = scores
        val_exp_scores.loc[counter] = val_scores
        counter += 1

        run_name = 'baseline_best_' + model.__name__  + "_ep" + str(percentile) + "_" + current_timestamp

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

