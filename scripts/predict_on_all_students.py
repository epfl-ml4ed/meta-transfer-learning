#!/usr/bin/env python
# coding: utf-8

# Experiment: Take a model and do inference on all students (not just the hard students)
# RQs: 2, Figure 4
# Code: All Students BO, BSM, BTM
# Author: vinitra



import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import normalize
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

def load_meta_model(course_list, vec_size):
    from gensim.models import FastText
    model = FastText(vector_size=vec_size, window=3, min_count=1, sentences=course_list, epochs=15)
    return model

def meta_feature(info, meta_model):
    encoded_info = meta_model.wv[[info]]
    print(encoded_info.shape)
    return np.array(encoded_info)

def get_meta_info(course):
    adding_info_about_course = pd.get_dummies(metadata[['weeks', 'language', 'level']])[metadata['course_id'] == course.replace('_', '-')]
    print(adding_info_about_course.values.shape)
    return adding_info_about_course.values

def predict_on_transfer(best_model, exp_type, percentile, name):
    week_type = 'eq_week'
    feature_types = [ "lalle_conati", "boroujeni_et_al", "chen_cui", "marras_et_al"]
    courses = ['dsp_002', 'villesafricaines_001', 'structures_001', 'progfun_002', 'geomatique_003', 'venture_001']
    metadata = pd.read_csv('new_data/metadata_augmented.csv')
    
    path = '../data/result/'
    experiment_scores = pd.DataFrame(columns=[ 'experiment_type','acc', 'bac','prec','rec','f1', 'auc', 'feature_type', 'week_type', 'course', 'model_name','data_balance', 'timestamp', 'percentile'])    
    counter = 0
    
    x_train = []
    x_test = []
    x_val = []
    y_train = []
    y_test = []
    y_val = []
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
        scores1 = evaluate(best_model, features, labels, week_type, feature_types, course, y_pred=None, model_name=name)
        scores1['experiment_type'] = exp_type
        experiment_scores.loc[counter] = scores1

        counter += 1
    return experiment_scores   

rnn_mode = True
path = '../data/result/easy-fail/'
week_type = 'eq_week'
feature_types = [ "lalle_conati", "boroujeni_et_al", "chen_cui", "marras_et_al"]

rnn_models = ['lstm-bi-32-32-1641514744.58229']
experiment = 'predict_all_students_baseline'
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

percentile = 0.6
best_models = []
for model in rnn_models:
    reconstructed_model = tf.keras.models.load_model("checkpoints/" + model)

    print("Running transfer experiments.")

    # run transfer experiments
    transfer_experiment_scores = pd.concat([transfer_experiment_scores, predict_on_transfer(reconstructed_model, exp_type, percentile)])
    transfer_experiment_scores.to_csv(save_name + "_transfer_results.csv")

experiment_scores.to_csv(save_stats)
val_exp_scores.to_csv(save_val_stats)
print(experiment_scores)

transfer_experiment_scores.to_csv(save_name + "_transfer_results.csv")

