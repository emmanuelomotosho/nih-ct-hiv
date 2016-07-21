#!/usr/bin/env python3

import json
import pickle
import pprint
import re
import sqlite3
import string
import sys


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import svm

from sklearn.feature_selection import chi2, SelectKBest
from sklearn import cross_validation
from scipy import stats as ST
import matplotlib.pyplot as plt

REMOVE_PUNC = str.maketrans({key: None for key in string.punctuation})


def filter_study(ec):
    """take one study and returns a filtered version with only relevant lines included"""
    lines = []
    segments = re.split(
        r'\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|(?:[A-Z][A-Za-z]+ )+?[A-Z][A-Za-z]+: +|; +| (?=[A-Z][a-z])',
        ec, flags=re.MULTILINE)
    for i, l in enumerate(segments):
        l = l.strip()
        if l:
            l = l.translate(REMOVE_PUNC).strip()
            if l:
                lines.append(l)
    return '\n'.join(lines)


def vectorize_all(vectorizer, input_docs, fit=False):
    if fit:
        dtm = vectorizer.fit_transform(input_docs)
    else:
        dtm = vectorizer.transform(input_docs)
    return dtm


if __name__ == '__main__':

    np.set_printoptions(precision=2)

    with open(sys.argv[1]) as f:
        config = json.load(f)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    if config.get('cui_file'):
        CUI = json.load(open(config['cui_file']))
    else:
        CUI = None

    conn = sqlite3.connect(config['database'])
    c = conn.cursor()
    c.execute('SELECT studies.NCTId, studies.EligibilityCriteria, annotations.%s \
        FROM studies, annotations WHERE studies.NCTId=annotations.NCTId \
        AND annotations.%s IS NOT NULL ORDER BY studies.NCTId' % (config['annotation'], config['annotation']))

    X = []
    y = []
    study_ids = []

    for row in c.fetchall():
        text = filter_study(row[1])
        if CUI is not None:
            text += '\n' + '\n'.join(CUI[row[0]])
        # print(text)
        if text:
            yv = row[2]
            for mr in config.get('merge', []):
                if yv in mr:
                    yv = mr[0]
                    break
            X.append(text)
            y.append(yv)
            study_ids.append(row[0])
        else:
            print("[WARNING] no text returned from %s after filtering" % row[0])

    study_ids = np.array(study_ids)

    model = None
    if config.get('model'):
        with open(config['model'], 'rb') as f:
            payload = pickle.load(f)
            vectorizer = payload['vectorizer']
            model = payload['model']
        X = vectorize_all(vectorizer, X, fit=False)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = vectorize_all(vectorizer, X, fit=True)

    y = np.array(y)
    print(X.shape)

    chi2_best = SelectKBest(chi2, k=config.get('chi2_k', 250))
    X = chi2_best.fit_transform(X, y)
    print(X.shape)
    print(np.asarray(vectorizer.get_feature_names())[chi2_best.get_support()])

    stats = []
    global_stats = []
    seed = 0
    folds = 10
    print("CV folds: %s" % folds)

    label_map = config['labels']
    mean_fpr = {}
    mean_tpr = {}
    y_test_class = {}
    y_pred_class = {}
    for x in label_map:
        mean_fpr[x] = np.linspace(0, 1, 100)
        mean_tpr[x] = [0.0]
        y_test_class[x] = []
        y_pred_class[x] = []

    study_ids_test = []
    y_test_all = []
    y_pred_all = []
    y_pred_proba_all = []

    skf = cross_validation.StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=seed)
    model_cache = []
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        y_test_all.extend(y_test)
        study_ids_test.extend(list(study_ids[test]))

        if not config.get('model'):
            config_svm = config.get('svm', {})
            class_weight = config_svm.get('class_weight', None)
            if class_weight != 'balanced':
                class_weight = dict(zip(range(len(class_weight)), class_weight))
            model = svm.LinearSVC(
                C=config_svm.get('C', 1),
                class_weight=class_weight,
                random_state=seed)
            model.fit(X_train, y_train)

        y_predicted = model.predict(X_test)
        y_pred_all.extend(y_predicted)

        if len(label_map) > 2:
            avg_mode = 'macro'
        else:
            avg_mode = 'binary'

        if config.get('export'):
            model_cache.append((model, metrics.fbeta_score(y_test, y_predicted, beta=2.0, average=avg_mode)))

        y_predicted_score = model.decision_function(X_test)
        prob_min = y_predicted_score.min()
        prob_max = y_predicted_score.max()
        for x in y_predicted_score:
            if len(label_map) > 2:  # handle binary vs multiclass probability format
                p = [(i - prob_min) / (prob_max - prob_min) for i in x]
                y_pred_proba_all.append(p)
            else:
                p = (x - prob_min) / (prob_max - prob_min)
                y_pred_proba_all.append((1 - p, p))

        sd = list(metrics.precision_recall_fscore_support(y_test, y_predicted, beta=2.0, average=None))[:3]
        aucs = []
        ap_score = []
        for i, label in enumerate(label_map):
            bt = (y_test == i)
            y_test_class[label].extend(list(bt))
            if len(label_map) > 2:
                bp = y_predicted_score[:,i]
                y_pred_class[label].extend(list(bp))
            else:
                if i == 0:
                    bp = [-x for x in y_predicted_score]
                else:
                    bp = [x for x in y_predicted_score]
                y_pred_class[label].extend(bp)

            aucs.append(metrics.roc_auc_score(bt, bp))
            fpr, tpr, thresholds = metrics.roc_curve(bt, bp)
            mean_tpr[label] += np.interp(mean_fpr[label], fpr, tpr)
            mean_tpr[label][0] = 0.0

            ap_score.append(metrics.average_precision_score(bt, bp))

        sd.append(np.array(aucs))
        sd.append(np.array(ap_score))
        stats.append(sd)

        # compute micro-averaged stats
        global_sd = list(metrics.precision_recall_fscore_support(
            y_test, y_predicted, beta=2.0, average=avg_mode))[:3]
        global_stats.append(global_sd)

    y_pred_proba_all = np.array(y_pred_proba_all)

    results = []
    for i in range(len(y_test_all)):
        results.append((study_ids_test[i], y_pred_all[i], y_test_all[i], y_pred_proba_all[i]))
    results.sort(key=lambda x: (x[1], x[2]))
    for x in results:
        print("[%s] %s %s %s" % x)

    for i, label in enumerate(label_map):
        stat_mean = {}
        for j, metric in enumerate(('precision', 'recall', 'F2 score', 'ROC-AUC score', 'PR-AUC score')):
            sd = np.array([x[j][i] for x in stats])
            print("%s %s: %s" % (label, metric, sd))
            sd_mean = np.mean(sd)
            stat_mean[metric] = sd_mean
            sd_ci = np.array(ST.t.interval(0.95, len(sd) - 1, loc=sd_mean, scale=ST.sem(sd)))
            print("%s %s: %.2f %s" % (label, metric, sd_mean, sd_ci))
        print("%s count: %s" % (label, len([x for x in y_test_all if x == i])))

        plt.figure(1)
        mean_tpr[label] /= folds
        mean_tpr[label][-1] = 1.0
        plt.plot(mean_fpr[label], mean_tpr[label],
                 label="%s (mean AUC = %0.2f)" % (label, stat_mean['ROC-AUC score']), lw=2)
        plt.figure(2)
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_class[label], y_pred_class[label]
        )
        plt.plot(recall, precision,
                 label="%s (PR-AUC = %0.2f)" % (label, stat_mean['PR-AUC score']), lw=2)

    stat_mean = {}
    for i, metric in enumerate(('precision', 'recall', 'F2 score')):
        sd = np.array([x[i] for x in global_stats])
        print("All %s: %s" % (metric, sd))
        sd_mean = np.mean(sd)
        stat_mean[metric] = sd_mean
        sd_ci = np.array(ST.t.interval(0.95, len(sd) - 1, loc=sd_mean, scale=ST.sem(sd)))
        print("All %s: %.2f %s" % (metric, sd_mean, sd_ci))


    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test_all, y_pred_all))

    if config.get('export'):
        model_cache.sort(key=lambda x: x[1], reverse=True)  # sort by descending F-score
        payload = {
            'vectorizer': vectorizer,
            'model': model_cache[0][0]
        }
        with open(config['export'], 'wb') as f:
            pickle.dump(payload, f)
        print("Exported vectorizer and model to " + config['export'])

    plt.figure(1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    ax = plt.gca()
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    plt.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(config["title"])
    plt.legend(loc="lower right")

    plt.figure(2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(config["title"])
    plt.legend(loc="lower left")

    plt.show()
