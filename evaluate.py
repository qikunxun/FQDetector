import json
import re
from sklearn import metrics

dataset = 'snopes'
fold = '0'
thr = 0.9995
pattern = r'"is it a fake news": "(.*?)"'
y_true = []
y_pred = []
labels = {}
with open('local_dataset/{}/processed/test_{}.txt'.format(dataset, fold), mode='r') as fd:
    for line in fd:
        if not line: continue
        items = line.strip().split('\t')
        query, page, label = items
        flag = 0
        if label == '1': flag =1
        labels[query] = flag
preds = {}
results = {}
count = 0
count_all = 0
with open('local_dataset/{}/preds/test_pred_{}.txt'.format(dataset, fold), mode='r') as fd:
    for line in fd:
        if not line: continue
        items = line.strip().split('\t')
        if len(items) != 2: continue
        query = items[0]
        if query not in labels: continue
        label = labels[query]
        probs = items[-1][1: -1].strip().split(' ')
        probs_ = []
        for prob in probs:
            if prob != '': probs_.append(float(prob))
        prediction = 0
        if probs_[1] > thr:
            prediction = 1
            preds[query] = prediction
            count += 1

with open('local_dataset/{}/preds/result.test_{}'.format(dataset, fold)) as fd:
    for line in fd:
        if not line: continue
        items = line.strip().split('\t')
        if len(items) == 2:
            query, output = items
            if query in preds: continue
            if not '{' in output:
                continue
            if query not in labels: continue
            result = re.findall(pattern, output)
            pred = 0
            if len(result) > 0 and (result[0] == 'yes'):
                pred = 1
            preds[query] = pred
            results[query] = output

for query in labels:
    y_true.append(labels[query])
    if query in preds:
        y_pred.append(preds[query])
    else:
        y_pred.append(0)
    count_all += 1

print(len(y_pred), len(y_true))
print(sum(y_pred) / len(y_pred))
print(metrics.classification_report(y_true, y_pred))

print('acc', metrics.accuracy_score(y_true, y_pred))
print('Macro precision', metrics.precision_score(y_true, y_pred, average='macro'))
print('Macro recall', metrics.recall_score(y_true, y_pred, average='macro'))
print('Macro f1', metrics.f1_score(y_true, y_pred, average='macro'))

print('Micro precision', metrics.precision_score(y_true, y_pred, average='micro'))
print('Micro recall', metrics.recall_score(y_true, y_pred, average='micro'))
print('Micro f1', metrics.f1_score(y_true, y_pred, average='micro'))

print('True precision', metrics.precision_score(y_true, y_pred, pos_label=0))
print('True recall', metrics.recall_score(y_true, y_pred, pos_label=0))
print('True f1', metrics.f1_score(y_true, y_pred, pos_label=0))

print('False precision', metrics.precision_score(y_true, y_pred, pos_label=1))
print('False recall', metrics.recall_score(y_true, y_pred, pos_label=1))
print('False f1', metrics.f1_score(y_true, y_pred, pos_label=1))

print(count, count_all, count / count_all, 1 - count / count_all)
