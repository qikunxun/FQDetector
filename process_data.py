import json

dataset = 'politifact'

label_map = {'true': 0, 'false': 1}

with open('local_dataset/{}/dev.json'.format(dataset)) as fd:
    data = json.load(fd)

fw = open('local_dataset/{}/processed/dev.txt'.format(dataset), mode='w')
for item in data:
    claim = data[item]['claim_text']
    label = data[item]['cred_label']
    evidences = data[item]['evidences']
    evidence_text = ''
    for j, evidence in enumerate(evidences):
        evidence_text += '[{}] '.format(j + 1) + evidence[1]
    fw.write('{}\t{}\t{}\n'.format(claim, evidence_text, label_map[label]))
fw.close()

for i in range(5):
    with open('local_dataset/{}/train_{}.json'.format(dataset, i)) as fd:
        data = json.load(fd)
    fw = open('local_dataset/{}/processed/train_{}.txt'.format(dataset, i), mode='w')
    for item in data:
        claim = data[item]['claim_text']
        label = data[item]['cred_label']
        evidences = data[item]['evidences']
        evidence_text = ''
        for j, evidence in enumerate(evidences):
            evidence_text += '[{}] '.format(j + 1) + evidence[1]
        fw.write('{}\t{}\t{}\n'.format(claim, evidence_text, label_map[label]))
    fw.close()

    with open('local_dataset/{}/test_{}.json'.format(dataset, i)) as fd:
        data = json.load(fd)
    fw = open('local_dataset/{}/processed/test_{}.txt'.format(dataset, i), mode='w')
    for item in data:
        claim = data[item]['claim_text']
        label = data[item]['cred_label']
        evidences = data[item]['evidences']
        evidence_text = ''
        for j, evidence in enumerate(evidences):
            evidence_text += '[{}] '.format(j + 1) + evidence[1]
        fw.write('{}\t{}\t{}\n'.format(claim, evidence_text, label_map[label]))
    fw.close()