import numpy as np
import faiss
import json

top_K = 20
def process_evidence(evidence):
    texts = evidence.split('||')
    text_new = []
    for text in texts:
        text_new.append('{}'.format(len(text_new) + 1) + text[:200])
    return ''.join(text_new)

dataset = 'politifact'
fold = '0'

outputs = {}
with open('local_dataset/{}/preds/result.train_{}'.format(dataset, fold), mode='r') as fd:
    for line in fd:
        if not line: continue
        items = line.strip().split('\t')
        if len(items) != 2: continue
        query, evi = items
        if not evi.startswith('{'): continue
        evi = evi.replace(',}', '}')
        if not evi.endswith('}'): evi = evi[:evi.index('}') + 1]
        outputs[query] = json.loads(evi.replace(',}', '}'))

train_data = []
queries = {}
filtered = []
with open('local_dataset/{}/processed/train_{}.txt'.format(dataset, fold)) as fd:
    for i, line in enumerate(fd):
        if not line: continue
        items = line.strip().split('\t')
        query = items[0]
        filtered.append(i)
        queries[query] = len(train_data)
        train_data.append(line.strip())

print(len(train_data))
test_data = []
test_index = []
with open('local_dataset/{}/processed/test_{}.txt'.format(dataset, fold)) as fd:
    for line in fd:
        if not line: continue
        items = line.strip().split('\t')
        query = items[0]
        test_data.append(line.strip())
        # test_index.append(queries[query])

train_vec = np.load('local_dataset/{}/preds/train.vec.{}.npy'.format(dataset, fold))
train_vec = train_vec[filtered]
test_vec = np.load('local_dataset/{}/preds/test.vec.{}.npy'.format(dataset, fold))

# test_index = np.array(test_index)
# test_vec = train_vec[test_index]

print(train_vec.shape, test_vec.shape)

d = train_vec.shape[-1]

index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(train_vec)
print(index.ntotal)

D, I = index.search(test_vec, top_K)

assert len(train_data) == train_vec.shape[0]
assert len(test_data) == test_vec.shape[0]


porn_prompt = """
You are a fake news detection expert. Due to the serious impact of fake news on user experience, you are now required to judge whether the news is a rumor based on the input news title, its search results, and your own knowledge. The criteria are as follows:
1. If all search results support the news or are irrelevant to the news, it is judged as not fake.
2. If there is any search result that opposes/denies/debunks this news, it is judged as fake. 
***Please note, it is crucial to ensure accuracy as this concerns the user experience.***
Output format：{"cause"："..."， "is it a fake news": "yes/no"}
Here are some positive instances:
[EXAMPLE-POS]
Here are some negative instances:
[EXAMPLE-NEG]
For the news：[QUERY]，its search results are :[PAGE]。
Based on these contents，Do you think the news: [QUERY] is it a fake news?
# """

example_prompt = """
（Input news: [QUERY]，its search results are:[PAGE]，output：{"cause"："[EVIDENCE]"， "is it a fake news": "[LABEL]"}）
"""
label_map = {'0': 'no', '1': 'yes'}

contents = []
for i, item in enumerate(test_data):
    query, evidence, label = item.split('\t')
    examples_pos = ''
    examples_neg = ''
    count_evi_pos = 0
    count_evi_neg = 0
    for j in range(top_K):
        rel_doc_id = I[i][j]
        data = train_data[rel_doc_id]
        query_r, evidence_r, label_r = data.split('\t')
        label_r = label_map[label_r]
        # evidence_r = process_evidence(evidence_r)
        pred = '0'
        if query_r not in outputs: continue
        if outputs[query_r]['is it a fake news'] == 'yes': pred = '1'
        if label_map[pred] == label_r and count_evi_pos < 2:
            examples_pos += 'Positive instance [{}] '.format(j) + example_prompt.replace('[QUERY]', query_r).replace('[PAGE]', evidence_r).replace('[LABEL]', label_map[pred]).replace('[EVIDENCE]', outputs[query_r]['cause'])
            count_evi_pos += 1
        elif label_map[pred] != label_r and count_evi_neg < 2:
            examples_neg += 'Negative instance [{}] '.format(j) + example_prompt.replace('[QUERY]', query_r).replace('[PAGE]', evidence_r).replace('[LABEL]', label_map[pred]).replace('[EVIDENCE]', outputs[query_r]['cause'])
            count_evi_neg += 1
        if count_evi_pos > 1 and count_evi_neg > 1: break
    content = porn_prompt.replace('[QUERY]', query).replace('[PAGE]', evidence).replace('[EXAMPLE-POS]', examples_pos).replace('[EXAMPLE-NEG]', examples_neg)
    print({'query': query, 'content': content, 'label': label})
    contents.append({'query': query, 'content': content, 'label': label})

print(len(contents))
with open('local_dataset/{}/preds/input.test.{}.json'.format(dataset, fold), mode='w') as fw:
    json.dump(contents, fw, ensure_ascii=False)
