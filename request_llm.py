import json
import time
from utils import get_gpt_singal

dataset = 'snopes'
fold = '0'
def request_gpt(input_file):
    with open(input_file) as fd:
        data = json.load(fd)
        for item in data:
            try:
                query = item['query']
                content = item['content']
                ret, response, code = get_gpt_singal(content, model="gpt-4")
                print('{}\t{}'.format(query, response.replace('\n', '')))
                time.sleep(1)
            except Exception as e:
                print('{}\t{}'.format(query, 'missing'))

if __name__ == "__main__":
    request_gpt('{}/preds/input.test.{}.json'.format(dataset, fold))