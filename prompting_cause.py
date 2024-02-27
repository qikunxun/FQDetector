from utils import get_gpt_singal

dataset = 'snopes'
def process_evidence(evidence):
    texts = evidence.split('||')
    text_new = []
    for text in texts:
        text_new.append('{}'.format(len(text_new) + 1) + text[:200])
    return ''.join(text_new)

porn_prompt = """
You are a fake news detection expert. Due to the serious impact of fake news on user experience, you are now required to judge whether the news is a rumor based on the input news title, its search results, and your own knowledge. The criteria are as follows:
1. If all search results support the news or are irrelevant to the news, it is judged as not fake.
2. If there is any search result that opposes/denies/debunks this news, it is judged as fake. 
***Please note, it is crucial to ensure accuracy as this concerns the user experience.***
Output format：{"cause"："..."， "is it a fake news": "yes/no"}
For the news：[QUERY]，its search results are :[PAGE]。
Based on these contents，Do you think the news: [QUERY] is it a fake news?
"""

def request_gpt(input_file):
    with open(input_file) as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            try:
                query = items[0]
                page = items[1]
                page_text = page
                content = porn_prompt.replace('[QUERY]', query).replace('[PAGE]', page_text)
                ret, response, code = get_gpt_singal(content, model="gpt-3.5-turbo")
                print('{}\t{}'.format(query, response.replace('\n', '')))
            except Exception as e:
                print('{}\t{}'.format(query, 'missing'))


if __name__ == "__main__":
    for i in range(5):
        request_gpt('{}/processed/test_{}.txt')
