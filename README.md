## Code of the paper "Detecting Fake Queries in Query Recommendation via Large Language Model"
## Prerequisites

 * Python 3.9
 * TensorFlow 1.15.0
 * Faiss-cpu 1.7.4


## Datasets
We use Snopes, PolitFact, FakeQQB in our experiments.

| Dataset           | Download Links                                                       |
|-------------------|----------------------------------------------------------------------|
| Snopes            | https://github.com/CRIPAC-DIG/GET/                                   |
| PolitiFact        | https://github.com/CRIPAC-DIG/GET/                                   |
| FakeQQB           | This work.                                                           |


## Use examples
For data preprocess, you can run the following scripts:

```
python process_data.py
```

For cause generation, you can run the following scripts:

```
python prompting_cause.py
python request_llm.py
```

For fake query/news detection, you can run the following scripts:

```
python faiss_retreive_evi.py
python request_llm.py
```

For evaluation, you can run the following scripts:

```
python evaluate.py
```



