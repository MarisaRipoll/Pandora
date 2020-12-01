from torchnlp.datasets import squad_dataset

train = squad_dataset(train=True)
train[0]['paragraphs'][0]['qas'][0]['question']
train[0]['paragraphs'][0]['qas'][0]['answers'][0]