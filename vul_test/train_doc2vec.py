import sys
sys.path.append('')  # 目录
import src.data as data

# from data import dataManager as data
from src.utils import codeTokenizer
from gensim.models.doc2vec import Doc2Vec
import os
import pandas as pd
from tqdm import tqdm
def get_tokens(raw_path, tokens_path, tokenizer_path):
    # 获取源代码分词 训练bpe分词器 并使用bpe分词
    raw_dataset = data.read(raw_path)
    tokens_dataset = data.tokenize_subword(raw_dataset)
    bpe_tokenizer = codeTokenizer.BPE(tokens_dataset, True, tokenizer_path)
    tokens_dataset.tokens = tokens_dataset.tokens.apply(lambda x: bpe_tokenizer.encode(x, is_pretokenized=True).tokens)
    data.write(tokens_dataset, tokens_path)


def doc2vec_train(tokens_path, model_path):
    from gensim.models.doc2vec import TaggedDocument
    # Word2VecConfig = settings.Word2VecConfig()
    print(f"train doc2vec from {tokens_path}")
    tokens_dataset = data.read(tokens_path)
    d2vmodel = Doc2Vec(vector_size=50, min_count=3, epochs=40)
    tagged_data = [TaggedDocument(words=row['tokens'], tags=[str(i)]) for i, row in tokens_dataset.iterrows()]
    d2vmodel.build_vocab(tagged_data)
    d2vmodel.train(tagged_data, epochs=d2vmodel.epochs,
    total_examples=d2vmodel.corpus_count, total_words=d2vmodel.corpus_total_words)
    # path = "./data/w2vmodel/d2vmodel_sard.model"
    d2vmodel.save(model_path)


if __name__ == '__main__':
    
    print('start')
    raw_data_path = '/data/nvd/raw/nvd.pkl'
    tokens_path = '/data/nvd/tokens/tokens.pkl'
    tokenizer_path = '/data/nvd/tokenizer/tokenizer.json'
    d2vmodel_path = '/data/nvd/d2vmodel/d2vmodel.model'
    
    get_tokens(raw_data_path, tokens_path, tokenizer_path)
    doc2vec_train(tokens_path, d2vmodel_path)
    
    print('over')
