import sys
import os
main_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
from utils.misc import clean_string, read_pretrain_glove, read_pretrain_fasttext, match_weight_matrix
from torch.utils.data import Dataset
import torch
import logging
from tqdm import tqdm
from collections import defaultdict
from nltk import word_tokenize
import numpy as np
import pickle


logger = logging.getLogger('dataset')
logger.setLevel(logging.DEBUG)


class Vocabulary:
    """Vocubulary for Dataset 
    
    Read the train dataset's words into vocabulary;
    Conduct necessary preprocessing;
    Obtain word2index and topic2index dictionary;

    """

    def __init__(self):
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # generate w2i, t2i, and train data
        self.get_vocab()

    def get_num_words(self):
        return len(self.w2i)

    def get_num_topics(self):
        return len(self.t2i)

    def get_dataset(self, split):
        if split == 'train':
            try:
                return self.train_data
            except:
               self.train_data = self.read_dataset(
                   os.path.join(main_dir, 'data/topicclass_train.txt'))
               return self.train_data
        elif split == 'valid':
            try:
                return self.valid_data
            except:
                self.valid_data = self.read_dataset(
                    os.path.join(main_dir, 'data/topicclass_valid.txt'))
                return self.valid_data
        elif split == 'test':
            try:
                return self.test_data
            except:
                self.test_data = self.read_dataset(
                    os.path.join(main_dir, 'data/topicclass_test.txt'))
                return self.test_data
        else:
            raise ValueError("Unkown split, split must in train/valid/test!")

    def get_vocab(self):
        """ Generate vocabulary from train dataset """


        # self.w2i
        self.w2i = defaultdict(lambda: len(self.w2i))
        # self.t2i
        self.t2i = defaultdict(lambda: len(self.t2i))
        PAD = self.w2i['[PAD]']
        UNK = self.w2i['[UNK]']
        self.w2i['[CLS]']
        self.w2i['[SEP]']

        filename = os.path.join(main_dir, 'data/topicclass_train.txt')
        self.train_data = []

        with open(filename, "r") as f:
            for line in tqdm(f):
                topic, text = line.strip().split(" ||| ")
                # sentence = text.lower().split(" ")
                sentence = word_tokenize(text.lower())
                sentence = [self.w2i[w] for w in sentence]
                sentence += [self.w2i['[SEP]']]
                sentence = [self.w2i['[CLS]']] + sentence
                # make train data
                self.train_data.append(
                    (sentence, self.t2i[topic]))
        
        # fix the word2index thus any new words in valid and test dataset will be unkown
        self.w2i = defaultdict(lambda: UNK, self.w2i)


    def read_dataset(self, filename):
        """ Read rawdata using word2index and topic2index """
        data = []
        logger.info("Reading {} into dataset...".format(filename))
        with open(filename, "r") as f:
            for line in tqdm(f):
                topic, text = line.strip().split(" ||| ")
                # sentence = text.lower().split(" ")
                sentence = word_tokenize(text.lower())
                sentence = [self.w2i[w] for w in sentence]
                sentence += [self.w2i['[SEP]']]
                sentence = [self.w2i['[CLS]']] + sentence
                data.append((sentence, self.t2i[topic]))
        return data

    def get_word2vec_weight(self, model_name, emb_size):
        """ Get weight matrix from pretrained word2vec 
        
        Args:
            model_name: str glove.6B
            emb_size: 50
        
        Returns:
            weight_matrix: len(self.wid) x emb_size
        """

        filename = os.path.join(
            main_dir, 'pretrained_word2vec/{}.{}d'.format(model_name, emb_size))

        if not os.path.exists(filename + '.npy'):
            if 'glove' in filename:
                # read raw data from .txt file
                word2vec = read_pretrain_glove(filename + '.txt')
            elif 'fasttext' in filename:
                word2vec = read_pretrain_fasttext(filename + '.vec')
            # match weight matrix
            weight_matrix = match_weight_matrix(word2vec, self.w2i, emb_size)
            # save weight matrix
            np.save(filename, weight_matrix)
        else:
            # if pre-processed weight matrix already exist, use it directly
            weight_matrix = np.load(filename + '.npy')
        logger.info("load pretrained word embedding from {}".format(filename))
        return weight_matrix


class TopicClassDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        words, topic = self.dataset[index]
        return words, topic, len(words)


if __name__ == '__main__':
    vocab = Vocabulary()
    num_words = vocab.get_num_words()
    num_topics = vocab.get_num_topics()
    dataset = TopicClassDataset(vocab.get_dataset('test'))
    print(len(dataset))

    weight_matrix_1 = vocab.get_word2vec_weight(model_name='glove.840B', emb_size=300)
    weight_matrix_2 = vocab.get_word2vec_weight(model_name='glove.42B', emb_size=300)
    weight_matrix_3 = vocab.get_word2vec_weight(model_name='fasttext-crawl', emb_size=300)
    weight_matrix_4 = vocab.get_word2vec_weight(
        model_name='glove.6B', emb_size=300)

    for i in range(10):
        words, topics, length = dataset.__getitem__(i)
        print(words)
        print(topics)
