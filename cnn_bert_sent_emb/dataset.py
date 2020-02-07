from pytorch_pretrained_bert import BertTokenizer
from utils.misc import clean_string
from torch.utils.data import Dataset
import torch
import logging
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pickle
import os
main_dir = os.path.dirname(os.path.dirname(__file__))


logger = logging.getLogger('dataset')
logger.setLevel(logging.DEBUG)


class Vocabulary:
    """Vocubulary for Dataset 
    
    Read the train dataset's words into vocabulary;
    Conduct necessary preprocessing;
    Obtain word2index and topic2index dictionary;

    """

    def __init__(self):
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(
            main_dir, 'pretrained_bert', 'uncased_L-12_H-768_A-12', 'vocab.txt'))

        # generate w2i, t2i, and train data
        self.get_vocab()


            # self.get_dataset(split='train')

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
        # create word2index and topic2index dict
        w2i = defaultdict(lambda: len(w2i))
        filename = os.path.join(
            main_dir, 'pretrained_bert/uncased_L-12_H-768_A-12/vocab.txt')
        
        with open(filename, "r") as f:
            for word in f:
                index = w2i[word.rstrip('\n')]

        UNK = w2i['[UNK]'] 
        # fix the word2index thus any new words in valid and test dataset will be unkown
        self.w2i = defaultdict(lambda: UNK, w2i)

        # self.t2i
        self.t2i = defaultdict(lambda: len(self.t2i))

        filename = os.path.join(main_dir, 'data/topicclass_train.txt')
        self.train_data = []

        with open(filename, "r") as f:
            for line in tqdm(f):
                topic, text = line.lower().strip().split(" ||| ")
                sentence = self.tokenizer.tokenize(text)
                sentence = [self.w2i[w] for w in sentence]
                sentence += [self.w2i['[SEP]']]
                sentence = [self.w2i['[CLS]']] + sentence
                # make train data
                self.train_data.append(
                    (sentence, self.t2i[topic]))

    def read_dataset(self, filename):
        """ Read rawdata using word2index and topic2index """
        data = []
        logger.info("Reading {} into dataset...".format(filename))
        with open(filename, "r") as f:
            for line in tqdm(f):
                topic, text = line.lower().strip().split(" ||| ")
                sentence = self.tokenizer.tokenize(text)
                sentence = [self.w2i[w] for w in sentence]
                sentence += [self.w2i['[SEP]']]
                sentence = [self.w2i['[CLS]']] + sentence
                data.append((sentence, self.t2i[topic]))
        return data


class TopicClassDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        words, topic = self.dataset[index]
        return words, topic


if __name__ == '__main__':
    vocab = Vocabulary()
    num_words = vocab.get_num_words()
    num_topics = vocab.get_num_topics()
    dataset = TopicClassDataset(vocab.get_dataset('valid'))
    print(len(dataset))

    for i in range(10):
        words, topics = dataset.__getitem__(i)
        print(words)
        print(topics)
