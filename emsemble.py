import os
import sys
import torch
import numpy as np
import argparse
import random
import logging
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from nltk import word_tokenize

from cnn_word2vec_wordvocab_fc2_wordpiece.dataset import Vocabulary as CNN_Vocabulary
from cnn_bert_sent_emb.dataset import Vocabulary as BERT_Vocabulary

from cnn_word2vec_wordvocab_fc2_wordpiece.model import TopicClassCNN as CNN
from bilstm_word2vec_wordvocab_wordpiece.model import TopicClassLSTM as LSTM
from cnn_bert_sent_emb.models import TopicClassCNN as BERT_SENT
from cnn_bert_word_emb.models import TopicClassCNN as BERT_WORD


# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")
tokenizer = BertTokenizer(vocab_file=os.path.join('pretrained_bert', 'uncased_L-12_H-768_A-12', 'vocab.txt'))

print("loading the best model for CNN and LSTM")

cnn_vocabulary = CNN_Vocabulary()
cnn_vocab_size = cnn_vocabulary.get_num_words()

bert_vocabulary = BERT_Vocabulary()
bert_vocab_size = bert_vocabulary.get_num_words()

cnn_model = CNN(vocab_size=cnn_vocab_size, emb_size=300, dropout=0.5)
checkpoint = torch.load(
    '/home/haochen/Projects/cnn-topic-classification/experiments/cnn_non_static_wordpiece_fasttext_crawl_300d_wd_1e-2_dropout_0.25_bs_32_lr_1e-3_fc_2_200207_045959/checkpoints/model_best.pth')
cnn_model.load_state_dict(checkpoint['state_dict'])
cnn_model = cnn_model.to(device)
cnn_model.eval()

lstm_model = LSTM(vocab_size=cnn_vocab_size, emb_size=300,
                  lstm_hidden=512, dropout=0.5)
checkpoint = torch.load(
                        '/home/haochen/Projects/cnn-topic-classification/experiments/bilstm_non_static_wordpiece_fasttext_crawl_300d_wd_1e-1_dropout_0.25_bs_32_lr_1e-3_hidden_512_fc_2_200207_063425/checkpoints/model_best.pth')
lstm_model.load_state_dict(checkpoint['state_dict'])
lstm_model = lstm_model.to(device)
lstm_model.eval()

bert_sent_model = BERT_SENT(vocab_size=bert_vocab_size)
checkpoint = torch.load(
    '/home/haochen/Projects/cnn-topic-classification/experiments/bert_sent_emb_wd_1e_3_lr_5e_5_bs_32_200207_013220/checkpoints/model_best.pth')
bert_sent_model.load_state_dict(checkpoint['state_dict'])
bert_sent_model = bert_sent_model.to(device)
bert_sent_model.eval()

bert_word_model = BERT_WORD(vocab_size=bert_vocab_size)
checkpoint = torch.load(
    '/home/haochen/Projects/cnn-topic-classification/experiments/cnn_bert_word_emb_wd_1e_3_lr_5e_5_bs_32_200207_030044/checkpoints/epoch_latest.pth')
bert_word_model.load_state_dict(checkpoint['state_dict'])
bert_word_model = bert_word_model.to(device)
bert_word_model.eval()

val_filename = '/home/haochen/Projects/cnn-topic-classification/data/topicclass_valid.txt'
test_filename = '/home/haochen/Projects/cnn-topic-classification/data/topicclass_test.txt'

i2t = dict()
for key, item in cnn_vocabulary.t2i.items():
    i2t[item] = key

results = []
with open(val_filename, 'r') as f:
    for line in tqdm(f):
        final_prob = 0
        topic, text = line.lower().strip().split(" ||| ")
        
        cnn_sentence = word_tokenize(text.lower())
        cnn_sentence = [cnn_vocabulary.w2i[w] for w in cnn_sentence]
        cnn_sentence += [cnn_vocabulary.w2i['[SEP]']]
        cnn_sentence = [cnn_vocabulary.w2i['[CLS]']] + cnn_sentence
        sent_len = [len(cnn_sentence)]
        cnn_sentence = torch.tensor([cnn_sentence]).type(
            torch.LongTensor).to(device)

        cnn_prob = cnn_model(cnn_sentence, (cnn_sentence > 0))
        final_prob += cnn_prob

        lstm_prob = lstm_model(cnn_sentence, sent_len)
        final_prob += lstm_prob 

        bert_sentence = tokenizer.tokenize(text)
        bert_sentence = [bert_vocabulary.w2i[w] for w in bert_sentence]
        bert_sentence += [bert_vocabulary.w2i['[SEP]']]
        bert_sentence = [bert_vocabulary.w2i['[CLS]']] + bert_sentence
        bert_sentence = torch.tensor([bert_sentence]).type(
            torch.LongTensor).to(device)


        bert_sent_prob = bert_sent_model(bert_sentence)
        final_prob += bert_sent_prob
        bert_word_prob = bert_word_model(bert_sentence)
        final_prob += bert_word_prob

        _, pred_topic = torch.max(final_prob, 1)
        pred_topic = pred_topic.cpu().numpy()[0]
        results.append(i2t[pred_topic])

dev_txt_name = 'dev_results.txt'
with open(dev_txt_name, 'a') as f:
    for result in results:
        f.write(result + '\n')

results = []
with open(test_filename, 'r') as f:
    for line in tqdm(f):
        final_prob = 0
        topic, text = line.lower().strip().split(" ||| ")

        cnn_sentence = word_tokenize(text.lower())
        cnn_sentence = [cnn_vocabulary.w2i[w] for w in cnn_sentence]
        cnn_sentence += [cnn_vocabulary.w2i['[SEP]']]
        cnn_sentence = [cnn_vocabulary.w2i['[CLS]']] + cnn_sentence
        sent_len = [len(cnn_sentence)]
        cnn_sentence = torch.tensor([cnn_sentence]).type(
            torch.LongTensor).to(device)

        cnn_prob = cnn_model(cnn_sentence, (cnn_sentence > 0))
        final_prob += cnn_prob

        lstm_prob = lstm_model(cnn_sentence, sent_len)
        final_prob += lstm_prob

        bert_sentence = tokenizer.tokenize(text)
        bert_sentence = [bert_vocabulary.w2i[w] for w in bert_sentence]
        bert_sentence += [bert_vocabulary.w2i['[SEP]']]
        bert_sentence = [bert_vocabulary.w2i['[CLS]']] + bert_sentence
        bert_sentence = torch.tensor([bert_sentence]).type(
            torch.LongTensor).to(device)

        bert_sent_prob = bert_sent_model(bert_sentence)
        final_prob += bert_sent_prob
        bert_word_prob = bert_word_model(bert_sentence)
        final_prob += bert_word_prob

        _, pred_topic = torch.max(final_prob, 1)
        pred_topic = pred_topic.cpu().numpy()[0]
        results.append(i2t[pred_topic])

dev_txt_name = 'test_results.txt'
with open(dev_txt_name, 'a') as f:
    for result in results:
        f.write(result + '\n')
