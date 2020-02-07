import os
import sys
import torch
import numpy as np
import argparse
import random
import logging
from tqdm import tqdm

from model import TopicClassLSTM
from dataset import Vocabulary

def eval_validation_set(filename, w2i, t2i, model, device):
    model.eval()
    data = []

    with open(filename, 'r') as f:
        for line in f:
            topic, sentence = line.lower().strip().split(" ||| ")
            sentence = sentence.split(" ")
            sentence = [w2i[w] for w in sentence]
            sentence += [w2i['[SEP]']]
            sentence = [w2i['[CLS]']] + sentence
            data.append((sentence, t2i[topic]))

    correct = 0
    total = 0
    confusion_matrix = np.zeros((len(t2i), len(t2i)), dtype=np.int32)
    with torch.no_grad():
        for sent, gt_topic in tqdm(data):
            length = [len(sent)]
            sent = torch.tensor([sent]).type(torch.LongTensor).to(device)
            logits = model(sent, length)
            _, pred_topic = torch.max(logits, 1)
            pred_topic = pred_topic.cpu().numpy()[0]

            confusion_matrix[pred_topic][gt_topic] += 1
            if pred_topic == gt_topic:
                correct += 1
            total += 1
    
    acc = correct/total
    return acc, confusion_matrix


def main(config):
    print("loading the best model...")
    vocabulary = Vocabulary()
    vocab_size = vocabulary.get_num_words()

    model = TopicClassLSTM(
        vocab_size=vocab_size,  
        emb_size=config.emb_size,
        dropout=config.dropout,
        embedding_tensor=None,
        freeze=False,
        lstm_hidden=config.lstm_hidden,
        num_classes=config.num_classes)

    checkpoint = torch.load(config.checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)

    filename = '/home/haochen/Projects/cnn-topic-classification/data/topicclass_valid.txt'
    
    acc, confusion_matrix = eval_validation_set(
        filename, vocabulary.w2i, vocabulary.t2i, model, device)

    print("\n {}".format(confusion_matrix))
    print("val acc: {0:.4f}".format(acc * 100))
    
    model_name = os.path.split(config.checkpoint_dir)[-1]
    txt_name = os.path.join(os.path.split(os.path.split(config.checkpoint_dir)[0])[0], 'results', 'val_acc_{}.txt'.format(model_name))
    result_str = "val acc: {0:.4f}".format(acc * 100)
    with open(txt_name, 'a') as f:
        result_str += '\n'
        f.write(result_str)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN text classificer')

    # model
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.3]')
    parser.add_argument('--emb-size', type=int, default=300, help='number of embedding dimension [default: 200]')
    parser.add_argument('--num-classes', type=int, default=16, help='number of tpoics [default: 16]')
    parser.add_argument('--lstm-hidden', type=int, default=300, help='number of tpoics [default: 16]')

    # device
    parser.add_argument('--checkpoint_dir', default='/home/haochen/Projects/cnn-topic-classification/experiments/lstm_static_plain_wordvocab_wd_1e-3_dropout_0.3_bs_64_lr_1e-3_200205_191054/checkpoints/model_best.pth',
                        type=str, help='pretrained checkpoint')
    config = parser.parse_args()
    
    main(config)
