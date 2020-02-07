import os
import sys
import torch
import numpy as np
import argparse
import random
import logging
from tqdm import tqdm

# fix random seeds for reproducibility
seed = 1997
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True

from trainer import Trainer
from utils.config import process_config


def eval_validation_set(filename, w2i, t2i, tokenizer, model, device):
    model.eval()
    data = []

    with open(filename, 'r') as f:
        for line in f:
            topic, text = line.lower().strip().split(" ||| ")
            sentence = tokenizer.tokenize(text)
            sentence = [w2i[w] for w in sentence]
            sentence += [w2i['[SEP]']]
            sentence = [w2i['[CLS]']] + sentence
            data.append((sentence, t2i[topic]))

    correct = 0
    total = 0
    confusion_matrix = np.zeros((len(t2i), len(t2i)), dtype=np.int32)
    with torch.no_grad():
        for sent, gt_topic in tqdm(data):
            sent = torch.tensor([sent]).type(torch.LongTensor).to(device)
            mask = torch.ones_like(sent).type(torch.FloatTensor).to(device)
            logits = model(sent, mask)
            _, pred_topic = torch.max(logits, 1)
            pred_topic = pred_topic.cpu().numpy()[0]

            confusion_matrix[pred_topic][gt_topic] += 1
            if pred_topic == gt_topic:
                correct += 1
            total += 1
    
    acc = correct/total
    return acc, confusion_matrix


def main(config):
    trainer = Trainer(config)
    trainer.train()

    logger = logging.getLogger("Evaluation")
    logger.info("loading the best model...")
    
    model = trainer.model
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'model_best.pth'))
    logger.info("loading best checkpoint from {}".format(checkpoint))
    model.load_state_dict(checkpoint['state_dict'])
    vocab = trainer.vocabulary
    device = trainer.device

    logger.info("evaluating the validation set")


    filename = '/home/haochen/Projects/cnn-topic-classification/data/topicclass_valid.txt'
    
    acc, confusion_matrix = eval_validation_set(
        filename, vocab.w2i, vocab.t2i, vocab.tokenizer, model, device)

    logger.info("\n {}".format(confusion_matrix))
    logger.info("val acc: {0:.4f}".format(acc * 100))

    txt_name = os.path.join(config.result_dir, 'val_acc.txt')
    result_str = "val acc: {0:.4f}".format(acc * 100)
    with open(txt_name, 'a') as f:
        result_str += '\n'
        f.write(result_str)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN text classificer')
    parser.add_argument('--exp-name', type=str, default="cnn_static", help='exp name')
    # learning
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay [default: 0.0001]')
    parser.add_argument('--exp-lr-gamma', type=float, default=0.95, help='exponential lr decay rate [default: 0.95]')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs for train [default: 200]')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('--early-stop', type=int, default=5, help='early stopping patience epochs [default: 10]')
    parser.add_argument('--monitor', type=str, default="min val_loss", help='whether to save when get best performance [default: min val_loss]')

    # model
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers [default: 4]')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.3]')
    parser.add_argument('--emb-size', type=int, default=300, help='number of embedding dimension [default: 200]')
    parser.add_argument('--num-classes', type=int, default=16, help='number of tpoics [default: 16]')
    parser.add_argument('--num-feat-maps', type=int, default=100,help='number of feature maps output of cnn [default: 100]')
    parser.add_argument('--kernel-sizes', nargs='+', default=[3, 4, 5], help='kernel sizes [default: [3, 4, 5]]')
    parser.add_argument('--pretrain_emb', action='store_true', default=True, help='use pretrained word embedding')
    parser.add_argument('--static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to train the model [default: 1]')
    parser.add_argument('--summary-step', type=int, default=100, help='summary step [default: 50]')
    parser.add_argument('--save-period', type=int, default=1, help='period of epochs for saving checkpoint [default: 2]')
    parser.add_argument('--checkpoint', type=str, help='pretrained checkpoint')
    parser.add_argument('--resume_epoch', type=int, help='pretrained checkpoint epoch')
    args = parser.parse_args()
    config = process_config(args)
    
    main(config)