import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from dataset import TopicClassDataset


def pad_collate_fn(batch, padding_value):
    sentences = []
    topics = []
    # pad the sentence to equal length
    for sent, topic in batch:
        sentences.append(torch.tensor(sent).type(torch.LongTensor))
        topics.append(topic)

    paded_sentences = pad_sequence(
        sentences, batch_first=True, padding_value=int(padding_value))
    # turn sentences and topics into LongTensors
    topics = torch.tensor(topics).type(torch.LongTensor)

    return paded_sentences, topics


class TopicClassDataLoader(DataLoader):
    def __init__(self, vocabulary, split, batch_size, num_workers=0):
        self.dataset = TopicClassDataset(vocabulary.get_dataset(split))
        self.batch_size = batch_size
        self.num_workers = num_workers

        super(TopicClassDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True if split == 'train' else False,
            collate_fn=lambda batch: pad_collate_fn(batch, padding_value=0),
            num_workers=self.num_workers
        )


if __name__ == '__main__':
    from dataset import Vocabulary
    vocab = Vocabulary()

    data_loader = TopicClassDataLoader(
        vocab, split='train', batch_size=16, num_workers=0)
    for i, (sent, topic, lengths, masks) in enumerate(data_loader):
        # print(sent.size)
        print(sent)
        print(masks)
        print(topic)
