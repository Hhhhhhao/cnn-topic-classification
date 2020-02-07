import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from dataset import TopicClassDataset


def pad_collate_fn(batch, padding_value):
    sentences = []
    topics = []
    lengths = []
    # pad the sentence to equal length
    for sent, topic, length in batch:
        sentences.append(torch.tensor(sent).type(torch.LongTensor))
        topics.append(topic)
        lengths.append(length)
    
    paded_sentences = pad_sequence(sentences, batch_first=True, padding_value=int(padding_value))
    # turn sentences and topics into LongTensors
    topics = torch.tensor(topics).type(torch.LongTensor)

    # sort the sentences according to its lengths
    perm_idx = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
    paded_sentences = paded_sentences[perm_idx]
    topics = topics[perm_idx]
    lengths[:] = [lengths[i] for i in perm_idx]
    
    # create a mask for the padded values
    masks = (paded_sentences > 0).type(torch.FloatTensor)

    return paded_sentences, topics, lengths, masks


class TopicClassDataLoader(DataLoader):
    def __init__(self, vocabulary, split, batch_size, num_workers=0):
        self.dataset = TopicClassDataset(vocabulary.get_dataset(split))
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if split is 'train':
            # all labels
            all_labels = self.dataset.get_labels()
            class_sample_count = np.array(
                [len(np.where(all_labels == t)[0]) for t in np.unique(all_labels)])
            
            self.weight = 1. / class_sample_count
            samples_weight = np.array([self.weight[t] for t in all_labels])
            self.samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler = RandomSampler(self.dataset)
            

        super(TopicClassDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=lambda batch: pad_collate_fn(batch, padding_value=0),
            num_workers=self.num_workers
        )

if __name__ == '__main__':
    from dataset import Vocabulary
    vocab = Vocabulary()

    data_loader = TopicClassDataLoader(vocab, split='train', batch_size=16, num_workers=0)
    for i, (sent, topic, lengths, masks) in enumerate(data_loader):
        # print(sent.size)
        print(sent)
        print(masks)
        print(topic)

