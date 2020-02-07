import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WeightClipper(object):
    def __init__(self, constraint=3):
        self.constraint = constraint
    
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.constraint, self.constraint)


class TopicClssCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, dropout, kernel_sizes=[3, 4, 5], num_feat_maps=100, num_classes=16):
        super(TopicClssCNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.num_feat_maps = num_feat_maps

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

        self.convs = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            # conv = nn.Conv2d(in_channels=1, out_channels=num_feat_maps,
            #                  kernel_size=(kernel_size, emb_size), stride=1, padding=0)
            # self.convs.append(conv)

            module = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=num_feat_maps, kernel_size=(kernel_size, emb_size), stride=1, padding=0),
                nn.BatchNorm2d(num_feat_maps)
            )

            self.convs.append(module)
        
        
        # self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(self.num_feat_maps * len(self.kernel_sizes), self.num_classes)

        self.dropout_1 = nn.Dropout(dropout)     
        self.fc1 = nn.Linear(self.num_feat_maps * len(self.kernel_sizes), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout_2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, self.num_classes)

    def init_emb(self, weight_tensor, freeze):
        """ initialize the embedding tensor from a pretrained weight tensor """
        self.embedding.from_pretrained(
            weight_tensor, freeze=freeze)
    
    def forward(self, x, masks):
        # input x's shape (B, L), L is the sentence length    
        # input masks shape (B, L), are used to masked the <pad> values in the embeddingm matrix

        x = self.embedding(x)  # (B, L, emb_size)
        x = x * masks.view(masks.size(0), -1, 1)
        # x = pack_padded_sequence(x, len_x, batch_first=True)
        # print(x)
        x = x.unsqueeze(1) # (B, 1, L, emb_size)

        features = []
        for conv in self.convs:
            feat = conv(x) # (B, 1, L, 1)
            # TODO check other activations, especially tanh
            feat = F.relu(feat)  # (B, 100, L, 1)
            feat = feat.squeeze(-1)  # (B, 100, L)
            feat =  F.max_pool1d(feat, feat.size(2)).squeeze(-1) # (B, 100)
            features.append(feat)
        
        # concatenate features
        features = torch.cat(features, dim=1)
        # features = self.dropout(features)
        # logits = self.linear(features)
        features = self.dropout_1(features)
        features = self.fc1(features)
        features = self.bn1(features)
        features = F.relu(features)
        # features = self.dropout_2(features)
        logits = self.fc2(features)
        return logits

if __name__ == '__main__':
    import torch
    from torch.nn.utils.rnn import pad_sequence

    sentences = [torch.LongTensor([2, 3, 4, 5]), torch.LongTensor([6, 7, 8]), torch.LongTensor([9, 10])]
    x = pad_sequence(sentences, batch_first=True,
                     padding_value=0)
    masks = (x != 0).type(torch.FloatTensor)
    
    len_x = [4, 3, 2]
    model = TopicClssCNN(vocab_size=11, emb_size=10, dropout=0.2, kernel_sizes=[2], num_feat_maps=20, num_classes=16)

    out = model(x, masks)
