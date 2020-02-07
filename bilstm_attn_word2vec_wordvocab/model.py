import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TopicClassLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_tensor, freeze, emb_size, dropout, lstm_hidden, num_classes=16):
        super(TopicClassLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_num_layers = 2
        self.dropout = dropout


        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        if embedding_tensor is not None:   
            self.init_emb(embedding_tensor, freeze)
        self.lstm = nn.LSTM(self.emb_size, self.lstm_hidden, num_layers=self.lstm_num_layers, bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

        self.dropout_1 = nn.Dropout(dropout) 
        self.fc1 = nn.Linear(self.lstm_hidden * self.lstm_num_layers * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout_2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, self.num_classes)

    def init_emb(self, weight_tensor, freeze):
        """ initialize the embedding tensor from a pretrained weight tensor """
        self.embedding.from_pretrained(weight_tensor, freeze=freeze)
    
    def forward(self, x, x_len):
        x = self.embedding(x)  # (B, L, emb_size)

        packed_x = pack_padded_sequence(x, x_len, batch_first=True)
        packed_out, (ht, ct) = self.lstm(packed_x)
        out, input_sizes = pad_packed_sequence(packed_out, batch_first=True)  # out (B, L, hidden * num_layers)

        # final hidden layer
        batch_size = ht.size(1)
        ht_final = ht.view(self.lstm_num_layers, 2, batch_size, self.lstm_hidden)[-1, :, :, :]   # (2, B, Hidden)
        hidden_state = torch.cat([ht_final[i] for i in range(ht_final.size(0))], dim=1)

        # apply attention
        hidden_state = hidden_state.unsqueeze(2) # (B, hidden * 2, 1)
        attention_scores = torch.bmm(out, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)   # (B, L, 1)
        attention_out = torch.bmm(out.permute(0, 2, 1), soft_attention_weights).squeeze(2)

        features = torch.cat([hidden_state.squeeze(2), attention_out], dim=1)
        features = self.dropout_1(features)
        features = self.fc1(features)
        features = self.bn1(features)
        features = F.relu(features)
        features = self.dropout_2(features)
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
    model = TopicClassLSTM(vocab_size=11, emb_size=10, embedding_tensor=None, freeze=False,
                           dropout=0.2, lstm_hidden=200, num_classes=16)
    out = model(x, len_x)
