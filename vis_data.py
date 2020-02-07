import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

topics = []
words = []
filename = 'data/topicclass_train.txt'
with open(filename, 'r') as f:
    for line in tqdm(f):
        topic, text = line.strip().split(" ||| ")
        topics.append(topic)
        words.extend(text.lower().split(" "))
words_cnt = Counter(words)

topics_cnt = Counter(topics)

labels, values = zip(*topics_cnt.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

bar_width = 0.1
plt.figure(figsize=(20,5))
plt.barh(indexes, values)

# add labels
plt.yticks(indexes + bar_width, labels)
plt.savefig('topic.png')
