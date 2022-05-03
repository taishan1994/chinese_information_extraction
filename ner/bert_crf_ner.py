#! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm



maxlen = 256  # 句子最大长度
epochs = 10  # 训练的epoch
batch_size = 32  # batchsize大小
learning_rate = 2e-5  # 学习率
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率，crf层的学习率要设置比bert层的大一些
categories = set()  # 类别
# bert配置
config_path = '../model_hub/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model_hub/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model_hub/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D


# 标注数据
"""人民日报数据集每一行是字空格标签"""
train_data = load_data('../data/ner/china-people-daily-ner-corpus/example.train')
valid_data = load_data('../data/ner/china-people-daily-ner-corpus/example.dev')
test_data = load_data('../data/ner/china-people-daily-ner-corpus/example.test')
categories = list(sorted(categories)) 

print(categories)  # ['LOC', 'ORG', 'PER']
print(train_data[0])  # ['海钓比赛地点在厦门与金门之间的海域。', [7, 8, 'LOC'], [10, 11, 'LOC']]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
tokens = tokenizer.tokenize(train_data[0][0], maxlen=maxlen)
print(tokens)
# ['[CLS]', '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。', '[SEP]']
mapping = tokenizer.rematch(train_data[0][0], tokens)
print(mapping)
# [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], []]


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)  # 获得每个字在列表中的位置
            """额外补充知识
              from bert4keras.tokenizers import Tokenizer
              dict_path = '../model_hub/chinese_L-12_H-768_A-12/vocab.txt'
              train_data = ['我的电话是13928761123，你记住了么', [5,15,'phone']]
              tokenizer = Tokenizer(dict_path, do_lower_case=True)
              text = train_data[0]
              tokens = tokenizer.tokenize(text, maxlen=256)
              print(tokens)
              mapping = tokenizer.rematch(text, tokens)
              print(mapping)
              ['[CLS]', '我', '的', '电', '话', '是', '139', '##28', '##76', '##11', '##23', '，', '你', '记', '住', '了', '么', '[SEP]']
              [[], [0], [1], [2], [3], [4], [5, 6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16], [17], [18], [19], [20], [21], []]
              token_ids = tokenizer.tokens_to_ids(tokens)
              print(token_ids)
              [101, 2769, 4638, 4510, 6413, 3221, 9081, 8835, 9624, 8452, 8748, 8024, 872, 6381, 857, 749, 720, 102]
            """
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}  # 这两行是计算mapping在tokens中的索引
            """
              start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
              end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
              print(start_mapping)
              print(end_mapping)
              {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 8: 7, 10: 8, 12: 9, 14: 10, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16}
              {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 7: 6, 9: 7, 11: 8, 13: 9, 15: 10, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16}
            """
            token_ids = tokenizer.tokens_to_ids(tokens)
            """
              categories = ['phone']
              import numpy as np
              labels = np.zeros(len(token_ids))
              for start, end, label in train_data[1:]:
                  if start in start_mapping and end in end_mapping:
                      start = start_mapping[start]
                      end = end_mapping[end]
                      labels[start] = categories.index(label) * 2 + 1  # 采用BIO标注
                      labels[start + 1:end + 1] = categories.index(label) * 2 + 2

              print(labels)
              [0. 0. 0. 0. 0. 0. 1. 2. 2. 2. 2. 0. 0. 0. 0. 0. 0. 0.]
            """
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1  # 采用BIO标注
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(config_path, checkpoint_path)
output = Dense(len(categories) * 2 + 1)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
# model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=maxlen)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

def predict(data):
    res = []
    for i,d in enumerate(data):
        R = set(NER.recognize(d[0]))
        print(d[0])
        print(R)
        if i == 10:
            break


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('../checkpoint/ner/best_model_peopledaily_crf.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    do_train = False
    do_predict = True

    if do_train:
      evaluator = Evaluator()
      train_generator = data_generator(train_data, batch_size)

      model.fit(
          train_generator.forfit(),
          steps_per_epoch=len(train_generator),
          epochs=epochs,
          callbacks=[evaluator]
      )

    if do_predict:
      model.load_weights('../checkpoint/ner/best_model_peopledaily_crf.weights')
      # print(CRF.trans)
      # NER.trans = K.eval(CRF.trans)
      predict(valid_data)
