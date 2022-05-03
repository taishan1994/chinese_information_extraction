#! -*- coding:utf-8 -*-
# 事件抽取任务，基于GPLinker
# DuEE v1.0数据集：https://aistudio.baidu.com/aistudio/competition/detail/46/0/datasets
# 文章介绍：https://kexue.fm/archives/8926

import json
import numpy as np
from itertools import groupby
from bert4keras.backend import keras, K
from bert4keras.backend import sparse_multilabel_categorical_crossentropy
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from tqdm import tqdm

maxlen = 128
batch_size = 28
epochs = 5
learning_rate = 2e-5
config_path = '../model_hub/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model_hub/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model_hub/chinese_L-12_H-768_A-12/vocab.txt'

# 读取schema
labels = []
with open('../data/ee/duee/duee_event_schema.json') as f:
    for l in f:
        l = json.loads(l)
        t = l['event_type']
        for r in [u'触发词'] + [s['role'] for s in l['role_list']]:
            labels.append((t, r))


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'events': [[(type, role, argument, start_index)]]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = {'text': l['text'], 'events': []}
            for e in l['event_list']:
                d['events'].append([(
                    e['event_type'], u'触发词', e['trigger'],
                    e['trigger_start_index']
                )])
                for a in e['arguments']:
                    d['events'][-1].append((
                        e['event_type'], a['role'], a['argument'],
                        a['argument_start_index']
                    ))
            D.append(d)
    return D


# 加载数据集
train_data = load_data('../data/ee/duee/duee_train.json')
valid_data = load_data('../data/ee/duee/duee_dev.json')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d['text'], maxlen=maxlen)
            mapping = tokenizer.rematch(d['text'], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            # 整理事件
            events = []
            for e in d['events']:
                events.append([])
                for t, r, a, i in e:
                    label = labels.index((t, r))
                    start, end = i, i + len(a) - 1
                    if start in start_mapping and end in end_mapping:
                        start, end = start_mapping[start], end_mapping[end]
                        events[-1].append((label, start, end))
            # 构建标签
            argu_labels = [set() for _ in range(len(labels))]
            head_labels, tail_labels = set(), set()
            for e in events:
                for l, h, t in e:
                    argu_labels[l].add((h, t))
                for i1, (_, h1, t1) in enumerate(e):
                    for i2, (_, h2, t2) in enumerate(e):
                        if i2 > i1:
                            head_labels.add((min(h1, h2), max(h1, h2)))
                            tail_labels.add((min(t1, t2), max(t1, t2)))
            for label in argu_labels + [head_labels, tail_labels]:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            argu_labels = sequence_padding([list(l) for l in argu_labels])
            head_labels = sequence_padding([list(head_labels)])
            tail_labels = sequence_padding([list(tail_labels)])
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_argu_labels.append(argu_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_argu_labels = sequence_padding(
                    batch_argu_labels, seq_dims=2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims=2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims=2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_argu_labels, batch_head_labels, batch_tail_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis=1))


# 加载预训练模型
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False
)
output = base.model.output

# 预测结果
argu_output = GlobalPointer(heads=len(labels), head_size=64)(output)
head_output = GlobalPointer(heads=1, head_size=64, RoPE=False)(output)
tail_output = GlobalPointer(heads=1, head_size=64, RoPE=False)(output)
outputs = [argu_output, head_output, tail_output]

# 构建模型
model = keras.models.Model(base.model.inputs, outputs)
model.compile(loss=globalpointer_crossentropy, optimizer=Adam(learning_rate))
model.summary()


class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）
    """
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def extract_events(text, threshold=0, trigger=True):
    """抽取输入text所包含的所有事件
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取论元
    argus = set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        argus.add(labels[l] + (h, t))
    # 构建链接
    links = set()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if outputs[1][0, min(h1, h2), max(h1, h2)] > threshold:
                    if outputs[2][0, min(t1, t2), max(t1, t2)] > threshold:
                        links.add((h1, t1, h2, t2))
                        links.add((h2, t2, h1, t1))
    # 析出事件
    events = []
    for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
        for event in clique_search(list(sub_argus), links):
            events.append([])
            for argu in event:
                start, end = mapping[argu[2]][0], mapping[argu[3]][-1] + 1
                events[-1].append(argu[:2] + (text[start:end], start))
            if trigger and all([argu[1] != u'触发词' for argu in event]):
                events.pop()
    return events


def evaluate(data, threshold=0):
    """评估函数，计算f1、precision、recall
    """
    ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
    ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别
    for d in tqdm(data, ncols=0):
        pred_events = extract_events(d['text'], threshold, False)
        # 事件级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            if any([argu[1] == u'触发词' for argu in event]):
                R.append(list(sorted(event)))
        for event in d['events']:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1
        ey += len(R)
        ez += len(T)
        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                if argu[1] != u'触发词':
                    R.append(argu)
        for event in d['events']:
            for argu in event:
                if argu[1] != u'触发词':
                    T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1
        ay += len(R)
        az += len(T)
    e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
    a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az
    return e_f1, e_pr, e_rc, a_f1, a_pr, a_rc


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_e_f1 = 0.
        self.best_val_a_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        e_f1, e_pr, e_rc, a_f1, a_pr, a_rc = evaluate(valid_data)
        if e_f1 >= self.best_val_e_f1:
            self.best_val_e_f1 = e_f1
            model.save_weights('../checkpoint/ee/best_model.e.weights')
        if a_f1 >= self.best_val_a_f1:
            self.best_val_a_f1 = a_f1
            model.save_weights('../checkpoint/ee/best_model.a.weights')
        print(
            '[event level] f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f'
            % (e_f1, e_pr, e_rc, self.best_val_e_f1)
        )
        print(
            '[argument level] f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n'
            % (a_f1, a_pr, a_rc, self.best_val_a_f1)
        )


def isin(event_a, event_b):
    """判断event_a是否event_b的一个子集
    """
    if event_a['event_type'] != event_b['event_type']:
        return False
    for argu in event_a['arguments']:
        if argu not in event_b['arguments']:
            return False
    return True


def predict(in_file):
  with open(in_file) as fr:
      for i,l in enumerate(fr):
          l = json.loads(l)
          event_list = DedupList()
          for event in extract_events(l['text']):
              final_event = {
                  'event_type': event[0][0],
                  'arguments': DedupList()
              }
              for argu in event:
                  if argu[1] != u'触发词':
                      final_event['arguments'].append({
                          'role': argu[1],
                          'argument': argu[2]
                      })
              event_list = [
                  event for event in event_list
                  if not isin(event, final_event)
              ]
              if not any([isin(final_event, event) for event in event_list]):
                  event_list.append(final_event)
          l['event_list'] = event_list
          print(l)
          if i == 10:
            break

def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            event_list = DedupList()
            for event in extract_events(l['text']):
                final_event = {
                    'event_type': event[0][0],
                    'arguments': DedupList()
                }
                for argu in event:
                    if argu[1] != u'触发词':
                        final_event['arguments'].append({
                            'role': argu[1],
                            'argument': argu[2]
                        })
                event_list = [
                    event for event in event_list
                    if not isin(event, final_event)
                ]
                if not any([isin(final_event, event) for event in event_list]):
                    event_list.append(final_event)
            l['event_list'] = event_list
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':


    do_train = True
    do_predict = True

    if do_train:
      train_generator = data_generator(train_data, batch_size)
      evaluator = Evaluator()

      model.fit(
          train_generator.forfit(),
          steps_per_epoch=len(train_generator),
          epochs=epochs,
          callbacks=[evaluator]
      )

    if do_predict:
      model.load_weights('../checkpoint/ee/best_model.e.weights')
      predict_to_file('../data/ee/duee/duee_test2.json', 'duee.json')
      predict('../data/ee/duee/duee_test2.json')
