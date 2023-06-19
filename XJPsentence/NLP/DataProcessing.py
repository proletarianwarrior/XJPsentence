# -*- coding: utf-8 -*-
# @Time : 2023/6/3 10:46
# @Author : DanYang
# @File : DataProcessing.py
# @Software : PyCharm
import re
from collections import defaultdict, Counter

import numpy as np
from text2vec import Similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def parsing_data(path):
    with open(path, 'r') as file:
        datas = file.readlines()
    results = []
    for data in datas:
        if data.endswith('.\n'):
            data = data[:-2]
        dicts = data.split(';')
        if len(dicts) <= 2:
            dicts = data.split(',')
            dicts = {d.strip(): 1 / len(dicts) for d in dicts}
            results.append(dicts)
            continue
        dicts = {re.split('[,:+]', d)[0].strip(): float(re.split('[,:+]', d)[1]) for d in dicts if d.strip() and float(re.split('[,:+]', d)[1]) <= 1}
        results.append(dicts)
    return results


def similarity_data(word1s, word2s):
    sim_model = Similarity()
    return sim_model.get_scores(word1s, word2s)


def merge_dicts(dicts, que_value=0.90):
    merged_dict = defaultdict(float)
    for dictionary in dicts:
        for key, value in dictionary.items():
            merged_dict[key] += value

    keys = list(merged_dict.keys())
    matrix = similarity_data(keys, keys)
    triu_matrix = np.triu(matrix)
    np.fill_diagonal(triu_matrix, 0)
    columns, lines = np.where(triu_matrix > que_value)

    for column, line in zip(columns, lines):
        print(keys[column], keys[line])
        merged_dict[keys[column]] += merged_dict[keys[line]]
        del merged_dict[keys[line]]
    return merged_dict


def plot_wordcloud(data):
    data = Counter(data)
    del data['习近平外交']
    del data['外交部发言人']
    del data['国家主席']
    del data['习近平']
    max_num = 50 if len(data) > 50 else len(data)
    data = {list(data.keys())[i]: list(data.values())[i] for i in range(max_num)}
    fig = plt.figure(dpi=5000)
    wordcloud = WordCloud(font_path='STSONG.TTF', background_color='white', max_words=1000, width=1000, height=500)
    wordcloud.generate_from_frequencies(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    results = parsing_data('./original_data/2023.txt')
    dicts = merge_dicts(results)
    plot_wordcloud(dicts)

