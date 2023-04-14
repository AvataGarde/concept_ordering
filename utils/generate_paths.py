import configparser
from dis import code_info
import json
import random
import numpy as np
from tqdm import tqdm
import igraph as ig
from torchnlp.word_to_vector import FastText
import os
import torch
#import fasttext 
import fasttext.util
import itertools as it
from itertools import product

os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read("path.ini")
conceptnet_path = 'conceptnet570'


cache_dir = "./"




def cos_sim(vec1,vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_fasttext_embedding(phrase):
    words = phrase.split("_")
    word_vectors = [ft.get_word_vector(word) for word in words]
    return np.mean(word_vectors, axis=0)



def get_commongen_embed():
    #Generate the fasttext embedding for the aligned vocabs of CommonGen
    ft = fasttext.load_model('cc.en.300.bin')
    vocabs_embedding = []
    with open(config["commongen"]["addressed_vocab"], "r", encoding="utf8") as f:
        for l in f.readlines():
            word = l.strip()
            vector = torch.reshape(torch.from_numpy(get_fasttext_embedding(word)),(1,-1))
            vocabs_embedding.append(vector)
    res = torch.cat(vocabs_embedding,dim=0)
    print(res.shape)
    torch.save(res, "concpetnet_fasttext_embeddings.pt")
            

import random

def split_dataset(filename, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 读取数据集
    with open(filename, 'r') as f:
        data = f.readlines()

    #Align the vocabs
    oov = {}
    with open(config['commongen']["oov"], "r", encoding="utf8") as f:
        for line in f.readlines():
            original_w, addressed_w = line.split("\t")[0].strip(), line.split("\t")[1].strip()
            oov[original_w] = addressed_w
            
    # 按 concept set 对数据进行分组，并将拥有相同 concept set 的行放到相邻的位置
    groups = {}
    for line in data:
        cols = line.strip().split("\t")
        words = cols[0].split(' ')
        vocabs = []
        for word in words:
            if word in oov.keys():
                    vocabs.append(oov[word])
            else:
                vocabs.append(word)
        concept_set = frozenset(vocabs)
        if concept_set not in groups:
            groups[concept_set] = []
        groups[concept_set].append(line)

    # 计算每个 concept set 的训练集、验证集和测试集大小，并计算出每个 concept set 的训练集、验证集和测试集的索引
    n_groups = len(groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    n_test = n_groups - n_train - n_val
    group_indices = list(range(n_groups))
    random.shuffle(group_indices)
    train_indices = group_indices[:n_train]
    val_indices = group_indices[n_train:n_train+n_val]
    test_indices = group_indices[n_train+n_val:]

    # 将分组后的数据按训练集、验证集和测试集的索引分配到三个列表中
    train_data, val_data, test_data = [], [], []
    for i, group in enumerate(groups.values()):
        if i in train_indices:
            train_data.extend(group)
        elif i in val_indices:
            val_data.extend(group)
        else:
            test_data.extend(group)

    return train_data, val_data, test_data


def generate_plan_matrix(filename,savename):
    commongen2id = {}
    id2commongen = {}
    
    with open(config['commongen']['addressed_vocab'], 'r', encoding="utf8") as f:
        for w in f.readlines():
            word = w.strip()
            commongen2id[word] = len(commongen2id)
            id2commongen[len(id2commongen)] = word
    print("commongen2id done")
    
    result_sample = np.zeros((len(commongen2id), len(commongen2id)), dtype=float)
    start_sample = np.zeros((len(commongen2id),), dtype=float)
    total_sample = np.zeros((len(commongen2id),), dtype=float)
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            words = line.split("\t")[0].strip().split(' ')
            commongen_id = [commongen2id[c] for c in words]

            start_sample[commongen_id[0]] += 1
            for i in range(len(commongen_id) - 1):
                result_sample[commongen_id[i]][commongen_id[i + 1]] += 1
                total_sample[commongen_id[i]] += 1
    transition_matrix = np.divide(result_sample, total_sample[:, None],
                                  out=np.zeros_like(result_sample, dtype=np.float64), where=total_sample[:, None] != 0)
    np.savez(savename, transition=transition_matrix)
    print(np.count_nonzero(result_sample))


if __name__ == "__main__":
    generate_plan_matrix("test.txt","test_matrix.npz")