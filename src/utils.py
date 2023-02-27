import configparser
import json
from tqdm import tqdm
import numpy as np
import itertools
import time
import numpy as np
from scipy import stats

import math
import os

os.chdir(os.path.dirname(__file__))
config = configparser.ConfigParser()
config.read("path.ini")



def asMinutes(s):
    h = math.floor(s / 3600)
    s = s - h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def load_vocabs():
    commongen2id = dict()
    id2commongen = dict()
    oov = dict()

    with open(config['commongen']["oov"], "r", encoding="utf8") as f:
        for line in f.readlines():
            original_w, addressed_w = line.split("\t")[0].strip(), line.split("\t")[1].strip()
            oov[original_w] = addressed_w

    with open(config['commongen']['addressed_vocab'], 'r', encoding="utf8") as f:
        for w in f.readlines():
            word = w.strip()
            commongen2id[word] = len(commongen2id)
            id2commongen[len(id2commongen)] = word
            
    return commongen2id, id2commongen, oov


def create_positionmatrix():
    commongen2id, _,oov = load_vocabs()
    position_matrix = np.zeros((len(commongen2id), len(commongen2id)), dtype=float)
    #Only read train
    with open(config['commongen']['subtrain_plan'], "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            vocabs = []
            words = line.split("\t")[0].strip().split(' ')
            for word in words:
                if word in oov.keys():
                    vocabs.append(oov[word])
                else:
                    vocabs.append(word)
            commongen_id = [commongen2id[c] for c in vocabs]


            for i in commongen_id:
                for j in commongen_id:
                    position_matrix[i][j] = 1
    print(np.count_nonzero(position_matrix ))
    print(np.nonzero(position_matrix[0]))
    np.savez(config['commongen']['subtrain_pos'], transition=position_matrix)




def topk_permutation(tran_matrix, k, cn_set):
    """Find the top-k permutation with the perdicted transition probability matrix of a given concept set

    Args:
        tran_matrix : The perdicted transition probability matrix between the concepts in the given concept set
        k : how many permutations we want to retirn
        cn_set (_type_): _description_

    Returns:
        the list of topk permuations
    """
    k = 1
    
    res = dict()
    concept_index = range(len(cn_set))
    
    for permut in itertools.permutations(concept_index):
        order = list(permut)
        temp_prob = 0
        for i in range(len(order)-1):
            temp_prob += tran_matrix[order[i]][order[i+1]]
        res[str(temp_prob)] = order
    #Rank all the permutations of a given set and take the first k permutations
    topk = sorted(res.items(), key=lambda item:float(item[0]),reverse=True)[:k]
    # Restore the index to the word
    
    dictdata = []
    for l in topk:
        dictdata.append([cn_set[i] for i in l[1]])
    return dictdata


def calculate_tau(labels, perdicted_topk):
    """Calulate the mean tau score with the ordering given in commongen and perdicted topk ordering

    Args:
        labels (List): True ordering
        perdicted_topk (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    """
    res = []
    for i in labels:
        temp_max = -1
        for j in perdicted_topk:
            temp,_ = stats.kendalltau(i,j)
            if temp > temp_max:
                temp_max = temp
        res.append(temp_max)
    return np.mean(res)
    """   
    temp_max = -1
    for i in labels:
        for j in perdicted_topk:
            temp,_ = stats.kendalltau(i,j)
            if temp > temp_max:
                temp_max = temp
    return temp_max
    

def predicted_matrix(matrix, commongen2id, cn_set):
    """Predict the transition matrix between concepts in the given conceptnet

    Args:
        model : the trained model
        commongen2id : change the commongen vocab to the corresponding row number 
        cn_set : a given concept set

    Returns:
        trans_matrix 
"""
    vocab_id = [commongen2id[c] for c in cn_set]
    #Generate the trans_matrix
    trans_matrix = np.zeros((len(vocab_id), len(vocab_id)))
    for i in range(len(vocab_id)):
        for j in range(len(vocab_id)):
            if i != j:
                trans_matrix[i][j] = matrix[vocab_id[i]][vocab_id[j]]
    return trans_matrix


def metric(matrix, plan):
    plan_orders = dict()
    with open(config['commongen'][plan],'r',encoding='utf-8') as f:
        for line in f.readlines():
            concepts = line.split('\t')[0].strip().split(' ')
            # Concept ordering saving in the dict
            concept_key = ' '.join(sorted(concepts))
            if concept_key not in plan_orders.keys():
                plan_orders[concept_key] = []
            if concepts not in plan_orders[concept_key]:
                plan_orders[concept_key].append(concepts)

    commongen2id, _, _ = load_vocabs()        
    #Initialize the score
    metric_avg = 0
    iteration = 0
    for temp in plan_orders.keys():
        #labels are the true ordering in the dev set
        labels = plan_orders[temp]
        cn_set = temp.split(" ")
        #Generate the transition matrix 
        trans_matrix = predicted_matrix(matrix, commongen2id, cn_set)
        #Generate the topk possibilityies 
        topk_order = topk_permutation(trans_matrix, len(labels), cn_set)
        #Calculate tau
        res = calculate_tau(labels, topk_order)
        iteration += 1
        metric_avg += res
    metric_print = metric_avg / iteration
    #print("concepts' average tau is: ", metric_print)
    return metric_print

