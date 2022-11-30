import torch
print(torch.cuda.is_available())
import configparser
import itertools
import os
import numpy as np
from scipy import stats

os.chdir(os.path.dirname(__file__))
config = configparser.ConfigParser()
config.read("path.ini")

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
    print("commongen2id done")
    return commongen2id, id2commongen, oov



def topk_permutation(tran_matrix, k, cn_set):
    """Find the top-k permutation with the perdicted transition probability matrix of a given concept set

    Args:
        tran_matrix : The perdicted transition probability matrix between the concepts in the given concept set
        k : how many permutations we want to retirn
        cn_set (_type_): _description_

    Returns:
        the list of topk permuations
    """    
    res = dict()
    concept_index = range(len(cn_set))
    
    for permut in itertools.permutations(concept_index):
        order = list(permut)
        temp_prob = 1
        for i in range(len(order)-1):
            temp_prob *= tran_matrix[order[i]][order[i+1]]
        res[str(temp_prob)] = order
    #Rank all the permutations of a given set and take the first k permutations
    topk = sorted(res.items(), key=lambda item:item[0],reverse=True)[:k]
    
    print(topk)
    # Restore the index to the word
    dictdata = []
    for l in topk:
        dictdata.append([cn_set[i] for i in l[1]])
    return dictdata


def calculate_tau(labels, perdicted_topk):
    """Calulate the mean tau score with the ordering given in commongen and perdicted topk ordering

    Args:
        labels (List): True 
        perdicted_topk (_type_): _description_

    Returns:
        _type_: _description_
    """    
    res = []
    for i in labels:
        temp_max = -1
        for j in perdicted_topk:
            temp,_ = stats.kendalltau(i,j)
            if temp > temp_max:
                temp_max = temp
        res.append(temp_max)
    print(res)
    return np.mean(res)
      

plan_orders = dict()
with open(config['commongen']['eval_plan'],'r',encoding='utf-8') as f:
    for line in f.readlines():
        concepts = line.split('\t')[0].strip().split(' ')
        concept_key = ' '.join(sorted(concepts))
        if concept_key not in plan_orders.keys():
            plan_orders[concept_key] = []
        if concepts not in plan_orders[concept_key]:
            plan_orders[concept_key].append(concepts)
            


temp = 'field look stand'
labels = plan_orders[temp]
cn_set = temp.split(' ')
print(labels)
trans_matrix = np.random.randint(0,10,size=[3,3])
print(trans_matrix)

topk_order = topk_permutation(trans_matrix, len(labels), cn_set)
print("Predicted topk order is:",topk_order)
print('-------------------')
res = calculate_tau(labels, topk_order)
print(res)


  