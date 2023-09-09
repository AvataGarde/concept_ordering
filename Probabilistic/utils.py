import configparser
import json
from tqdm import tqdm
import numpy as np
import itertools
import time
import numpy as np
from scipy import stats
import json
import random
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import math
import os
from aligner import Aligner


aligner = Aligner()
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


def create_positionmatrix(plan, pos):
    commongen2id, _,oov = load_vocabs()
    position_matrix = np.zeros((len(commongen2id), len(commongen2id)), dtype=float)
    with open(config['commongen'][plan], "r", encoding="utf8") as f:
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
    np.savez(config['commongen'][pos], transition=position_matrix)




def topk_permutation(tran_matrix, k, cn_set):
    """Find the top-k permutation with the perdicted transition weight matrix of a given concept set

    Args:
        tran_matrix : The perdicted transition weight matrix between the concepts in the given concept set
        k : how many permutations we want to retirn
        cn_set (_type_): _description_

    Returns:
        the list of topk permuations
    """
    
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
    """Calulate the max tau score with the ordering given in commongen and perdicted topk ordering

    Args:
        labels (List): True ordering
        perdicted_topk (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    return max(
        stats.kendalltau(i, j)[0] 
        for i in labels 
        for j in perdicted_topk
    )

    

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
        topk_order = topk_permutation(trans_matrix, 1, cn_set)
        #Calculate tau
        res = calculate_tau(labels, topk_order)
        iteration += 1
        metric_avg += res
    metric_print = metric_avg / iteration
    print("concepts' average tau is: ", metric_print)
    return metric_print


def metric_from_files(file, plan):
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
             
                
    predict_orders = dict()
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            concepts = line.split('\t')[0].strip().split(' ')
            concept_key = ' '.join(sorted(concepts))
            if concept_key not in predict_orders.keys():
                predict_orders[concept_key] = []     
                 #We only evaluate the first ordering following other generation metrics.
                predict_orders[concept_key].append(concepts)
    
    
    #Initialize the score
    metric_avg = 0
    iteration = 0
    for temp in plan_orders.keys():
        labels = plan_orders[temp]
        top_orders = predict_orders[temp]
        res = calculate_tau(labels, top_orders)
        iteration += 1
        metric_avg += res
    metric_print = metric_avg / iteration
    print("concepts' average tau is: ", metric_print)
    return metric_print
        



def generate_order(matrix, src_file,res_file):
    commongen2id, id2commongen, oov = load_vocabs()
    res = []
    previous_set = None
    count =0
    with open(src_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            concepts = line.strip().split(' ')
            for i in range(len(concepts)):
                if concepts[i] in oov.keys():
                    concepts[i] = oov[concepts[i]]
            if concepts != previous_set:
                if previous_set is not None:
                    #print(previous_set, count)
                    trans_matrix = predicted_matrix(matrix, commongen2id, previous_set)
                    topk_order = topk_permutation(trans_matrix, count, previous_set)
                    
                    #print(topk_order)
                    res.extend(topk_order)
                count = 0
                previous_set = concepts
            count +=1

        if previous_set is not None:
            #print(previous_set, count)
            trans_matrix = predicted_matrix(matrix, commongen2id, previous_set)
            topk_order = topk_permutation(trans_matrix, count, previous_set)
            res.extend(topk_order)
    print(len(res))
    with open(res_file,'w', encoding="utf-8") as f:
        for line in res:
            f.write(" ".join(line)+"\n")



def create_dataset(alpha_file, order_file, res_file):
    res = []
    with open(alpha_file,'r',encoding='utf-8') as f1, open(order_file,'r',encoding='utf-8') as f2:
        for alpha_line,orde_line in zip(f1.readlines(),f2.readlines()):
            res.append(alpha_line.strip("\n")+' [ORDERING] '+orde_line.strip("\n")+' [ORDERING]')
    
    with open(res_file,'w',encoding='utf-8') as f:
        for line in res:
            f.write(line+"\n")


def get_file_len(file):
    with open(file, "r") as f:
        return len(f.readlines())


def convert_realization_to_plan(src_file, tgt_file, plan_file):
    oov = dict()

    with open(config['commongen']["oov"], "r", encoding="utf8") as f:
        for line in f.readlines():
            original_w, addressed_w = line.split("\t")[0].strip(), line.split("\t")[1].strip()
            oov[original_w] = addressed_w
            
    with open(src_file, 'r') as fr1, open(tgt_file, 'r') as fr2, open(plan_file, 'w') as fw:
        for line_src, line_tgt in tqdm(zip(fr1.readlines(), fr2.readlines()), total=get_file_len(src_file)):
            concepts = line_src.strip().split()
            for i in range(len(concepts)):
                if concepts[i] in oov.keys():
                    concepts[i] = oov[concepts[i]]
            sentence = line_tgt.strip()
            plan, sentence, plan_idx, _ = aligner.align(concepts, sentence, multi=False, distance=1)
            plan = plan.split()
            fw.write(' '.join(plan) + '\n')


def create_dataset2(alpha_file, tgt_file, res_file):
    with open(alpha_file, 'r') as fr1, open(tgt_file, 'r') as fr2, open(res_file, 'w') as fw:
        for line_src, line_tgt in tqdm(zip(fr1.readlines(), fr2.readlines()), total=get_file_len(alpha_file)):
            concepts = line_src.strip().split()
            sentence = line_tgt.strip()
            plan, sentences, plan_idx, _ = aligner.align(concepts, sentence, multi=False, distance=1)
            plan = plan.split()
            formats = [sentences.split(' ')[idx] for idx in plan_idx]
            res = line_src.strip() +' [ORDERING] '+' '.join(formats)+' [ORDERING]'
            #print(res)
            fw.write(res + '\n')


def shuffle_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            words = line.strip().split()
            random.shuffle(words)
            shuffled_line = ' '.join(words) + '\n'
            f_out.write(shuffled_line)


#实验性看看matrix能否help
def predict_order(matrix,src_file,pred_file,res_file):
    plan_orders = dict()
    with open(pred_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            concepts = line.split('\t')[0].strip().split(' ')
            concept_key = ' '.join(sorted(concepts))
            if concept_key not in plan_orders.keys():
                plan_orders[concept_key] = []
            if concepts not in plan_orders[concept_key]:
                plan_orders[concept_key].append(concepts)
    commongen2id, _, _ = load_vocabs()
    
    res = []
    previous_set = None
    count =0
    with open(src_file,'r',encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            concepts = line.strip().split(' ')
            if concepts != previous_set:
                if previous_set is not None:
                    previous_list = sorted(previous_set)
                    preds = plan_orders[" ".join(previous_list)]
                    trans_matrix = predicted_matrix(matrix, commongen2id, previous_list)
                    topk_order = topk_permutation(trans_matrix, math.factorial(len(previous_list)), previous_list)
                    min_index = float('inf')
                    for pred in preds:
                        index = topk_order.index(pred)
                        if index < min_index:
                            min_index = index
                    res.extend([topk_order[min_index]]*count)
                count = 0
                previous_set = concepts
            count +=1

        if previous_set is not None:
            previous_list = sorted(previous_set)
            preds = plan_orders[" ".join(previous_list)]
            trans_matrix = predicted_matrix(matrix, commongen2id, previous_list)
            topk_order = topk_permutation(trans_matrix, math.factorial(len(previous_list)), previous_list)
            min_index = float('-inf')
            for pred in preds:
                index = topk_order.index(pred)
                if index < min_index:
                    min_index = index
            res.extend([topk_order[min_index]]*count)
    with open(res_file,'w', encoding="utf-8") as f:
        for line in res:
            f.write(" ".join(line)+"\n")




if __name__ == '__main__':
    # Generate the probavilistic ordering from the trained matrix
    """
    M_o = np.load("../models/ordered_ordering/NN/res_matrix_4266.npz")["transition"]
    row_sums = np.abs(M_o).sum(axis=1)
    normalized_matrix = M_o / row_sums[:, np.newaxis]
    src_file = "../../commongen/dataset/commongen.dev.src_alpha.txt"
    generate_order(normalized_matrix, src_file, res_file="commongen.dev.src_matrix.txt")
    """
    #Evaluate the Kendall's tau given a concept ordering file
    """
    metric_from_files("commongen.train.src_matrix.txt","train_plan")
    """
