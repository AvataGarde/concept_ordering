import torch
import torch.utils.data as tud 
import torch.nn.functional as F 

from scipy import stats
import numpy as np
import configparser
from tqdm import tqdm
import itertools
import time
import math
import os
os.chdir(os.path.dirname(__file__))

from model import transitionDataset, transitionModel


EMBEDDING_SIZE = 1324
MAX_VOCAB_SIZE = 4913

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

print_every = 1000
save_every = 10000


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



def load_trans_matrix(if_train):
    
    M_c = np.load(config['commongen']['m21'])["transition"]
    M_g = None
    if if_train:
        M_g = np.load(config['commongen']['m1'])["transition"]
    else:
        M_g = np.load(config['commongen']['eval'])["transition"]
        
    return M_g, M_c


def load_init_embedding():
    bert_embed = torch.load(config['commongen']["bert_embed"])
    fasttext_embed = torch.load(config['commongen']["fasttext_embed"])
    concated_embed = torch.cat((bert_embed,fasttext_embed), dim=1).cuda()
    return concated_embed


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


def load_model():
    path =config['model']['model_path']
    pretrained_emb = load_init_embedding()
    model = transitionModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE,pretrained_emb)
    model.load_state_dict(torch.load(path))
    return model


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
        #For each true order, select the predicted order with the highest tau score 
        temp_max = -1
        for j in perdicted_topk:
            temp,_ = stats.kendalltau(i,j)
            if temp > temp_max:
                temp_max = temp
        res.append(temp_max)
    #Reture the mean of tau score
    return np.mean(res)
    
    
def predicted_matrix(model, commongen2id, cn_set):
    """Predict the transition matrix between concepts in the given conceptnet

    Args:
        model : the trained model
        commongen2id : change the commongen vocab to the corresponding row number 
        cn_set : a given concept set

    Returns:
        _type_: _description_
    """    
    #Load the model trained embeddings
    res_v, res_w, res_bv,res_bw = model.get_matrix()
    vocab_id = [commongen2id[c] for c in cn_set]
    #Generate the trans_matrix
    trans_matrix = np.zeros((len(vocab_id), len(vocab_id)))
    for i in range(len(vocab_id)):
        for j in range(len(vocab_id)):
            if i != j:
                temp_prob = torch.dot(res_v[vocab_id[i]], res_w[vocab_id[j]])/math.sqrt(model.embed_size)
                temp_prob = torch.sigmoid(temp_prob + res_bv[vocab_id[i]] + res_bw[vocab_id[j]])
                trans_matrix[i][j] = temp_prob
                #print(cn_set[i],cn_set[j], temp_prob, cn_matrix[vocab_id[i]][vocab_id[j]], plan_matrix[vocab_id[i]][vocab_id[j]])
    return trans_matrix

    
def train_part():
    plan_matrix, cn_matrix = load_trans_matrix(if_train=True)
    dataset = transitionDataset(plan_matrix,cn_matrix, if_train=True)
    dataloader = tud.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    init_embedding = load_init_embedding()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = transitionModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE, init_embedding)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    epochs = NUM_EPOCHS

    iters_per_epoch = int(dataset.__len__() / BATCH_SIZE)
    total_iterations = iters_per_epoch * epochs
    print("Iterations: %d per one epoch, Total iterations: %d " % (iters_per_epoch, total_iterations))

    start = time.time()
    for epoch in tqdm(range(epochs)):
        print("Iteration: ", epoch)
        loss_print_avg = 0
        iteration = iters_per_epoch * epoch
        
        for i, j, plan, cn in dataloader:
            i, j, plan, cn = i.cuda(), j.cuda(), plan.cuda(), cn.cuda()
            iteration += 1
            
            model.train()
            optimizer.zero_grad()
            
            predictions = model(i, j)
            
            loss = model.loss_func(predictions, plan, cn)
                    
            loss.backward()
            optimizer.step()
            
            loss_print_avg += loss.item()

            if iteration % print_every == 0:
                time_desc = timeSince(start, iteration / total_iterations)
                iter_percent = iteration / total_iterations * 100
                loss_avg = loss_print_avg / print_every
                loss_print_avg = 0
                print("epoch: %d, iter: %d/%d (%.4f%%), loss: %.5f, %s" %
                      (epoch, iteration, total_iterations, iter_percent, loss_avg, time_desc))

            if iteration % save_every == 0:
                valid_part(model)
                metric(model,test_plan='eval_plan')
                #torch.save(model.state_dict(), config['model']['model_path'])
        
        
    torch.save(model.state_dict(), config['model']['model_path'])


def valid_part(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    plan_matrix, cn_matrix = load_trans_matrix(if_train=False)
    dataset = transitionDataset(plan_matrix,cn_matrix, if_train=False)
    dataloader = tud.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    loss_print_avg = 0
    for i, j, plan, cn in dataloader:
        i, j, plan, cn = i.cuda(), j.cuda(), plan.cuda(), cn.cuda()
        model.eval()
        predictions = model(i, j)
        loss = model.valid_loss(predictions, plan)
        
        loss_print_avg += loss.item()
        
    loss_avg = loss_print_avg / int(dataset.__len__())
    print("The loss of evaluation is: ", loss_avg)
        


def metric(model=None, test_plan="eval_plan"):
    """Evaluate the model with the designed tau score

    Args:
        model (_type_, optional): _description_. Defaults to None.
        test_plan (str, optional): _description_. Defaults to "eval_plan".
    """
    if not model:
        model = load_model()
    commongen2id, id2commongen, oov = load_vocabs()
    #read the correct order
    plan_orders = dict()
    with open(config['commongen'][test_plan],'r',encoding='utf-8') as f:
        for line in f.readlines():
            concepts = line.split('\t')[0].strip().split(' ')
            # Concept ordering saving in the dict
            concept_key = ' '.join(sorted(concepts))
            if concept_key not in plan_orders.keys():
                plan_orders[concept_key] = []
            if concepts not in plan_orders[concept_key]:
                plan_orders[concept_key].append(concepts)
    #Initialize the score
    metric_avg = 0
    iteration = 0
    for temp in tqdm(plan_orders.keys()):
        #labels are the true ordering in the dev set
        labels = plan_orders[temp]
        cn_set = temp.split(" ")
        #Generate the transition matrix 
        trans_matrix = predicted_matrix(model, commongen2id, cn_set)
        #Generate the topk possibilityies 
        topk_order = topk_permutation(trans_matrix, len(labels), cn_set)
        #Calculate tau
        res = calculate_tau(labels, topk_order)
        #print(res)
        iteration += 1
        metric_avg += res
    metric_print = metric_avg / iteration
    print("concepts' average tau is: ", metric_print)
        
    
    
    
if __name__ == '__main__':
    train_part()






