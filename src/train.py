import torch
import torch.utils.data as tud 
import numpy as np
import time
import configparser
from tqdm import tqdm
from utils import *


import os
#os.chdir(os.path.dirname(__file__))

from model import transitionDataset, transitionModel


EMBEDDING_SIZE = 300
MAX_VOCAB_SIZE = 4913

NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.0005

print_every = 1000
save_every = 10000


config = configparser.ConfigParser()
config.read("path.ini")


def load_trans_matrix(if_train):
    
    M_c = np.load(config['commongen']['m2'])["transition"]
    M_p = None
    M_g = None
    if if_train:
        M_g = np.load(config['commongen']['m1'])["transition"]
        M_p = np.load(config['commongen']['train_pos'])["transition"]
    else:
        M_g = np.load(config['commongen']['eval'])["transition"]
        M_p = np.load(config['commongen']['eval_pos'])["transition"]
        
    return M_g, M_c, M_p


def load_init_embedding():
    bert_embed = torch.load(config['commongen']["bert_embed"])
    fasttext_embed = torch.load(config['commongen']["fasttext_embed"])
    concated_embed = torch.cat((bert_embed,fasttext_embed), dim=1).cuda()
    return fasttext_embed   


def load_model():
    path =config['model']['model_path']
    pretrained_emb = load_init_embedding()
    model = transitionModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE,pretrained_emb)
    model.load_state_dict(torch.load(path))
    return model

    
def train_part():
    plan_matrix, cn_matrix, pos_matrix = load_trans_matrix(if_train=True)
    dataset = transitionDataset(plan_matrix,cn_matrix, pos_matrix, if_train=True)
    dataloader = tud.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    eval_plan, eval_cn, eval_pos = load_trans_matrix(if_train=False)
    eval_dataset = transitionDataset(eval_plan, eval_cn, eval_pos, if_train=False)
    eval_dataloader = tud.DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    
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
    min_valid_loss = 10
    max_tau_score = -1
    for epoch in tqdm(range(epochs)):
        print("Iteration: ", epoch)
        loss_print_avg = 0
        iteration = iters_per_epoch * epoch
        
        for i, j, m_g, m_c in dataloader:
            i, j, m_g, m_c = i.cuda(), j.cuda(), m_g.cuda(), m_c.cuda()
            iteration += 1
            
            model.train()
            optimizer.zero_grad()
            
            predictions = model(i, j)
            loss = model.loss_func(predictions, m_g, m_c)
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
           
                
                h_matrix = model.get_matrix()
                temp_tau = metric(h_matrix,"eval_plan")
                #print("concepts' average tau is: ", temp_tau)
                if temp_tau > max_tau_score:
                    max_tau_score = temp_tau
                    print("concepts' average tau is: ", temp_tau)
                    temp_loss = valid_part(model,eval_dataset, eval_dataloader)
                    print("The loss of evaluation is: ", temp_loss)
                

    print("The min loss of evaluation is: ", min_valid_loss)
    print("The max tau of evaluation is:", max_tau_score)      
        
        

def valid_part(model, dataset, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_print_avg = 0
    for i, j, m_g, m_c in dataloader:
        i, j, m_g, m_c = i.cuda(), j.cuda(), m_g.cuda(), m_c.cuda()
        model.eval()
        predictions = model(i, j)
        loss = model.valid_loss(predictions, m_g)
        loss_print_avg += loss.item()
        
    loss_avg = loss_print_avg / int(dataset.__len__())
    return loss_avg
        



        
    
    
if __name__ == '__main__':
    train_part()






