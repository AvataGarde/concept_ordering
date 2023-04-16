import torch
import torch.utils.data as tud
import numpy as np
import time
import configparser
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import pylab as pl

import os

os.chdir(os.path.dirname(__file__))

from model import TransitionDataset, TransitionModel


EMBEDDING_SIZE = 300
MAX_VOCAB_SIZE = 4913

NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.0003

save_every = 10000

config = configparser.ConfigParser()
config.read("path.ini")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_trans_matrix(dataset_type):
    if dataset_type == 'train':
        files = {
            'M_g': config['commongen']['m1'],
            'M_c': config['commongen']['m2'],
            'M_p': config['commongen']['train_pos'],
        }
    elif dataset_type == 'eval':
        files = {
            'M_g': config['commongen']['eval'],
            'M_c': config['commongen']['m2'],
            'M_p': config['commongen']['eval_pos'],
        }
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Must be either 'train' or 'eval'.")

    M_g = torch.tensor(np.load(files['M_g'])['transition'], dtype=torch.float)
    M_c = torch.tensor(np.load(files['M_c'])['transition'], dtype=torch.float)
    M_p = torch.tensor(np.load(files['M_p'])['transition'], dtype=torch.float)
    return M_g, M_c, M_p


def load_init_embedding():
    fasttext_embed = torch.load(config['commongen']['fasttext_embed'], map_location=torch.device('cpu'))
    return fasttext_embed


def load_model():
    model_path = config['model']['model_path']
    pretrained_emb = load_init_embedding()
    model = TransitionModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE, pretrained_emb)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def train(model, train_loader, eval_loader, optimizer, scheduler):
    train_loss_history = []
    eval_loss_history = []
    eval_metrics_history = []
    best_eval_loss = float('inf')
    best_h_matrix = None
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.
        for i, (i_batch, j_batch, Mg_p_batch, Mc_p_batch) in enumerate(train_loader):
            i_batch = i_batch.to(device)
            j_batch = j_batch.to(device)
            Mg_p_batch = Mg_p_batch.to(device)
            Mc_p_batch = Mc_p_batch.to(device)
            
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass and compute loss
            output = model(i_batch, j_batch)
            loss = model.loss_func(output, Mg_p_batch, Mc_p_batch)

            # Backward pass and update model parameters
            loss.backward()
            optimizer.step()

            # Record the training loss
            train_loss += loss.item()
        
        # Update the learning rate scheduler
        scheduler.step()
        
        # Validation phase
        model.eval()
        eval_loss = 0.
        with torch.no_grad():
            for i, (i_batch, j_batch, Mg_p_batch, Mc_p_batch) in enumerate(eval_loader):
                i_batch = i_batch.to(device)
                j_batch = j_batch.to(device)
                Mg_p_batch = Mg_p_batch.to(device)
                Mc_p_batch = Mc_p_batch.to(device)

                # Forward pass and compute loss
                output = model(i_batch, j_batch)
                loss = model.loss_func(output, Mg_p_batch, Mc_p_batch)

                # Record the validation loss
                eval_loss += loss.item()
            
            # Generate matrix and compute the metric score
            h_matrix = model.get_matrix()
            eval_metrics = metric(h_matrix, "eval_plan")
            
            # Record the best h_matrix and metric score
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_h_matrix = h_matrix
                best_eval_metrics = eval_metrics
                torch.save(model.state_dict(), config['model']['model_path'])
                print('Saved the best model.')
                print('Current metric score: %.4f' % eval_metrics)
            
        # Record the training and validation losses
        train_loss_history.append(train_loss / len(train_loader))
        eval_loss_history.append(eval_loss / len(eval_loader))
        eval_metrics_history.append(eval_metrics)
        
        # Print the epoch summary
        print('Epoch %d: Train loss = %.4f, Eval loss = %.4f' % (
            epoch + 1, train_loss_history[-1], eval_loss_history[-1]))
    
    return train_loss_history, eval_loss_history, eval_metrics_history, best_eval_metrics, best_h_matrix



if __name__ == '__main__':
    train_plan, train_cn, train_pos = load_trans_matrix("train")
    train_dataset = TransitionDataset(train_plan, train_cn, train_pos)
    train_dataloader = tud.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    eval_plan, eval_cn, eval_pos = load_trans_matrix("eval")
    eval_dataset = TransitionDataset(eval_plan, eval_cn, eval_pos)
    eval_dataloader = tud.DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    init_embedding = load_init_embedding()
    
    
    model = TransitionModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE, init_embedding)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Train the model
    # Train the model and record the best h_matrix and metric score
    train_loss_history, eval_loss_history, eval_metrics_history,best_eval_metrics, best_h_matrix = train(model, train_dataloader, eval_dataloader, optimizer, scheduler)

    # Print the final metric score
    print('Final best metric score: %.4f' % best_eval_metrics)
    # Plot the training and validation losses
    plt.plot(train_loss_history, label='train')
    plt.plot(eval_loss_history, label='eval')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plot the metric scores
    plt.plot(eval_metrics_history)
    plt.title('Metric Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.show()
    # Save the best h_matrix to file
    np.savez(config['model']['output_path'], transition=best_h_matrix)
    




        
    
    







