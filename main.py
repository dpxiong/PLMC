import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import torch.utils.data
import csv
from network import *
from metrics import *

def load_esm2(file, MAX_SEQ_LEN=1000):
    with open(file, 'rb') as infile:
        d = pickle.load(infile)[1:-1, :]
    if d.shape[0] < MAX_SEQ_LEN:
        tmp = np.zeros((MAX_SEQ_LEN-d.shape[0], d.shape[1]))
        d = np.concatenate((d, tmp))
    else:
        d = d[:MAX_SEQ_LEN, :]
        
    return d

def read_csv(file):
    save_list = []
    csv_reader = csv.reader(open(file))
    for row in csv_reader:
        row = map(float, row)
        row = list(row)
        save_list.append(row)
        
    return save_list

def read_fasta(file, esm2_path):
    with open(file,'r') as f:
        lines = f.readlines()
    seq_dict, protein_sequences, protein_names = {}, [], []
    for i in range(0, len(lines), 2):
        protein_name = lines[i].strip()[1:]
        protein_seq = lines[i+1].strip()
        
        protein_names.append(protein_name)
        protein_sequences.append(protein_seq)
        seq_dict[protein_name] = protein_seq
        
    return seq_dict, protein_sequences, protein_names

def load_data(args):
    rawdata_dir = args.rawpath
    test_fa, test_protein_sequences, test_names = read_fasta(rawdata_dir + '/' + 'TE_Sequence.fasta', rawdata_dir+'/TE_esm2_650m/')
    
    test_protein_feats1 = []
    for i in test_names:
        test_protein_feats1.append(load_esm2(rawdata_dir+'/TE_esm2_650m/'+i+'.pkl'))
    test_protein_feats1 = np.array(test_protein_feats1)
    
    test_protein_feats2 = read_csv(rawdata_dir + '/' + 'TE_feature.csv')
    test_protein_feats2 = np.array(test_protein_feats2)

    test_protein_labels = []
    test_label_file = rawdata_dir + '/' + 'TE_label'
    with open(test_label_file, 'r') as file_obj:
        for row in file_obj:
            test_protein_labels.append(int(row))
    test_protein_labels = np.array(test_protein_labels)

    
    train_fa, train_protein_sequences, train_names = read_fasta(rawdata_dir + '/' + 'TR_Sequence.fasta', rawdata_dir+'/TR_esm2_650m/')
    
    train_protein_feats1 = []
    for i in train_names:
        train_protein_feats1.append(load_esm2(rawdata_dir+'/TR_esm2_650m/'+i+'.pkl'))
        
    train_protein_feats2 = read_csv(rawdata_dir + '/' + 'TR_feature.csv')

    train_protein_labels = []
    train_label_file = rawdata_dir + '/' + 'TR_Label'
    with open(train_label_file, 'r') as file_obj:
        for row in file_obj:
            train_protein_labels.append(int(row))
    
    pos_ids = [i for i in range(len(train_protein_labels)) if train_protein_labels[i] == 1]
    neg_ids = [i for i in range(len(train_protein_labels)) if train_protein_labels[i] == 0]
    valid_pos_ids = sorted(random.sample(pos_ids, int(len(pos_ids)/5)))
    valid_neg_ids = sorted(random.sample(neg_ids, int(len(neg_ids)/5)))
    
    train_protein_feats1 = [train_protein_feats1[i] for i in range(len(train_protein_feats1)) if i not in valid_pos_ids+valid_neg_ids]
    train_protein_feats2 = [train_protein_feats2[i] for i in range(len(train_protein_feats2)) if i not in valid_pos_ids+valid_neg_ids]
    train_protein_labels = [train_protein_labels[i] for i in range(len(train_protein_labels)) if i not in valid_pos_ids+valid_neg_ids]
    
    valid_protein_feats1 = [train_protein_feats1[i] for i in range(len(train_protein_feats1)) if i in valid_pos_ids+valid_neg_ids]
    valid_protein_feats2 = [train_protein_feats2[i] for i in range(len(train_protein_feats2)) if i in valid_pos_ids+valid_neg_ids]
    valid_protein_labels = [train_protein_labels[i] for i in range(len(train_protein_labels)) if i in valid_pos_ids+valid_neg_ids]
    
    train_protein_feats1 = np.array(train_protein_feats1)
    train_protein_feats2 = np.array(train_protein_feats2)
    train_protein_labels = np.array(train_protein_labels)
    
    valid_protein_feats1 = np.array(valid_protein_feats1)
    valid_protein_feats2 = np.array(valid_protein_feats2)
    valid_protein_labels = np.array(valid_protein_labels)
    
    return train_protein_feats1, train_protein_feats2, train_protein_labels, valid_protein_feats1, valid_protein_feats2, valid_protein_labels, test_protein_feats1, test_protein_feats2, test_protein_labels

def train_epoch(dataset_tensor, model, optimizer, lossfunction, device):
    model.train()
    
    loss_ = 0.0
    for index, batch_feats1, batch_feats2, batch_labels in dataset_tensor:
        optimizer.zero_grad()
        
        batch_feats1, batch_feats2, batch_labels = batch_feats1.to(device), batch_feats2.to(device), batch_labels.to(device)

        scores = model(batch_feats1, batch_feats2, device)
        loss = lossfunction(scores, batch_labels)

        loss.backward(retain_graph=True)
        optimizer.step()

        loss_ += loss.item()

    return loss_

def evaluate_epoch(dataset_tensor, model, lossfunction, device):
    model.eval()
    
    loss_ = 0.0
    pred_probs = []
    labels = []
    for index, batch_feats1, batch_feats2, batch_labels in dataset_tensor:

        batch_feats1, batch_feats2, batch_labels = batch_feats1.to(device), batch_feats2.to(device), batch_labels.to(device)
        scores = model(batch_feats1, batch_feats2, device)
        loss = lossfunction(scores, batch_labels)
        
        loss_ += loss.item()
        
        pred_probs.append(list(scores.data.cpu().numpy()))
        labels.append(list(batch_labels.data.cpu().numpy()))
        
    pred_probs = np.concatenate(pred_probs)
    labels = np.concatenate(labels)

    return loss_, pred_probs, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--lr', type = float, default = 0.002,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--droprate', type = float, default=0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default=32,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--num_heads', type = int, default=8,
                        metavar = 'N', help = 'number of heads')
    parser.add_argument('--epochs', type = int, default=200,
                        metavar = 'N', help = 'epochs for training')
    parser.add_argument('--rawpath', type=str, default='../data/CRYS_DS',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('Hyper-parameters')
    print('learning rate: ' + str(args.lr))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('dimension of embedding: ' + str(args.embed_dim))
    print('batchsize: ' + str(args.batch_size))
    print('num_heads: ' + str(args.num_heads))
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = PLMC(256, args.embed_dim, args.num_heads, args.droprate).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_protein_feats1, train_protein_feats2, train_protein_labels, valid_protein_feats1, valid_protein_feats2, valid_protein_labels, test_protein_feats1, test_protein_feats2, test_protein_labels = load_data(args)
    
    
    indices = np.zeros((len(train_protein_feats1), 1))
    for i in range(indices.shape[0]):
        indices[i] = i
    indices = np.array(indices)
    train_dataset_tensor = torch.utils.data.TensorDataset(torch.LongTensor(indices), 
                                                          torch.FloatTensor(train_protein_feats1), 
                                                          torch.FloatTensor(train_protein_feats2), 
                                                          torch.LongTensor(train_protein_labels))
    train_dataset_tensor = torch.utils.data.DataLoader(train_dataset_tensor, batch_size=args.batch_size, 
                                                       shuffle=True, num_workers=0, pin_memory=True)
    
    indices = np.zeros((len(valid_protein_feats1), 1))
    for i in range(indices.shape[0]):
        indices[i] = i
    indices = np.array(indices)
    valid_dataset_tensor = torch.utils.data.TensorDataset(torch.LongTensor(indices), 
                                                          torch.FloatTensor(valid_protein_feats1), 
                                                          torch.FloatTensor(valid_protein_feats2), 
                                                          torch.LongTensor(valid_protein_labels))
    valid_dataset_tensor = torch.utils.data.DataLoader(valid_dataset_tensor, batch_size=args.batch_size, 
                                                       shuffle=True, num_workers=0, pin_memory=True)
    
    indices = np.zeros((len(test_protein_feats1), 1))
    for i in range(indices.shape[0]):
        indices[i] = i
    indices = np.array(indices)
    test_dataset_tensor = torch.utils.data.TensorDataset(torch.LongTensor(indices), 
                                                         torch.FloatTensor(test_protein_feats1),
                                                         torch.FloatTensor(test_protein_feats2), 
                                                         torch.LongTensor(test_protein_labels))
    test_dataset_tensor = torch.utils.data.DataLoader(test_dataset_tensor, batch_size=args.batch_size, 
                                                       shuffle=True, num_workers=0, pin_memory=True)
    
    for epoch in range(1, args.epochs + 1):
        print('Epoch %d' % epoch)
        train_loss = train_epoch(train_dataset_tensor, model, optimizer, criterion, device)
        
        train_loss, train_probs, train_labels = evaluate_epoch(train_dataset_tensor, model, criterion, device)
        aupr = compute_aupr(train_labels, train_probs[:, 1])
        auc = compute_roc(train_labels, train_probs[:, 1])
        print('train: aupr:%0.6f, auc:%0.6f' % (aupr, auc))
        
        
        valid_loss, valid_probs, valid_labels = evaluate_epoch(valid_dataset_tensor, model, criterion, device)
        aupr = compute_aupr(valid_labels, valid_probs[:, 1])
        auc = compute_roc(valid_labels, valid_probs[:, 1])
        p_max, sn_max, sp_max, mcc_max, acc_max, t_max, predictions_max = compute_performance_max(valid_labels, valid_probs[:, 1])
        print('validation: aupr:%0.6f, auc:%0.6f, cutoff:%0.6f, mcc:%0.6f, p:%0.6f, sn:%0.6f, sp:%0.6f, acc:%0.6f' % (aupr, auc, t_max, mcc_max, p_max, sn_max, sp_max, acc_max))
        
        
        test_loss, test_probs, test_labels = evaluate_epoch(test_dataset_tensor, model, criterion, device)
        aupr = compute_aupr(test_labels, test_probs[:, 1])
        auc = compute_roc(test_labels, test_probs[:, 1])
        p, sn, sp, mcc, acc, predictions = compute_performance(test_labels, test_probs[:, 1], t_max)
        print('test: aupr:%0.6f, auc:%0.6f, cutoff:%0.6f, mcc:%0.6f, p:%0.6f, sn:%0.6f, sp:%0.6f, acc:%0.6f' % (aupr, auc, t_max, mcc, p, sn, sp, acc))
        
        print('\n')
        