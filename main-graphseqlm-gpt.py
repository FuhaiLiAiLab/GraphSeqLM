import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom modules
from texttable import Texttable
from models.pretrain_gnn.pretrain_gnn_model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from models.finetune_llm.model_gpt import DNAGPT_LM, RNAGPT_LM, ProtGPT_LM
from models.graphseqlm.graphseqlm import GraphSeqLM

# custom dataloader
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to print the logs in a nice tabular format.
def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()

def write_best_model_info(fold_n, path, max_test_acc_id, epoch_loss_list, epoch_acc_list, test_loss_list, test_acc_list):
    best_model_info = (
        f'\n-------------Fold: {fold_n} -------------\n'
        f'\n-------------BEST TEST ACCURACY MODEL ID INFO: {max_test_acc_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {epoch_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TRAIN ACCURACY: {epoch_acc_list[max_test_acc_id - 1]}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {test_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TEST ACCURACY: {test_acc_list[max_test_acc_id - 1]}\n'
    )
    with open(os.path.join(path, 'best_model_info.txt'), 'w') as file:
        file.write(best_model_info)

def build_pretrain_model(args, num_feature, num_node, device):
    encoder = GNNEncoder(num_feature, args.encoder_channels, args.hidden_channels,
                        num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                        bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    internal_encoder = GNNEncoder(num_feature, args.input_dim, args.input_dim,
                            num_layers=args.internal_encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    pretrain_model = MaskGAE(input_dim=args.input_dim, 
                    num_node=num_node,
                    encoder=encoder, 
                    internal_encoder=internal_encoder,
                    edge_decoder=edge_decoder, 
                    degree_decoder=degree_decoder, 
                    edge_p=args.edge_p,
                    node_p=args.node_p).to(device)
    return pretrain_model

def build_finetune_model(device):
    # Create an instance of DNAGPT_LM
    dna_gpt_lm = DNAGPT_LM(model_path="./models/finetune_llm/checkpoints", model_name="dna_gpt0.1b_h", device=device)
    rna_gpt_lm = RNAGPT_LM(model_path="./models/finetune_llm/checkpoints", model_name="dna_gpt0.1b_h", device=device)
    prot_gpt_lm = ProtGPT_LM(model_name="nferruz/ProtGPT2", device=device)
    return dna_gpt_lm, rna_gpt_lm, prot_gpt_lm

def build_graphclas_model(args, num_node, device):
    model = GraphSeqLM(input_dim=args.train_input_dim, 
                       hidden_dim=args.train_hidden_dim, 
                       embedding_dim=args.train_embedding_dim, 
                       lm_dim=args.lm_dim,
                       num_nodes=num_node, 
                       num_heads=args.num_heads,
                       num_classes=args.num_classes,
                       device=device).to(device)
    return model

def language_model_embedding(num_type_node, dna_gpt_lm, rna_gpt_lm, prot_gpt_lm, seq):
    seq = seq.tolist()
    # DNA sequence embedding
    dna_gpt_lm.load_model()
    dna_seq = seq[:int(num_type_node)]
    dna_embedding = dna_gpt_lm.generate_embeddings(dna_seq)
    # RNA sequence embedding
    rna_gpt_lm.load_model()
    rna_seq = seq[int(num_type_node):2*int(num_type_node)]
    rna_embedding = rna_gpt_lm.generate_embeddings(rna_seq)
    # Protein sequence embedding
    prot_gpt_lm.load_model()
    protein_seq = seq[2*int(num_type_node):]
    protein_embedding = prot_gpt_lm.generate_embeddings(protein_seq)
    return dna_embedding, rna_embedding, protein_embedding

def train_graphclas_model(train_dataset_loader, dna_embedding, rna_embedding, protein_embedding, pretrain_model, model, device, args):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.train_lr, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.9)
    batch_loss = 0
    for batch_idx, data in enumerate(train_dataset_loader):
        optimizer.zero_grad()
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index)
        embedding = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        # Get the language model embedding
        dna_embedding = Variable(torch.Tensor(dna_embedding), requires_grad=False).to(device)
        rna_embedding = Variable(torch.Tensor(rna_embedding), requires_grad=False).to(device)
        protein_embedding = Variable(torch.Tensor(protein_embedding), requires_grad=False).to(device)
        # Use graphseqlm model to get the output
        x = x + torch.normal(mean=0, std=0.01, size=x.size()).to(device) # Add noise to x
        output, ypred = model(x, embedding, edge_index, dna_embedding, rna_embedding, protein_embedding)
        loss = model.loss(output, label)
        loss.backward()
        batch_loss += loss.item()
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step(loss)
        # # check pretrain model parameters
        # state_dict = pretrain_model.internal_encoder.state_dict()
        # print(state_dict['convs.1.lin.weight'])
        # print(model.embedding.weight.data)
    torch.cuda.empty_cache()
    return model, batch_loss, batch_acc, ypred


def test_graphclas_model(test_dataset_loader, dna_embedding, rna_embedding, protein_embedding, pretrain_model, model, device):
    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(test_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index)
        embedding = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        # Get the language model embedding
        dna_embedding = Variable(torch.Tensor(dna_embedding), requires_grad=False).to(device)
        rna_embedding = Variable(torch.Tensor(rna_embedding), requires_grad=False).to(device)
        protein_embedding = Variable(torch.Tensor(protein_embedding), requires_grad=False).to(device)
        # Use graphseqlm model to get the output
        output, ypred = model(x, embedding, edge_index, dna_embedding, rna_embedding, protein_embedding)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def train_model(nth, args, device):
    # Training dataset basic parameters
    dataset = args.train_dataset
    fold_n = args.fold_n
    form_data_path = './data/' + dataset + '-graph-data'
    # [num_feature, num_gene]
    num_feature = 1
    final_annotation_gene_df = pd.read_csv(os.path.join(form_data_path, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list) 
    num_type = 3
    num_type_node = num_node / num_type
    # Read these feature label files
    print('--- LOADING TRAINING FILES ... ---')
    xTr = np.load(form_data_path + '/xTr' + str(fold_n) + '.npy')
    yTr = np.load(form_data_path + '/yTr' + str(fold_n) + '.npy')
    seq = np.load(form_data_path + '/seq.npy', allow_pickle=True)
    all_edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long()
    internal_edge_index = torch.from_numpy(np.load(form_data_path + '/internal_edge_index.npy') ).long()
    ppi_edge_index = torch.from_numpy(np.load(form_data_path + '/ppi_edge_index.npy') ).long()

    # Load pretrain model
    pretrain_model = build_pretrain_model(args, num_feature, num_node, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()

    # Load finetuned language model
    if args.load_lm_embed == 1:
        dna_embedding = np.load(form_data_path + '/dna_gpt_embedding.npy')
        rna_embedding = np.load(form_data_path + '/rna_gpt_embedding.npy')
        protein_embedding = np.load(form_data_path + '/protein_gpt_embedding.npy')
    else:
        dna_gpt_lm, rna_gpt_lm, prot_gpt_lm = build_finetune_model(device)
        dna_embedding, rna_embedding, protein_embedding = language_model_embedding(num_type_node, dna_gpt_lm, rna_gpt_lm, prot_gpt_lm, seq)
        np.save(form_data_path + '/dna_gpt_embedding.npy', dna_embedding.cpu().numpy())
        np.save(form_data_path + '/rna_gpt_embedding.npy', rna_embedding.cpu().numpy())
        np.save(form_data_path + '/protein_gpt_embedding.npy', protein_embedding.cpu().numpy())

    # Training model stage starts
    model = build_graphclas_model(args, num_node, device)
    dl_input_num = xTr.shape[0]
    epoch_num = args.num_train_epoch
    batch_size = args.batch_size

    # Record training results
    epoch_loss_list = []
    epoch_acc_list = []
    test_loss_list = []
    test_acc_list = []
    max_test_acc = 0
    max_test_acc_id = 0

    # Clean result previous epoch_i_pred files
    folder_name = 'epoch_' + str(epoch_num) + '_fold_' + str(fold_n)
    unit = nth
    path = './data/' + dataset + '-result/' + args.train_result_path + '/%s-%d' % (folder_name, unit)
    while os.path.exists('./data/' + dataset + '-result/' + args.train_result_path) == False:
        os.mkdir('./data/' + dataset + '-result/' + args.train_result_path)
    while os.path.exists(path):
        unit += 1
        path = './data/' + dataset + '-result/' + args.train_result_path + '/%s-%d' % (folder_name, unit)
    os.mkdir(path)

    for i in range(1, epoch_num + 1):
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        model.train()
        epoch_ypred = np.zeros((1, 1))
        upper_index = 0
        batch_loss_list = []
        dl_input_num = xTr.shape[0]
        
        for index in range(0, dl_input_num, batch_size):
            if (index + batch_size) < dl_input_num:
                upper_index = index + batch_size
            else:
                upper_index = dl_input_num
            geo_train_datalist = read_batch(index, upper_index, xTr, yTr, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index)
            train_dataset_loader = GeoGraphLoader.load_graph(geo_train_datalist, args)
            model, batch_loss, batch_acc, batch_ypred = train_graphclas_model(train_dataset_loader, dna_embedding, rna_embedding, protein_embedding,
                                                                               pretrain_model, model, device, args)
            print('BATCH LOSS: ', batch_loss)
            print('BATCH ACCURACY: ', batch_acc)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = (Variable(batch_ypred).data).cpu().numpy().reshape(-1, 1)
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = float(np.mean(batch_loss_list))
        print('TRAIN EPOCH ' + str(i) + ' LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        # print('ITERATION NUMBER UNTIL NOW: ' + str(iteration_num))
        # Preserve acc corr for every epoch
        score_lists = list(yTr)
        score_list = [item for elem in score_lists for item in elem]
        epoch_ypred_lists = list(epoch_ypred)
        epoch_ypred_list = [item for elem in epoch_ypred_lists for item in elem]
        train_dict = {'label': score_list, 'prediction': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # Calculating metrics
        accuracy = accuracy_score(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        epoch_acc_list.append(accuracy)
        f1 = f1_score(tmp_training_input_df['label'], tmp_training_input_df['prediction'], average='binary')
        conf_matrix = confusion_matrix(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        tn, fp, fn, tp = conf_matrix.ravel()
        print('EPOCH ' + str(i) + ' TRAINING ACCURACY: ', accuracy)
        print('EPOCH ' + str(i) + ' TRAINING F1: ', f1)
        print('EPOCH ' + str(i) + ' TRAINING CONFUSION MATRIX: ', conf_matrix)
        print('EPOCH ' + str(i) + ' TRAINING TN: ', tn)
        print('EPOCH ' + str(i) + ' TRAINING FP: ', fp)
        print('EPOCH ' + str(i) + ' TRAINING FN: ', fn)
        print('EPOCH ' + str(i) + ' TRAINING TP: ', tp)

        print('\n-------------EPOCH TRAINING ACCURACY LIST: -------------')
        print(epoch_acc_list)
        print('\n-------------EPOCH TRAINING LOSS LIST: -------------')
        print(epoch_loss_list)

        # # # Test model on test dataset
        test_acc, test_loss, tmp_test_input_df = test_model(args, pretrain_model, model, device, i)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index=False, header=True)
        print('\n-------------EPOCH TEST ACCURACY LIST: -------------')
        print(test_acc_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)
        # SAVE BEST TEST MODEL
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            max_test_acc_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pt')
            torch.save(model.state_dict(), path + '/best_train_model.pt')
            tmp_training_input_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
            tmp_test_input_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
        print('\n-------------BEST TEST ACCURACY MODEL ID INFO:' + str(max_test_acc_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN ACCURACY: ', epoch_acc_list[max_test_acc_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TEST ACCURACY: ', test_acc_list[max_test_acc_id - 1])
        write_best_model_info(fold_n, path, max_test_acc_id, epoch_loss_list, epoch_acc_list, test_loss_list, test_acc_list)

def test_model(args, pretrain_model, model, device, i):
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    # Test model on test dataset
    fold_n = args.fold_n
    dataset = args.train_dataset
    form_data_path = './data/' + dataset + '-graph-data'
    xTe = np.load(form_data_path + '/xTe' + str(fold_n) + '.npy')
    yTe = np.load(form_data_path + '/yTe' + str(fold_n) + '.npy')
    all_edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long()
    internal_edge_index = torch.from_numpy(np.load(form_data_path + '/internal_edge_index.npy') ).long()
    ppi_edge_index = torch.from_numpy(np.load(form_data_path + '/ppi_edge_index.npy') ).long()
    dna_embedding = np.load(form_data_path + '/dna_gpt_embedding.npy')
    rna_embedding = np.load(form_data_path + '/rna_gpt_embedding.npy')
    protein_embedding = np.load(form_data_path + '/protein_gpt_embedding.npy')

    dl_input_num = xTe.shape[0]
    batch_size = args.batch_size
    # Clean result previous epoch_i_pred files
    # [num_feature, num_node]
    num_feature = 1
    final_annotation_gene_df = pd.read_csv(os.path.join(form_data_path, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)
    # Run test model
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, dl_input_num, batch_size):
        if (index + batch_size) < dl_input_num:
            upper_index = index + batch_size
        else:
            upper_index = dl_input_num
        geo_datalist = read_batch(index, upper_index, xTe, yTe, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index)
        test_dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args)
        print('TEST MODEL...')
        model, batch_loss, batch_acc, batch_ypred = test_graphclas_model(test_dataset_loader, dna_embedding, rna_embedding, protein_embedding,
                                                                          pretrain_model, model, device)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        print('BATCH ACCURACY: ', batch_acc)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = batch_ypred.reshape(-1, 1)
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = float(np.mean(batch_loss_list))
    print('EPOCH ' + str(i) + ' TEST LOSS: ', test_loss)
    # Preserve accuracy for every epoch
    all_ypred = np.delete(all_ypred, 0, axis=0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_lists = list(yTe)
    score_list = [item for elem in score_lists for item in elem]
    test_dict = {'label': score_list, 'prediction': all_ypred_list}
    # import pdb; pdb.set_trace()
    tmp_test_input_df = pd.DataFrame(test_dict)
    # Calculating metrics
    accuracy = accuracy_score(tmp_test_input_df['label'], tmp_test_input_df['prediction'])
    f1 = f1_score(tmp_test_input_df['label'], tmp_test_input_df['prediction'], average='binary')
    conf_matrix = confusion_matrix(tmp_test_input_df['label'], tmp_test_input_df['prediction'])
    tn, fp, fn, tp = conf_matrix.ravel()
    print('EPOCH ' + str(i) + ' TEST ACCURACY: ', accuracy)
    print('EPOCH ' + str(i) + ' TEST F1: ', f1)
    print('EPOCH ' + str(i) + ' TEST CONFUSION MATRIX: ', conf_matrix)
    print('EPOCH ' + str(i) + ' TEST TN: ', tn)
    print('EPOCH ' + str(i) + ' TEST FP: ', fp)
    print('EPOCH ' + str(i) + ' TEST FN: ', fn)
    print('EPOCH ' + str(i) + ' TEST TP: ', tp)
    test_acc = accuracy
    return test_acc, test_loss, tmp_test_input_df


def test_trained_model(args, device):
    print('\n------------- LOAD MODEL AND TEST -------------')
    dataset = args.train_dataset
    fold_n = args.fold_n
    form_data_path = './data/' + dataset + '-graph-data'
    final_annotation_gene_df = pd.read_csv(os.path.join(form_data_path, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)
    num_feature = 1
    # Load pretrain model
    pretrain_model = build_pretrain_model(args, num_feature, num_node, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()
    # Load model
    model = build_graphclas_model(args, num_node, device)
    folder_name = 'epoch_' + str(args.num_train_epoch) + '_fold_' + str(fold_n) + '_best'
    path = './data/' + dataset + '-result/' + args.train_result_path + '/%s' % (folder_name)
    model.load_state_dict(torch.load(path + '/best_train_model.pt'))
    test_model(args, pretrain_model, model, device, i=0)


def arg_parse():
    parser = argparse.ArgumentParser()
    # pre-training parameters
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=1, help='Channels of GNN encoder layers. (default: 1)')
    parser.add_argument('--hidden_channels', type=int, default=1, help='Channels of hidden representation. (default: 1)')
    parser.add_argument('--decoder_channels', type=int, default=1, help='Channels of decoder layers. (default: 1)')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=3, help='Number of layers for internal encoder. (default: 3)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--edge_p', type=float, default=0.001, help='Mask ratio or sample ratio for EdgeMask')
    parser.add_argument('--node_p', type=float, default=0.001, help='Mask ratio or sample ratio for NodeMask')
    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization. (default: False)')
    parser.add_argument('--layer', nargs='?', default='gat', help='GNN layer, (default: gat)')
    parser.add_argument('--encoder_activation', nargs='?', default='elu', help='Activation function for GNN encoder, (default: elu)')
    parser.add_argument('--save_path', nargs='?', default='./models/pretrain_gnn/pretrained-gnn.pt', 
                        help='save path for model. (default: ./models/pretrain_gnn/pretrained-gnn.pt)')
    parser.add_argument('--load_lm_embed', dest='load_lm_embed', type=int, default=1, help='Whether to load the language model embedding. (default: 1)')

    # training parameters
    parser.add_argument('--fold_n', dest='fold_n', type=int, default=1, help='Fold number for training. (default: 1)')
    parser.add_argument('--train_dataset', dest='train_dataset', type=str, default='UCSC', help='Dataset for training. (default: UCSC)')
    parser.add_argument('--num_train_epoch', dest='num_train_epoch', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size of training.')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int, default=0, help = 'Number of workers to load data.')
    parser.add_argument('--train_lr', dest='train_lr', type=float, default=0.005, help='Learning rate for training. (default: 0.005)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay for training. (default: 1e-5)')
    parser.add_argument('--eps', dest='eps', type=float, default=1e-7, help='Epsilon for Adam. (default: 1e-7)')

    parser.add_argument('--train_input_dim', dest='train_input_dim', type=int, default=3, help='Input dimension of training. (default: 3)')
    parser.add_argument('--train_hidden_dim', dest='train_hidden_dim', type=int, default=9, help='Hidden dimension of training. (default: 9)')
    parser.add_argument('--train_embedding_dim', dest='train_embedding_dim', type=int, default=9, help='Embedding dimension of training. (default: 9)')
    parser.add_argument('--lm_dim', dest='lm_dim', type=int, default=1, help='Language model embedding dimension. (default: 1)')
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes for classification. (default: 2)')
    parser.add_argument('--num_heads', dest='num_heads', type=int, default=3, help='Number of heads for attention. (default: 3)')

    parser.add_argument('--train_result_path', nargs='?', dest='train_result_path', default='graphseqlm-gpt', help='save path for model result. (default: graphseqlm-gpt)')

    # test parameters by loading model (both pretrained and trained)
    parser.add_argument('--load', dest='load', type=int, default=0, help='Whether to load the model. (default: 0)')

    return parser.parse_args()


if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))

    # Set cuda device
    device = torch.device('cuda:0') 
    torch.cuda.set_device(device)
    print('MAIN DEVICE: ', device)

    # Train
    k = 5
    fold_num_train = 10
    if args.load == 0: 
        for fold_n in range(1, k + 1):
            args.fold_n = fold_n
            for nth in range(1, fold_num_train + 1):
                train_model(nth, args, device)
    else: 
        test_trained_model(args, device)