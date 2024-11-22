import os
import torch
import argparse
import pandas as pd
import numpy as np
import torch_geometric.transforms as T
# custom modules
from texttable import Texttable
from models.pretrain_gnn.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
# custom dataloader
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader


# Function to print the logs in a nice tabular format.
def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()


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


def pretrain_linkpred(pretrain_model, splits, args, device='cpu'):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    test_data = splits['test'].to(device)

    pretrain_model.reset_parameters()

    loss = pretrain_model.train_step(train_data, optimizer,
                                alpha=args.alpha, beta=args.beta)
    torch.save(pretrain_model.state_dict(), args.save_path)

    test_auc, test_ap = pretrain_model.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index)   
    
    # print(f'Link Prediction Pretraining Results:\n'
    #       f'AUC: {test_auc:.2%}',
    #       f'AP: {test_ap:.2%}')
    return test_auc, test_ap


def pretrain_foundation(args, device, num_feature=1):
    # Dataset Selection
    dataset = args.dataset
    graph_output_folder = './data/' + dataset + '-graph-data'
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)

    # Build Pretrain Model
    pretrain_model = build_pretrain_model(args, num_feature, num_node, device)

    # Training dataset basic parameters
    # [num_feature, num_node]
    num_feature = 1
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)
    # Read these feature label files
    print('--- LOADING TRAINING FILES ... ---')
    xAll = np.load(graph_output_folder + '/x.npy')
    yAll = np.load(graph_output_folder + '/y.npy')
    all_edge_index = torch.from_numpy(np.load(graph_output_folder + '/edge_index.npy') ).long()
    internal_edge_index = torch.from_numpy(np.load(graph_output_folder + '/internal_edge_index.npy') ).long()
    ppi_edge_index = torch.from_numpy(np.load(graph_output_folder + '/ppi_edge_index.npy') ).long()
    upper_index = 0
    dl_input_num = xAll.shape[0]
    batch_size = args.batch_size
    
    for epoch in range(args.epochs):
        epoch_auc = 0
        epoch_ap = 0
        num_batches = 0

        for index in range(0, dl_input_num, batch_size):
            upper_index = min(index + batch_size, dl_input_num)
            geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_node, 
                                      all_edge_index, internal_edge_index, ppi_edge_index)
            dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args)

            for batch_idx, data in enumerate(dataset_loader):
                train_data, val_data, test_data = T.RandomLinkSplit(num_test=0.1, num_val=0.0,
                                                                    is_undirected=False,
                                                                    split_labels=True,
                                                                    add_negative_train_samples=False)(data)
                splits = dict(train=train_data, test=test_data) if not args.full_data else dict(train=data, test=test_data)

                # Train and get AUC, AP for this batch
                batch_auc, batch_ap = pretrain_linkpred(pretrain_model, splits, args, device=device)
                epoch_auc += batch_auc
                epoch_ap += batch_ap
                num_batches += 1

            print(f'Epoch {epoch + 1}/{args.epochs} | Batch {index} To - {upper_index} - AUC: {batch_auc:.2%}, AP: {batch_ap:.2%}')

        # Average AUC and AP for the epoch
        epoch_auc /= num_batches
        epoch_ap /= num_batches

        print(f'Average AUC for Epoch {epoch + 1}: {epoch_auc:.2%}, Average AP: {epoch_ap:.2%}')

    # Save final model
    torch.save(pretrain_model.state_dict(), args.save_path)

    return epoch_auc, epoch_ap


def arg_parse():
    parser = argparse.ArgumentParser()
    # pre-training parameters
    parser.add_argument('--dataset', nargs='?', default='UCSC', help='Datasets. (default: UCSC)')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for model and dataset. (default: 2024)')
    parser.add_argument('--pretrain', type=int, default=1, help='Whether to pretrain the model. (default: False)')
    parser.add_argument('--layer', nargs='?', default='gat', help='GNN layer, (default: gat)')
    parser.add_argument('--encoder_activation', nargs='?', default='elu', help='Activation function for GNN encoder, (default: elu)')

    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=1, help='Channels of GNN encoder layers. (default: 1)')
    parser.add_argument('--hidden_channels', type=int, default=1, help='Channels of hidden representation. (default: 1)')
    parser.add_argument('--decoder_channels', type=int, default=1, help='Channels of decoder layers. (default: 1)')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=3, help='Number of layers for internal encoder. (default: 3)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0.002, help='Degree loss weight for degree prediction. (default: 0.002)')
    parser.add_argument('--beta', type=float, default=0.001, help='Node loss weight for degree prediction. (default: 0.001)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for pre-training. (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size for link prediction training. (default: 2**16)')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int, default=0, help = 'Number of workers to load data.')

    parser.add_argument('--start', nargs='?', default='node', help='Which Type to sample starting nodes for random walks, (default: node)')
    parser.add_argument('--edge_p', type=float, default=0.001, help='Mask ratio or sample ratio for EdgeMask')
    parser.add_argument('--node_p', type=float, default=0.001, help='Mask ratio or sample ratio for NodeMask')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=5, help='Number of pre-training epochs. (default: 5)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
    parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
    parser.add_argument('--save_path', nargs='?', default='./models/pretrain_gnn/pretrained-gnn.pt', 
                        help='save path for model. (default: ./models/pretrain_gnn/pretrained-gnn.pt)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--full_data', action='store_true', help='Whether to use full data for pretraining. (default: False)')

    return parser.parse_args()


if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))

    # Set cuda device
    device = torch.device('cuda:0') 
    torch.cuda.set_device(device)
    print('MAIN DEVICE: ', device)
    
    # Pretrain model
    pretrain_model = pretrain_foundation(args, device, num_feature=1)