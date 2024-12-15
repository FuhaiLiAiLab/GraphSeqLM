# ## 7.Convert the processed data into node dictionary

# load processed data
import pandas as pd
import os

# read the file names under the folder
# Define the path to the output folder where CSV files are stored
output_folder = 'UCSC-process'

# List of file names you saved earlier
file_names = [
    'gene-tran-list', 
    'gene-geno-list', 
    'gene-protein-list', 
    'gene-all-list', 
    'gene-kegg-edge-list', 
    'patient-sample-list', 
    'phenotype-lists', 
    'processed-genotype-mutation', 
    'processed-genotype-gene-expression', 
    'processed-genotype-proteomics', 
    'processed-sequence-dna',
    'processed-sequence-rna',
    'processed-sequence-protein',
    'processed-phenotype-immune-subtype-transposed', 
    'processed-phenotype-survival-transposed', 
    'processed-phenotype-dense-transposed', 
    'processed-phenotype-cellsub-transposed'
]

# Dictionary to hold the dataframes
dataframes = {}

# Read each file and assign to a dataframe
for file_name in file_names:
    full_path = os.path.join(output_folder, file_name + '.csv')
    dataframes[file_name] = pd.read_csv(full_path)

# Assign each dataframe to a variable
sorted_gene_tran_df = dataframes['gene-tran-list']
sorted_gene_genomics_df = dataframes['gene-geno-list']
sorted_gene_protein_df = dataframes['gene-protein-list']
sorted_all_gene_df = dataframes['gene-all-list']
filtered_up_kegg_df = dataframes['gene-kegg-edge-list']
sorted_patient_sample_df = dataframes['patient-sample-list']
phenotype_lists = dataframes['phenotype-lists']
mutation_filtered = dataframes['processed-genotype-mutation']
gene_expression_filtered = dataframes['processed-genotype-gene-expression']
protein_filtered = dataframes['processed-genotype-proteomics']
dna_seq_filtered = dataframes['processed-sequence-dna']
rna_seq_filtered = dataframes['processed-sequence-rna']
protein_seq_filtered = dataframes['processed-sequence-protein']
immune_subtype_filtered = dataframes['processed-phenotype-immune-subtype-transposed']
survival_filtered = dataframes['processed-phenotype-survival-transposed']
dense_filtered = dataframes['processed-phenotype-dense-transposed']
cellsub_filtered = dataframes['processed-phenotype-cellsub-transposed']

# outputfile name
graph_output_folder = 'UCSC-graph-data'
# create folder if not exist
if not os.path.exists(graph_output_folder):
    os.makedirs(graph_output_folder)

# ### 7.1 Make nodes dictionary

sorted_all_gene_dict = sorted_all_gene_df['Gene'].to_dict()
sorted_all_gene_name_dict = {value: key for key, value in sorted_all_gene_dict.items()}
num_gene = sorted_gene_tran_df.shape[0]
num_gene_protein = sorted_gene_protein_df.shape[0]
nodetype_list = ['Gene-Geno'] * num_gene + ['Gene-Tran'] * num_gene +  ['Gene-Prot'] * num_gene_protein
map_all_gene_df = pd.DataFrame({'Gene_num': sorted_all_gene_dict.keys(), 'Gene_name': sorted_all_gene_dict.values(), 'NodeType': nodetype_list})
map_all_gene_df.to_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'), index=False)

# ### 7.2 Create the edges connection between promoter methylations and proteins

# [Gene-Epi - Gene-Geno]
sorted_gene_geno = sorted_gene_genomics_df['Gene'].tolist()
sorted_gene_tran = sorted_gene_tran_df['Gene'].tolist()
sorted_gene_protein = sorted_gene_protein_df['Gene'].tolist()
# [Gene-Geno - Gene-Tran]
geno_tran_edge_df = pd.DataFrame({'src': sorted_gene_geno, 'dest': sorted_gene_tran})
# [Gene-Tran - Gene-Prot]
tran_prot_edge_df = pd.DataFrame({'src': sorted_gene_tran, 'dest': sorted_gene_protein})

print(sorted_all_gene_name_dict['ABL1-Geno'])
print(sorted_all_gene_name_dict['ABL1-Tran'])
print(sorted_all_gene_name_dict['ABL1-Prot'])

# replace gene name with gene number
geno_tran_num_edge_df = geno_tran_edge_df.copy()
geno_tran_num_edge_df['src'] = geno_tran_edge_df['src'].map(sorted_all_gene_name_dict)
geno_tran_num_edge_df['dest'] = geno_tran_edge_df['dest'].map(sorted_all_gene_name_dict)
tran_prot_num_edge_df = tran_prot_edge_df.copy()
tran_prot_num_edge_df['src'] = tran_prot_edge_df['src'].map(sorted_all_gene_name_dict)
tran_prot_num_edge_df['dest'] = tran_prot_edge_df['dest'].map(sorted_all_gene_name_dict)

# ### 7.3 Concat all of the edges

selected_database = 'KEGG'
if selected_database == 'KEGG':
    filtered_up_num_df = filtered_up_kegg_df.copy()

# Add 'PROT' to the end of each gene name in the 'src' and 'dest' columns
filtered_up_num_df['src'] = filtered_up_num_df['src'].apply(lambda x: x + '-Prot')
filtered_up_num_df['dest'] = filtered_up_num_df['dest'].apply(lambda x: x + '-Prot')

filtered_up_num_df['src'] = filtered_up_num_df['src'].map(sorted_all_gene_name_dict)
filtered_up_num_df['dest'] = filtered_up_num_df['dest'].map(sorted_all_gene_name_dict)
all_gene_edge_num_df = pd.concat([filtered_up_num_df, geno_tran_num_edge_df, tran_prot_num_edge_df])

num_ppi_edge = filtered_up_num_df.shape[0]
num_geno_tran_edge = geno_tran_num_edge_df.shape[0]
num_tran_prot_edge = tran_prot_num_edge_df.shape[0]
edgetype_list = ['Gene-Prot-Gene-Prot'] * num_ppi_edge + ['Gene-Geno-Gene-Tran'] * num_geno_tran_edge + ['Gene-Tran-Gene-Prot'] * num_tran_prot_edge
all_gene_edge_num_df['EdgeType'] = edgetype_list
all_gene_edge_num_df = all_gene_edge_num_df.sort_values(by=['src', 'dest']).reset_index(drop=True)
all_gene_edge_num_df.to_csv(os.path.join(graph_output_folder, 'all-gene-edge-num.csv'), index=False)

internal_gene_edge_num_df = pd.concat([geno_tran_num_edge_df, tran_prot_num_edge_df])
num_geno_tran_edge = geno_tran_num_edge_df.shape[0]
num_tran_prot_edge = tran_prot_num_edge_df.shape[0]
edgetype_list = ['Gene-Geno-Gene-Tran'] * num_geno_tran_edge + ['Gene-Tran-Gene-Prot'] * num_tran_prot_edge
internal_gene_edge_num_df['EdgeType'] = edgetype_list
internal_gene_edge_num_df = internal_gene_edge_num_df.sort_values(by=['src', 'dest']).reset_index(drop=True)
internal_gene_edge_num_df.to_csv(os.path.join(graph_output_folder, 'internal-gene-edge-num.csv'), index=False)

ppi_edge_num_df = filtered_up_num_df.copy()
num_ppi_edge = filtered_up_num_df.shape[0]
edgetype_list = ['Gene-Prot-Gene-Prot'] * num_ppi_edge
ppi_edge_num_df['EdgeType'] = edgetype_list
ppi_edge_num_df = ppi_edge_num_df.sort_values(by=['src', 'dest']).reset_index(drop=True)
ppi_edge_num_df.to_csv(os.path.join(graph_output_folder, 'ppi-edge-num.csv'), index=False)

# gene edge interactions without map
all_gene_edge_df = all_gene_edge_num_df.copy()
all_gene_edge_df = all_gene_edge_df.replace(sorted_all_gene_dict)
internal_gene_edge_df = internal_gene_edge_num_df.replace(sorted_all_gene_dict)
ppi_edge_df = ppi_edge_num_df.replace(sorted_all_gene_dict)

# all_gene_edge_df = all_gene_edge_df.sort_values(by=['src', 'dest']).reset_index(drop=True)
all_gene_edge_df.to_csv(os.path.join(graph_output_folder, 'all-gene-edge.csv'), index=False)
internal_gene_edge_df.to_csv(os.path.join(graph_output_folder, 'internal-gene-edge.csv'), index=False)
ppi_edge_df.to_csv(os.path.join(graph_output_folder, 'ppi-edge.csv'), index=False)

# ## 8.Load data into graph format

# ### 8.1 Form up the input samples

# recommends the use of the endpoints of OS, PFI, DFI, and DSS for each TCGA cancer type
# 
# * OS: overall survial
# * PFI: progression-free interval
# * DSS: disease-specific survival
# * DFI: disease-free interval

survival_filtered

survival_filtered_feature_df = survival_filtered.copy()
survival_filtered_feature_df = survival_filtered_feature_df[['sample', 'cancer type abbreviation', 'OS', 'vital_status']]

nan_counts = survival_filtered_feature_df.isna().sum()  # or df.isnull()
print(nan_counts)

# Convert 'alive' to 0.0 and 'dead' to 1.0
survival_filtered_feature_df['vital_status'] = survival_filtered_feature_df['vital_status'].map({'Alive': 0.0, 'Dead': 1.0})
survival_filtered_feature_df['OS'] == survival_filtered_feature_df['vital_status']


# Check if each row in Column1 and Column2 have the same value
rows_same = (survival_filtered_feature_df['OS'] == survival_filtered_feature_df['vital_status']).all()
print("All rows have the same value in column 'OS' and column 'vital_status' :", rows_same)

survival_filtered_feature_df = survival_filtered_feature_df[['sample', 'OS', 'cancer type abbreviation']]
survival_filtered_feature_df.to_csv(os.path.join(graph_output_folder, 'survival-label.csv'), index=False)

# Separate the sample by cancer type and show each cancer type's number of samples and their names
cancer_types = survival_filtered_feature_df['cancer type abbreviation'].unique()
print(cancer_types)
cancer_type_df_dict = {}
for cancer_type in cancer_types:
    cancer_type_df_dict[cancer_type] = survival_filtered_feature_df[survival_filtered_feature_df['cancer type abbreviation'] == cancer_type]
    cancer_type_df_dict[cancer_type] = cancer_type_df_dict[cancer_type].drop(columns=['cancer type abbreviation'])
    # Example directory path
    dir_path = os.path.join(graph_output_folder, cancer_type)
    # Use makedirs with exist_ok=True to avoid errors if the directory already exists
    os.makedirs(dir_path, exist_ok=True)
    cancer_type_df_dict[cancer_type].to_csv(os.path.join(graph_output_folder, cancer_type, 'survival-label.csv'), index=False)
    print(cancer_type, cancer_type_df_dict[cancer_type].shape[0])


# Fetch samples of [mutation_filtered, gene_expression_filtered, protein_filtered] data for each cancer type, (filter colmun names)
cancer_type_dict = {}
for cancer_type in cancer_types:
    cancer_type_dict[cancer_type] = {}
    cancer_type_dict[cancer_type]['mutation'] = mutation_filtered[['gene_name'] + list(cancer_type_df_dict[cancer_type]['sample'])]
    cancer_type_dict[cancer_type]['gene_expression'] = gene_expression_filtered[['gene_name'] + list(cancer_type_df_dict[cancer_type]['sample'])]
    cancer_type_dict[cancer_type]['protein'] = protein_filtered[['gene_name'] + list(cancer_type_df_dict[cancer_type]['sample'])]
    cancer_type_dict[cancer_type]['mutation'].to_csv(os.path.join(graph_output_folder, cancer_type, 'mutation.csv'), index=False)
    cancer_type_dict[cancer_type]['gene_expression'].to_csv(os.path.join(graph_output_folder, cancer_type, 'gene_expression.csv'), index=False)
    cancer_type_dict[cancer_type]['protein'].to_csv(os.path.join(graph_output_folder, cancer_type, 'protein.csv'), index=False) 


# ### 8.2 Generate X and Y numpy files for pan-cancer and cancer-specific data
import numpy as np
# Pan-cancer data
x_df = pd.concat([mutation_filtered, gene_expression_filtered, protein_filtered], axis=0)
x_df = x_df.drop(columns=['gene_name'])
x_df = x_df.T
# convert x_df to numpy array
x = x_df.values
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)
# save x to numpy file
np.save(os.path.join(graph_output_folder, 'x.npy'), x)

# Pan-cancer label
y_df = survival_filtered_feature_df[['OS']]
# convert y_df to numpy array
y = y_df.values
print(y.shape)
# save y to numpy file
np.save(os.path.join(graph_output_folder, 'y.npy'), y)

# Cancer-specific data
for cancer_type in cancer_types:
    x_df = pd.concat([cancer_type_dict[cancer_type]['mutation'], cancer_type_dict[cancer_type]['gene_expression'], cancer_type_dict[cancer_type]['protein']], axis=0)
    x_df = x_df.drop(columns=['gene_name'])
    x_df = x_df.T
    # convert x_df to numpy array
    x = x_df.values
    x = x.reshape((x.shape[0], x.shape[1], 1))
    print(x.shape)
    # save x to numpy file
    np.save(os.path.join(graph_output_folder, cancer_type, 'x.npy'), x)

    y_df = cancer_type_df_dict[cancer_type][['OS']]
    # convert y_df to numpy array
    y = y_df.values
    print(y.shape)
    # save y to numpy file
    np.save(os.path.join(graph_output_folder, cancer_type, 'y.npy'), y)

# ### 8.3 Generate edge index information

import torch
from scipy import sparse
# Form a whole adjacent matrix
internal_gene_num_df = pd.read_csv(os.path.join(graph_output_folder, 'internal-gene-edge-num.csv'))
src_gene_list = list(internal_gene_num_df['src'])
dest_gene_list = list(internal_gene_num_df['dest'])
final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
gene_name_list = list(final_annotation_gene_df['Gene_name'])
num_node = len(gene_name_list)
adj = np.zeros((num_node, num_node))
# gene-gene adjacent matrix
for i in range(len(src_gene_list)):
    row_idx = src_gene_list[i]
    col_idx = dest_gene_list[i]
    adj[row_idx, col_idx] = 1
# import pdb; pdb.set_trace()
# np.save(form_data_path + '/adj.npy', adj)
adj_sparse = sparse.csr_matrix(adj)
sparse.save_npz(graph_output_folder + '/internal_adj_sparse.npz', adj_sparse)
# [edge_index]
source_nodes, target_nodes = adj_sparse.nonzero()
internal_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
print(internal_edge_index.shape)
print(internal_edge_index)
np.save(graph_output_folder + '/internal_edge_index.npy', internal_edge_index)

# Form a whole adjacent matrix
ppi_gene_num_df = pd.read_csv(os.path.join(graph_output_folder, 'ppi-edge-num.csv'))
src_gene_list = list(ppi_gene_num_df['src'])
dest_gene_list = list(ppi_gene_num_df['dest'])
final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
gene_name_list = list(final_annotation_gene_df['Gene_name'])
num_node = len(gene_name_list)
adj = np.zeros((num_node, num_node))
# gene-gene adjacent matrix
for i in range(len(src_gene_list)):
    row_idx = src_gene_list[i]
    col_idx = dest_gene_list[i]
    adj[row_idx, col_idx] = 1
    adj[col_idx, row_idx] = 1 # whether we want ['sym']
adj_sparse = sparse.csr_matrix(adj)
sparse.save_npz(graph_output_folder + '/ppi_adj_sparse.npz', adj_sparse)
# [edge_index]
source_nodes, target_nodes = adj_sparse.nonzero()
ppi_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
print(ppi_edge_index.shape)
print(ppi_edge_index)
np.save(graph_output_folder + '/ppi_edge_index.npy', ppi_edge_index)

edge_index = torch.cat([internal_edge_index, ppi_edge_index], dim=1)
print(edge_index.shape)
print(edge_index)
np.save(graph_output_folder + '/edge_index.npy', edge_index)

# ### 8.4 Reprocess the edge_index file after loading

import os
import numpy as np
import pandas as pd

graph_output_folder = 'UCSC-graph-data'
edge_index = np.load(graph_output_folder + '/edge_index.npy')
# Convert the 2D array into a DataFrame
edge_index_df = pd.DataFrame(edge_index.T, columns=['src', 'dest'])

gene_edge_num_df = pd.read_csv(os.path.join(graph_output_folder, 'all-gene-edge-num.csv'))
src_gene_list = list(gene_edge_num_df['src'])
dest_gene_list = list(gene_edge_num_df['dest'])
edgetype_list = list(gene_edge_num_df['EdgeType'])
gene_edge_num_reverse_df = pd.DataFrame({'src': dest_gene_list, 'dest': src_gene_list, 'EdgeType': edgetype_list})
gene_edge_num_all_df = pd.concat([gene_edge_num_df, gene_edge_num_reverse_df]).drop_duplicates().sort_values(by=['src', 'dest']).reset_index(drop=True)

merged_gene_edge_num_all_df = pd.merge(gene_edge_num_all_df, edge_index_df, on=['src', 'dest'], how='inner')
merged_gene_edge_num_all_df.to_csv(os.path.join(graph_output_folder, 'merged-gene-edge-num-all.csv'), index=False)

merged_gene_edge_name_all_df = merged_gene_edge_num_all_df.replace(sorted_all_gene_dict)
merged_gene_edge_name_all_df.to_csv(os.path.join(graph_output_folder, 'merged-gene-edge-name-all.csv'), index=False)

# ### 8.5 Concat the molecular sequence files

# concat dna_seq_filtered, rna_seq_filtered, protein_seq_filtered (replaceing column names dna_sequence, rna_sequence, protein_sequence with sequence)
dna_seq_filtered.rename(columns={'dna_sequence': 'sequence'}, inplace=True)
rna_seq_filtered.rename(columns={'transcript_sequence': 'sequence'}, inplace=True)
protein_seq_filtered.rename(columns={'protein_sequence': 'sequence'}, inplace=True)
seq_df = pd.concat([dna_seq_filtered, rna_seq_filtered, protein_seq_filtered], axis=0)[['sequence']].reset_index(drop=True)
# convert seq_df to numpy array
seq = seq_df.values
print(seq.shape)
print(seq)  
# save seq to numpy file
np.save(os.path.join(graph_output_folder, 'seq.npy'), seq)

# ### 8.6 K-fold split

import os
import numpy as np
import pandas as pd
graph_output_folder = 'UCSC-graph-data'
x = np.load(os.path.join(graph_output_folder, 'x.npy'))
y = np.load(os.path.join(graph_output_folder, 'y.npy'))
survival_label_df = pd.read_csv(os.path.join(graph_output_folder, 'survival-label.csv'))

from sklearn.model_selection import StratifiedKFold
# Initialize 5-fold cross-validator
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare dictionaries to hold the splits
folds_data = {
    "x_train": [],
    "y_train": [],
    "label_train": [],
    "x_test": [],
    "y_test": [],
    "label_test": []
}

# Generate the 5-fold splits
for train_index, test_index in kf.split(x, y):
    folds_data["x_train"].append(x[train_index])
    folds_data["y_train"].append(y[train_index])
    folds_data["label_train"].append(survival_label_df.iloc[train_index])
    
    folds_data["x_test"].append(x[test_index])
    folds_data["y_test"].append(y[test_index])
    folds_data["label_test"].append(survival_label_df.iloc[test_index])

# Save folds into variables for direct access
for i in range(5):
    globals()[f"xTr{i+1}"] = folds_data["x_train"][i]
    globals()[f"yTr{i+1}"] = folds_data["y_train"][i]
    globals()[f"label_train{i+1}"] = folds_data["label_train"][i]
    
    globals()[f"xTe{i+1}"] = folds_data["x_test"][i]
    globals()[f"yTe{i+1}"] = folds_data["y_test"][i]
    globals()[f"label_test{i+1}"] = folds_data["label_test"][i]

# Verify the shapes of generated folds
for i in range(5):
    print(f"Fold {i+1}:")
    print(f"xTr{i+1}.shape: {globals()[f'xTr{i+1}'].shape}, xTe{i+1}.shape: {globals()[f'xTe{i+1}'].shape}")
    print(f"yTr{i+1}.shape: {globals()[f'yTr{i+1}'].shape}, yTe{i+1}.shape: {globals()[f'yTe{i+1}'].shape}")
    print(f"label_train{i+1}.shape: {globals()[f'label_train{i+1}'].shape}, label_test{i+1}.shape: {globals()[f'label_test{i+1}'].shape}")
    print()

    # Save training data
    np.save(os.path.join(graph_output_folder, f"xTr{i+1}.npy"), globals()[f"xTr{i+1}"])
    np.save(os.path.join(graph_output_folder, f"yTr{i+1}.npy"), globals()[f"yTr{i+1}"])
    globals()[f"label_train{i+1}"].to_csv(
        os.path.join(graph_output_folder, f"label_train{i+1}.csv"), index=False
    )

    # Save testing data
    np.save(os.path.join(graph_output_folder, f"xTe{i+1}.npy"), globals()[f"xTe{i+1}"])
    np.save(os.path.join(graph_output_folder, f"yTe{i+1}.npy"), globals()[f"yTe{i+1}"])
    globals()[f"label_test{i+1}"].to_csv(
        os.path.join(graph_output_folder, f"label_test{i+1}.csv"), index=False
    )

# Function to check consistency between OS column and y arrays
def check_consistency():
    for i in range(5):
        # Training consistency check
        train_os = globals()[f"label_train{i+1}"]['OS'].values.reshape(-1, 1)
        train_y = globals()[f"yTr{i+1}"]
        train_consistent = np.array_equal(train_os, train_y)

        # Testing consistency check
        test_os = globals()[f"label_test{i+1}"]['OS'].values.reshape(-1, 1)
        test_y = globals()[f"yTe{i+1}"]
        test_consistent = np.array_equal(test_os, test_y)

        # Print results
        print(f"Fold {i+1}:")
        print(f"  Training consistency: {'Consistent' if train_consistent else 'Inconsistent'}")
        print(f"  Testing consistency: {'Consistent' if test_consistent else 'Inconsistent'}")
        print()

# Run the consistency check
check_consistency()

seq = np.load(os.path.join(graph_output_folder, 'seq.npy'), allow_pickle=True)
print(seq[0])
print(seq[2111])
print(seq[4222])
# Node Type
map_all_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
