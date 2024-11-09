import os
import glob
import requests
import pandas as pd
from pyensembl import EnsemblRelease

# Function to download the gene names from Ensembl
def fecth_gene_names():
    # Initialize PyEnsembl with the desired release
    # Use the latest release number for GRC.hg38
    latest_release_number = 108  # Replace 108 with the latest release number
    ensembl_release = EnsemblRelease(release=latest_release_number)
    # Ensure data is downloaded and indexed
    ensembl_release.download()
    ensembl_release.index()
    # Fetch all genes (this might be a large list, so be cautious)
    genes = ensembl_release.genes()
    print(f"Total genes: {len(genes)}")
    return genes

# Function to fetch gene information, DNA, transcript, and protein sequences using Ensembl REST API
def fetch_gene_details(gene_id):
    server = "https://rest.ensembl.org"
    headers_json = {"Content-Type": "application/json"}
    headers_text = {"Content-Type": "text/plain"}
    # Initialize outputs
    gene_info, dna_sequence, transcript_sequence, protein_sequence = {}, "", "", ""
    try:
        # Fetch gene information
        gene_info_response = requests.get(f"{server}/lookup/id/{gene_id}?expand=1", headers=headers_json)
        gene_info = gene_info_response.json() if gene_info_response.ok else {}
    except Exception as e:
        print(f"Error fetching gene information for {gene_id}: {e}")
    try:
        # Fetch DNA sequence
        dna_seq_response = requests.get(f"{server}/sequence/id/{gene_id}?type=genomic", headers=headers_text)
        if dna_seq_response.ok:
            dna_sequence = dna_seq_response.text
    except Exception as e:
        print(f"Error fetching DNA sequence for {gene_id}: {e}")
    # Assuming we fetch sequences for the canonical transcript as a demonstration
    if gene_info.get("canonical_transcript"):
        try:
            # Fetch canonical transcript sequence
            canonical_transcript_id = gene_info["canonical_transcript"]
            transcript_id = canonical_transcript_id.split(".")[0]
            transcript_seq_response = requests.get(f"{server}/sequence/id/{transcript_id}", headers=headers_text)
            if transcript_seq_response.ok:
                transcript_sequence = transcript_seq_response.text
        except Exception as e:
            print(f"Error fetching transcript sequence for {transcript_id}: {e}")
        # Attempt to fetch first protein sequence (if applicable)
        transcripts = gene_info.get('Transcript')
        for transcript in transcripts:
            if 'Translation' in transcript:
                protein_id = transcript['Translation']['id']
                protein_sequence_endpoint = f"/sequence/id/{protein_id}?type=protein"
                try:
                    # Fetch protein sequence
                    protein_response = requests.get(server + protein_sequence_endpoint, headers=headers_text)
                    if protein_response.ok:
                        protein_sequence = protein_response.text
                    else:
                        print(f"Error fetching protein sequence for {protein_id}")
                except Exception as e:
                    print(f"Error fetching protein sequence for {protein_id}: {e}")
                break  # Stop after fetching the first protein sequence
    # Compile details into a dictionary
    return {
        "gene_id": gene_id,
        "gene_name": gene_info.get("display_name", ""),
        "contig": gene_info.get("seq_region_name", ""),
        "start": gene_info.get("start", ""),
        "end": gene_info.get("end", ""),
        "strand": gene_info.get("strand", ""),
        "dna_sequence": dna_sequence,
        "transcript_sequence": transcript_sequence,
        "protein_sequence": protein_sequence
    }

# Function to divide the genes list into chunks of 1000.
def divide_chunks(l, n):
    # Looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

# Function to fetch gene details for each chunk of genes and save to CSV files
def fetch_gene_details_by_chunk(gene_chunks, batch_size):
    # Process each chunk
    chunk_num = 0 # current chunk number to start from
    gene_count = chunk_num * batch_size  # Reset count for all chunk.
    for chunk in gene_chunks[chunk_num: ]:
        print(f"Processing chunk of {len(chunk)} genes ...")
        gene_data_df_list = []  # Reset this for each chunk.
        for gene in chunk:
            print(f"Fetching {gene.gene_id} ...")
            gene_count += 1
            print(f"Gene count: {gene_count}")
            # Assuming gene.gene_id is how you access the ID, adjust if necessary.
            gene_details = fetch_gene_details(gene.gene_id)
            gene_details_df = pd.DataFrame([gene_details])
            print(gene_details_df)
            gene_data_df_list.append(gene_details_df)
        # Convert the list to a DataFrame
        gene_chunk_df = pd.concat(gene_data_df_list)
        # Save the DataFrame to a CSV file
        gene_chunk_df.to_csv("./sequences/Chunk" + str(chunk_num) + "_gene_transcript_protein_sequences.csv", index=False)
        chunk_num += 1

# Replace Thymine (T) with Uracil (U) in RNA sequences
def replace_thymine_with_uracil_in_rna():
    # Get all CSV files in the 'sequence' folder
    sequence_files = glob.glob('./sequences/*.csv')
    # Loop through each file and replace 'T' with 'U' in the DNA and RNA sequences
    for file in sequence_files:
        df = pd.read_csv(file)
        df['transcript_sequence'] = df['transcript_sequence'].str.replace('T', 'U')
        # Save the updated DataFrame to the same file
        df.to_csv(file, index=False)

# Combine all gene sequences file into one file
def combine_gene_sequence_files():
    # Get all CSV files in the 'sequence' folder
    sequence_files = glob.glob('./sequences/*.csv')
    # Initialize an empty DataFrame to store the combined data
    combined_df_list = []
    # Loop through each file and append the data to the combined DataFrame
    for file in sequence_files:
        df = pd.read_csv(file)
        combined_df_list.append(df)
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('./sequences/combined_sequences.csv', index=False)


if __name__ == "__main__":
    # # Fetch gene names from Ensembl
    # genes = fecth_gene_names()

    # # Fetch gene details for each gene and save to CSV files
    # batch_size = 1000 # Specify the batch size
    # gene_chunks = list(divide_chunks(genes, batch_size)) # Divide the genes into chunks of 1000
    # fetch_gene_details_by_chunk(gene_chunks, batch_size)
    
    # # Thymine (T) in DNA is replaced by Uracil (U) in RNA
    # replace_thymine_with_uracil_in_rna()

    # Combine all gene sequences file into one file
    combine_gene_sequence_files()