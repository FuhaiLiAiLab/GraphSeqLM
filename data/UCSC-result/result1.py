import os
import re
import warnings
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def parse_model_info(file_path):
    """Parses the `best_model_info.txt` file and extracts relevant metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        fold = int(re.search(r"Fold: (\d+)", content).group(1))
    except AttributeError:
        # print(f"Error: Could not find 'Fold' in {file_path}")
        fold = None

    try:
        train_loss = float(re.search(r"BEST MODEL TRAIN LOSS: ([\d.]+)", content).group(1))
    except AttributeError:
        # print(f"Error: Could not find 'BEST MODEL TRAIN LOSS' in {file_path}")
        train_loss = None

    try:
        train_acc = float(re.search(r"BEST MODEL TRAIN ACCURACY: ([\d.]+)", content).group(1))
    except AttributeError:
        # print(f"Error: Could not find 'BEST MODEL TRAIN ACCURACY' in {file_path}")
        train_acc = None

    try:
        test_loss = float(re.search(r"BEST MODEL TEST LOSS: ([\d.]+)", content).group(1))
    except AttributeError:
        # print(f"Error: Could not find 'BEST MODEL TEST LOSS' in {file_path}")
        test_loss = None

    try:
        test_acc = float(re.search(r"BEST MODEL TEST ACCURACY: ([\d.]+)", content).group(1))
    except AttributeError:
        # print(f"Error: Could not find 'BEST MODEL TEST ACCURACY' in {file_path}")
        test_acc = None

    return fold, train_loss, train_acc, test_loss, test_acc


def calculate_metrics(file_path):
    """Calculates survival/death rates, F1-score, recall, sensitivity, and specificity."""
    df = pd.read_csv(file_path)
    
    # Calculate survival and death rates
    total = len(df)
    survival_rate = (df['label'] == 1.0).sum() / total
    death_rate = (df['label'] == 0.0).sum() / total

    # Calculate F1-score, recall, sensitivity, and specificity
    f1 = f1_score(df['label'], df['prediction'])
    recall = recall_score(df['label'], df['prediction'])
    precision = precision_score(df['label'], df['prediction'])

    # Confusion matrix for specificity and sensitivity
    tn, fp, fn, tp = confusion_matrix(df['label'], df['prediction'], labels=[0.0, 1.0]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return survival_rate, death_rate, f1, recall, sensitivity, specificity


def count_samples_and_labels(file_path):
    """
    Counts the total number of samples, survival (1), and death (0) from a file.
    
    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: (total_samples, survival_count, death_count)
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip the header
            survival_count = sum(1 for line in lines if line.strip().split(',')[-1] == '1.0')
            death_count = sum(1 for line in lines if line.strip().split(',')[-1] == '0.0')
            total_samples = survival_count + death_count
        return total_samples, survival_count, death_count
    return None, None, None


def extract_model_data(root_dir):
    """Extracts model data from the given folder structure."""
    data = []
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if not os.path.isdir(model_path):  # Ensure it's a directory
            continue
        for model_cancer in os.listdir(model_path):
            # Count the number of samples in the 'dataset' folder
            dataset_path = os.path.join('../UCSC-graph-data/', model_cancer)
            sample_file = os.path.join(dataset_path, 'survival-label.csv')
            total_samples, survival_count, death_count = count_samples_and_labels(sample_file)
            # import pdb; pdb.set_trace()
            model_cancer_dir = os.path.join(model_path, model_cancer)
            if os.path.isdir(model_cancer_dir):
                # Extract model name
                model_name = model_path + '-' + model_cancer
                model_name = model_name.replace('./', '')
                for subdir in os.listdir(model_cancer_dir):  # Corrected to iterate over the correct directory
                    if subdir.startswith('epoch'):
                        epoch = int(re.search(r"epoch_(\d+)", subdir).group(1))
                        fold = int(re.search(r"fold_(\d+)", subdir).group(1))
                        experiment_path = os.path.join(model_cancer_dir, subdir)
                        experiment_number = int(re.search(r'-(\d+)$', experiment_path).group(1))
                        info_file = os.path.join(experiment_path, 'best_model_info.txt')
                        train_file = os.path.join(experiment_path, 'BestTrainingPred.txt')
                        test_file = os.path.join(experiment_path, 'BestTestPred.txt')

                        if os.path.exists(info_file):
                            fold, train_loss, train_acc, test_loss, test_acc = parse_model_info(info_file)
                            if train_loss is None or train_acc is None or test_loss is None or test_acc is None:
                                continue
                            if os.path.exists(train_file):
                                train_survival_rate, train_death_rate, train_f1, train_recall, train_sensitivity, train_specificity = calculate_metrics(train_file)
                            else:
                                train_survival_rate = train_death_rate = train_f1 = train_recall = train_sensitivity = train_specificity = None
                            
                            if os.path.exists(test_file):
                                test_survival_rate, test_death_rate, test_f1, test_recall, test_sensitivity, test_specificity = calculate_metrics(test_file)
                            else:
                                test_survival_rate = test_death_rate = test_f1 = test_recall = test_sensitivity = test_specificity = None
                            
                            data.append({
                                'Model': f"{model_name}-fold_{fold}-{experiment_number}",
                                'Epoch': epoch,
                                'Train Loss': train_loss,
                                'Train Accuracy': train_acc,
                                'Test Loss': test_loss,
                                'Test Accuracy': test_acc,
                                'Train Survival Rate': train_survival_rate,
                                'Train Death Rate': train_death_rate,
                                'Train F1-Score': train_f1,
                                'Test Survival Rate': test_survival_rate,
                                'Test Death Rate': test_death_rate,
                                'Test F1-Score': test_f1,
                                'Number of Samples': total_samples,  # Number of samples
                                'Survival Count': survival_count,
                                'Death Count': death_count
                            })
    return data


def keep_best_rows(df):
    """
    Keeps only the rows with the best Test Accuracy and best Test F1-Score under each unique model_cancer_name-fold.

    Args:
        df (pd.DataFrame): The detailed results DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only the best rows.
    """
    # Step 1: Extract model_cancer_name-fold (truncate after '-fold_x')
    df['Model_Group'] = df['Model'].apply(lambda x: '-'.join(x.split('-')[:-1]))

    # Step 2: Initialize an empty list to store the best rows
    best_rows = []

    # Step 3: Group by 'Model_Group' (model_cancer_name-fold)
    for group_name, group_df in df.groupby('Model_Group'):
        # Find the row with the best Test Accuracy
        best_acc_row = group_df.loc[group_df['Test Accuracy'].idxmax()]
        
        # # Find the row with the best Test F1-Score
        # best_f1_row = group_df.loc[group_df['Test F1-Score'].idxmax()]
        
        # Append both rows to the list
        best_rows.append(best_acc_row)
        # best_rows.append(best_f1_row)
    
    # # Step 4: Combine all best rows into a new DataFrame and drop duplicates
    best_df = pd.DataFrame(best_rows).drop_duplicates()

    # Step 5: Sort the rows for clarity and reset the index
    best_df = best_df.sort_values(by=['Model_Group', 'Test Accuracy'], ascending=[True, False]).reset_index(drop=True)

    # Step 6: Drop the temporary 'Model_Group' column
    best_df = best_df.drop(columns=['Model_Group'])

    return best_df


def summarize_results(df):
    """
    Summarizes results grouped by model-cancer_type with avg ± std for key metrics.

    Args:
        df (pd.DataFrame): The DataFrame containing detailed results.

    Returns:
        pd.DataFrame: Summarized metrics grouped by model-cancer_type.
    """
    # Step 1: Extract model-cancer_type (e.g., 'graphseqlm-gpt-ACC')
    df['Model_Cancer'] = df['Model'].apply(lambda x: '-'.join(x.split('-')[:-2]))

    # Step 2: Group by 'Model_Cancer' and calculate summary statistics
    summary = df.groupby('Model_Cancer').agg(
        Train_Loss_Mean=('Train Loss', 'mean'),
        Train_Loss_Std=('Train Loss', 'std'),
        Train_Accuracy_Mean=('Train Accuracy', 'mean'),
        Train_Accuracy_Std=('Train Accuracy', 'std'),
        Train_F1_Mean=('Train F1-Score', 'mean'),
        Train_F1_Std=('Train F1-Score', 'std'),
        Test_Loss_Mean=('Test Loss', 'mean'),
        Test_Loss_Std=('Test Loss', 'std'),
        Test_Accuracy_Mean=('Test Accuracy', 'mean'),
        Test_Accuracy_Std=('Test Accuracy', 'std'),
        Test_F1_Mean=('Test F1-Score', 'mean'),
        Test_F1_Std=('Test F1-Score', 'std'),
        Train_Survival_Rate_Mean=('Train Survival Rate', 'mean'),
        Train_Survival_Rate_Std=('Train Survival Rate', 'std'),
        Train_Death_Rate_Mean=('Train Death Rate', 'mean'),
        Train_Death_Rate_Std=('Train Death Rate', 'std'),
        Test_Survival_Rate_Mean=('Test Survival Rate', 'mean'),
        Test_Survival_Rate_Std=('Test Survival Rate', 'std'),
        Test_Death_Rate_Mean=('Test Death Rate', 'mean'),
        Test_Death_Rate_Std=('Test Death Rate', 'std'),
        Number_of_Samples=('Number of Samples', 'mean'),
        Surivial_Count=('Survival Count', 'mean'),
        Death_Count=('Death Count', 'mean')
    )

    # Step 3: Combine mean and std into "avg ± std" format
    summary = summary.apply(lambda x: x.round(4))
    summary['Train Loss'] = summary['Train_Loss_Mean'].astype(str) + ' ± ' + summary['Train_Loss_Std'].astype(str)
    summary['Train Accuracy'] = summary['Train_Accuracy_Mean'].astype(str) + ' ± ' + summary['Train_Accuracy_Std'].astype(str)
    summary['Train F1-Score'] = summary['Train_F1_Mean'].astype(str) + ' ± ' + summary['Train_F1_Std'].astype(str)
    summary['Test Loss'] = summary['Test_Loss_Mean'].astype(str) + ' ± ' + summary['Test_Loss_Std'].astype(str)
    summary['Test Accuracy'] = summary['Test_Accuracy_Mean'].astype(str) + ' ± ' + summary['Test_Accuracy_Std'].astype(str)
    summary['Test F1-Score'] = summary['Test_F1_Mean'].astype(str) + ' ± ' + summary['Test_F1_Std'].astype(str)
    # summary['Train Survival Rate'] = summary['Train_Survival_Rate_Mean'].astype(str) + ' ± ' + summary['Train_Survival_Rate_Std'].astype(str)
    # summary['Train Death Rate'] = summary['Train_Death_Rate_Mean'].astype(str) + ' ± ' + summary['Train_Death_Rate_Std'].astype(str)
    # summary['Test Survival Rate'] = summary['Test_Survival_Rate_Mean'].astype(str) + ' ± ' + summary['Test_Survival_Rate_Std'].astype(str)
    # summary['Test Death Rate'] = summary['Test_Death_Rate_Mean'].astype(str) + ' ± ' + summary['Test_Death_Rate_Std'].astype(str)
    summary['# of Samples'] = summary['Number_of_Samples'].astype(int).astype(str) + \
                            ' (Survival: ' + summary['Surivial_Count'].astype(int).astype(str) + \
                            ', Death: ' + summary['Death_Count'].astype(int).astype(str) + ')'

    # Step 4: Add a prefix column for sorting (e.g., 'graphseqlm')
    summary = summary.reset_index()
    summary['Cancer'] = summary['Model_Cancer'].apply(lambda x: x.split('-')[-1])

    # Step 5: Sort by prefix and then by full 'Model_Cancer' name
    summary = summary.sort_values(by=['Cancer']).reset_index(drop=True)
    
    # Step 6: Return final summary table
    return summary[['Cancer', 'Model_Cancer',  
                    '# of Samples',
                    # 'Train Loss', 'Train Accuracy', 'Train F1-Score',
                    'Test Loss', 'Test Accuracy', 'Test F1-Score']]


def main():
    root_dir = './'  # Replace with the path to your folder structure
    data = extract_model_data(root_dir)
    
    # Create a DataFrame for analysis
    df = pd.DataFrame(data)
    
    # # Detailed per-fold statistics
    # print("Detailed per-fold statistics:")
    # print(df)

    # Keep only the best rows based on Test Accuracy and Test F1-Score
    best_df = keep_best_rows(df)
    print("\nBest rows based on Test Accuracy and Test F1-Score:")
    print(best_df)
    
    # Summarize results with avg ± std
    summary = summarize_results(best_df)
    print("\nSummary (avg ± std) per model:")
    print(summary)
    
    # Save the results to CSV if needed
    best_df.to_csv('detailed_model_statistics.csv', index=False)
    summary.to_csv('summary_statistics_avg_std.csv')


if __name__ == "__main__":
    main()

