import os
import re
import pandas as pd


def parse_model_info(file_path):
    """Parses the `best_model_info.txt` file and extracts relevant metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        fold = int(re.search(r"Fold: (\d+)", content).group(1))
    except AttributeError:
        print(f"Error: Could not find 'Fold' in {file_path}")
        fold = None

    try:
        train_loss = float(re.search(r"BEST MODEL TRAIN LOSS: ([\d.]+)", content).group(1))
    except AttributeError:
        print(f"Error: Could not find 'BEST MODEL TRAIN LOSS' in {file_path}")
        train_loss = None

    try:
        train_acc = float(re.search(r"BEST MODEL TRAIN ACCURACY: ([\d.]+)", content).group(1))
    except AttributeError:
        print(f"Error: Could not find 'BEST MODEL TRAIN ACCURACY' in {file_path}")
        train_acc = None

    try:
        test_loss = float(re.search(r"BEST MODEL TEST LOSS: ([\d.]+)", content).group(1))
    except AttributeError:
        print(f"Error: Could not find 'BEST MODEL TEST LOSS' in {file_path}")
        test_loss = None

    try:
        test_acc = float(re.search(r"BEST MODEL TEST ACCURACY: ([\d.]+)", content).group(1))
    except AttributeError:
        print(f"Error: Could not find 'BEST MODEL TEST ACCURACY' in {file_path}")
        test_acc = None

    return fold, train_loss, train_acc, test_loss, test_acc


def extract_model_data(root_dir):
    """Extracts model data from the given folder structure."""
    data = []
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if os.path.isdir(model_path):
            # Extract model name
            model_name = model_dir
            for subdir in os.listdir(model_path):
                if subdir.startswith('epoch'):
                    epoch = int(re.search(r"epoch_(\d+)", subdir).group(1))
                    fold = int(re.search(r"fold_(\d+)", subdir).group(1))
                    experiment_path = os.path.join(model_path, subdir)
                    info_file = os.path.join(experiment_path, 'best_model_info.txt')
                    
                    if os.path.exists(info_file):
                        fold, train_loss, train_acc, test_loss, test_acc = parse_model_info(info_file)
                        data.append({
                            'Model': model_name,
                            'Epoch': epoch,
                            'Fold': fold,
                            'Train Loss': train_loss,
                            'Train Accuracy': train_acc,
                            'Test Loss': test_loss,
                            'Test Accuracy': test_acc
                        })
    return data


def summarize_results(df):
    """Summarizes results with avg ± std for each model."""
    summary = df.groupby('Model').agg(
        Train_Loss_Mean=('Train Loss', 'mean'),
        Train_Loss_Std=('Train Loss', 'std'),
        Train_Accuracy_Mean=('Train Accuracy', 'mean'),
        Train_Accuracy_Std=('Train Accuracy', 'std'),
        Test_Loss_Mean=('Test Loss', 'mean'),
        Test_Loss_Std=('Test Loss', 'std'),
        Test_Accuracy_Mean=('Test Accuracy', 'mean'),
        Test_Accuracy_Std=('Test Accuracy', 'std')
    )

    # Combine mean and std into "avg ± std" format
    summary = summary.apply(lambda x: x.round(4))
    summary['Train Loss (avg ± std)'] = summary['Train_Loss_Mean'].astype(str) + ' ± ' + summary['Train_Loss_Std'].astype(str)
    summary['Train Accuracy (avg ± std)'] = summary['Train_Accuracy_Mean'].astype(str) + ' ± ' + summary['Train_Accuracy_Std'].astype(str)
    summary['Test Loss (avg ± std)'] = summary['Test_Loss_Mean'].astype(str) + ' ± ' + summary['Test_Loss_Std'].astype(str)
    summary['Test Accuracy (avg ± std)'] = summary['Test_Accuracy_Mean'].astype(str) + ' ± ' + summary['Test_Accuracy_Std'].astype(str)
    
    return summary[['Train Loss (avg ± std)', 'Train Accuracy (avg ± std)', 'Test Loss (avg ± std)', 'Test Accuracy (avg ± std)']]


def main():
    root_dir = './'  # Replace with the path to your folder structure
    data = extract_model_data(root_dir)
    
    # Create a DataFrame for analysis
    df = pd.DataFrame(data)
    
    # Detailed per-fold statistics
    print("Detailed per-fold statistics:")
    print(df)
    
    # Summarize results with avg ± std
    summary = summarize_results(df)
    print("\nSummary (avg ± std) per model:")
    print(summary)
    
    # Save the results to CSV if needed
    df.to_csv('detailed_model_statistics.csv', index=False)
    summary.to_csv('summary_statistics_avg_std.csv')


if __name__ == "__main__":
    main()


