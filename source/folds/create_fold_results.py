import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
​
import seaborn as sns



def compute_max_accuracy(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['Accuracy']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
    
def compute_max_P(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['Precision Macro']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
    
def compute_max_R(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['Recall Macro']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
    
def compute_max_F(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['F1 Score Macro']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
    
    
def compute_max_Pm(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['Precision Micro']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
    
def compute_max_Rm(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['Recall Micro']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
    
def compute_max_Fm(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if 'Accuracy' column exists
    if 'Accuracy' in data.columns:
        # Extract the 'Accuracy' column
        accuracy_column = data['F1 Score Micro']

        # Find the maximum value in the 'Accuracy' column
        max_acc = accuracy_column.argmax()
        return accuracy_column[max_acc]
    else:
        return "Hel"
# Specify the top-level
# Specify the top-level directory
top_directory = './models'  # Assuming the script is in the same directory as the CSV files

# Create an empty DataFrame to store results
result_df = []

# Loop through each subdirectory
for directory in os.listdir(top_directory):
    directory_path = os.path.join(top_directory, directory)
    
    # Check if the item is a directory
    if os.path.isdir(directory_path):
        # Loop through each file in the directory
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory_path, file_name)
                print(file_path)

                # Compute the maximum accuracy for the current CSV file
                max_accuracy = compute_max_accuracy(file_path)
                R = compute_max_R(file_path)
                P = compute_max_P(file_path)
                F = compute_max_F(file_path)
                Rm = compute_max_Rm(file_path)
                Pm = compute_max_Pm(file_path)
                Fm = compute_max_Fm(file_path)
                # Append the result to the DataFrame
                if max_accuracy is not None:
                    result_df.append({'Directory Name': directory, 'Max Accuracy': max_accuracy,
                                      'Recall Macro':R, "Precision Macro":P, "F1 Score Macro":F,
                                     'Recall Micro':Rm, "Precision Micro":Pm, "F1 Score Micro":Fm})

# Display the result DataFrame
print(result_df)

df = pd.DataFrame(result_df)
df["fold"] = df["Directory Name"].apply(lambda x: x.split("fold")[1])
df = df[['fold','Max Accuracy','Recall Macro','Recall Micro','Precision Macro','Precision Micro','F1 Score Macro','F1 Score Micro']].sort_values(by="fold")
df.to_csv("fold_results.csv")


