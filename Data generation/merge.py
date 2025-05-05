import csv

def split_csv(data):
    """Splits a list of rows into two halves."""
    midpoint = len(data) // 2
    return data[:midpoint], data[midpoint:]

def read_csv(file_path):
    """Reads a CSV file and returns its rows."""
    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
            if not data:  # Check if the file is empty
                print(f"Warning: {file_path} is empty.")
                return []
            return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def merge_csv_splits(file1, file2):
    """Merges two CSV files by appending the first half of file2 to the first half of file1, 
       and the second half of file2 to the second half of file1."""
    
    # Read the first CSV file
    data1 = read_csv(file1)
    # Read the second CSV file
    data2 = read_csv(file2)
    
    # If either file is empty, return an empty list or handle accordingly
    if not data1 or not data2:
        print("One or both CSV files are empty or invalid.")
        return []

    # Split both data sets (skipping header row)
    header1, rows1 = data1[0], data1[1:]
    header2, rows2 = data2[0], data2[1:]

    # Ensure headers match before merging (optional, based on use case)
    if header1 != header2:
        print("Warning: CSV headers do not match. Merging may be incorrect.")

    first_half1, second_half1 = split_csv(rows1)
    first_half2, second_half2 = split_csv(rows2)
    
    # Merge the first halves and the second halves
    merged_first_half = first_half1 + first_half2
    merged_second_half = second_half1 + second_half2
    
    # Combine them into one list, starting with the header
    merged_data = [header1] + merged_first_half + merged_second_half
    
    return merged_data




file1 = 'text_statistics_trans_1.csv'
file2 = 'text_statistics_trans_2.csv'
merged_data = merge_csv_splits(file1, file2)

if merged_data:
    with open('merged_file.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(merged_data)

for i in range(3, 11):
    file1 = 'merged_file.csv'
    file2 = f'text_statistics_trans_{i}.csv'
    merged_data = merge_csv_splits(file1, file2)

    if merged_data:
        with open('merged_file.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(merged_data)