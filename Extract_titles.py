# Function to process the file and extract the titles
def extract_titles(file_name, output_file):
    titles = []
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            # Process each line in the file
            for line in file:
                # Split by the tab character to separate namespace and title
                parts = line.strip().split('\t', 1)

                # Check if namespace is 0 and if we have a title after the tab (second part)
                if len(parts) > 1 and parts[0] == '0':
                    title = parts[1].replace('_', ' ')  # Replace underscores with spaces
                    titles.append(title)
        
        # Write the titles to the output file
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for title in titles:
                out_file.write(f"{title}\n")
        
        print(f"Titles have been successfully written to '{output_file}'")
        
        return titles
    except FileNotFoundError:
        print(f"The file '{file_name}' was not found.")
        return []

file_name = 'enwiki-latest-all-titles.txt' 
output_file = 'extracted_titles.txt' 

# Extract titles with namespace 0
titles = extract_titles(file_name, output_file)