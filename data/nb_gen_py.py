# Prepare the content for the Python file with markdown as comments
converted_content = []

import nbformat

# Load the notebook file
notebook_path = 'union_raw_data_process_UCSC.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as file:
    notebook = nbformat.read(file, as_version=4)


# Prepare content for the Python file, starting from Section 7 and removing 'display' statements
converted_content = []
in_target_section = False

for cell in notebook.cells:
    if cell.cell_type == 'markdown':
        # Check for specific headers
        if '## 7.Convert the processed data into node dictionary' in cell.source:
            in_target_section = True
            print("Processing begins: 7.Convert the processed data into node dictionary")
        elif '## 8.Load data into graph format' in cell.source:
            print("Processing ends: 8.Load data into graph format")
            in_target_section = True

        if in_target_section:
            # Add markdown content as comments
            markdown_lines = cell.source.split('\n')
            converted_content.extend([f"# {line}" for line in markdown_lines])
            converted_content.append("")  # Add a blank line after comments

    elif cell.cell_type == 'code' and in_target_section:
        # Add code content, excluding lines with 'display'
        code_lines = cell.source.split('\n')
        filtered_code = [line for line in code_lines if 'display' not in line]
        converted_content.extend(filtered_code)
        converted_content.append("")  # Add a blank line after code

# Combine all content into a single block
final_python_code = '\n'.join(converted_content)

# Save the Python code to a file
output_file_path = 'processed_data_gen.py'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(final_python_code)

output_file_path
