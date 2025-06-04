import re
import os
import json

def generate_toc_from_jupyter_notebook(notebook_path):
    """
    Reads a Jupyter Notebook (.ipynb) file, identifies Markdown headers, 
    and generates a Markdown Table of Contents with clickable links.
    
    This version correctly handles Jupyter's anchor generation rules.

    Args:
        notebook_path (str): The full path to the .ipynb notebook file.

    Returns:
        str: A string containing the Markdown for the Table of Contents.
             Returns None if the file is not found, is not a valid notebook,
             or no headers are found.
    """
    toc_entries = []
    raw_titles_for_anchor_generation = []

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook file '{notebook_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{notebook_path}'. Is it a valid .ipynb file?")
        return None

    if 'cells' not in notebook_content:
        print(f"Error: No 'cells' found in '{notebook_path}'. Not a valid notebook structure.")
        return None

    for cell in notebook_content['cells']:
        if cell.get('cell_type') == 'markdown':
            # cell['source'] can be a list of strings or a single string
            source_lines = cell.get('source', [])
            if isinstance(source_lines, str):
                source_lines = source_lines.splitlines()
            
            for line in source_lines:
                stripped_line = line.strip()
                # Look for Markdown headers
                match = re.match(r'^(#+)\s+(.*)', stripped_line)
                if match:
                    level_hashes = match.group(1)  # e.g., "##"
                    header_title = match.group(2).strip()  # e.g., "Section Title"
                    
                    # Store the full Markdown line for display and level detection
                    toc_entries.append(stripped_line) 
                    # Store just the title part for anchor generation
                    raw_titles_for_anchor_generation.append(header_title)
    
    if not toc_entries:
        print("No Markdown headers found in the notebook to generate a table of contents.")
        return None

    # Generate Markdown TOC
    markdown_toc = "### Table of Contents \n\n"
    
    for i, display_title_md_header in enumerate(toc_entries):
        anchor_title_base = raw_titles_for_anchor_generation[i]
        
        # Generate the anchor ID following Jupyter's rules
        slug = jupyter_anchor_generation(anchor_title_base)
        
        # Determine indentation based on Markdown header level
        match = re.match(r'^(#+)\s+', display_title_md_header)
        level = len(match.group(1))  # Number of '#'
        indent = "  " * (level - 1)  # Two spaces per level, starting from level 1
        
        # Use the clean title (without #) for the link text
        display_title_for_link = anchor_title_base

        markdown_toc += f"{indent}- [{display_title_for_link}](#{slug})\n"

    print("--- Generated Markdown Table of Contents ---")
    print(markdown_toc)
    print("--- End of Markdown ---")
    print("\nInstructions: ")
    print("1. Ensure the notebook you analyzed has been SAVED.")
    print("2. Copy the Markdown above.")
    print("3. Create a new Markdown cell at the top of THAT SAME notebook.")
    print("4. Paste the copied Markdown into it and run the cell.")
    return markdown_toc


def jupyter_anchor_generation(text):
    """
    Generate anchor ID following Jupyter's actual rules:
    1. Convert to lowercase
    2. Replace spaces with hyphens
    3. Remove most special characters except hyphens and underscores
    4. DO NOT collapse multiple hyphens (this preserves " - " as "---")
    5. Strip leading/trailing hyphens
    
    Args:
        text (str): The header text to convert to an anchor
        
    Returns:
        str: The anchor ID
    """
    # Convert to lowercase
    slug = text.lower()
    
    # Replace spaces with hyphens
    slug = slug.replace(' ', '-')
    
    # Remove special characters except alphanumeric, hyphens, and underscores
    # This keeps parentheses, colons, etc. as empty strings (effectively removing them)
    slug = re.sub(r'[^a-z0-9\-_]', '', slug)
    
    # DO NOT collapse multiple hyphens - Jupyter keeps them!
    # This is the key difference - " - " becomes "---" and stays that way
    
    # Remove leading and trailing hyphens
    slug = slug.strip('-')
    
    return slug

# --- How to use it ---
# 1. Make sure the Jupyter Notebook you want to create a TOC for is SAVED.
# 2. In a Python cell WITHIN THAT SAME NOTEBOOK (or another notebook/script), 
#    call the function, providing the path to the notebook file.
# 3. Copy the Markdown output printed by the function.
# 4. Create a new Markdown cell at the top of THE NOTEBOOK YOU ANALYZED.
# 5. Paste the copied Markdown into it and run the cell. You should see a clickable Table of Contents.

# Example usage:
# generate_toc_from_jupyter_notebook('notebook.ipynb')