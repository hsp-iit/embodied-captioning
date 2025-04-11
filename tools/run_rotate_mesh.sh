#!/bin/bash

# Check if the folder path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

# Use the first argument as the folder path
FOLDER_PATH="$1"

# Loop through all files in the folder ending with '_semantic.ply'
for file in "$FOLDER_PATH"/*_semantic.ply; do
    # Call the Python script with the file as an argument
    python scripts/rotate_mesh.py "$file"
    
    # Get the base filename without the extension
    BASENAME=$(basename "$file" .ply)
    
    # Rename the file by appending '_not_rotated' before the extension
    mv "$file" "$FOLDER_PATH/${BASENAME}_not_rotated.ply"
    
    # Output the renaming confirmation
    echo "Renamed: ${BASENAME}.ply to ${BASENAME}_not_rotated.ply"
done

# After the loop, rename all the files ending with '_semantic_rotate.ply' back to '_semantic.ply'
for rotated_file in "$FOLDER_PATH"/*_semantic_rotated.ply; do
    # Get the original name by removing '_rotate' from the filename
    original_name="${rotated_file/_rotated/}"
    
    # Rename the file
    mv "$rotated_file" "$original_name"
    
    # Output the renamed file for confirmation
    echo "Renamed: $rotated_file to $original_name"
done
