#!/usr/bin/env python3
import os
import re
import sys

if len(sys.argv) != 2:
    print("Usage: python rename_files.py <offset>")
    sys.exit(1)

try:
    offset = int(sys.argv[1])
except ValueError:
    print("Offset must be an integer.")
    sys.exit(1)

pattern = re.compile(r'image_values_iteration_([0-9]+)\.txt\.npz')
temp_suffix = ".tmp_rename"

# Phase 1: Rename to temporary filenames
for filename in os.listdir('.'):
    match = pattern.fullmatch(filename)
    if match:
        number = int(match.group(1))
        new_number = number + offset
        temp_filename = f'image_values_iteration_{new_number}.txt.npz{temp_suffix}'
        print(f"Temporarily renaming '{filename}' to '{temp_filename}'")
        os.rename(filename, temp_filename)

# Phase 2: Rename temporary files to final filenames
for filename in os.listdir('.'):
    if filename.endswith(temp_suffix):
        final_filename = filename[:-len(temp_suffix)]
        print(f"Final renaming '{filename}' to '{final_filename}'")
        os.rename(filename, final_filename)

