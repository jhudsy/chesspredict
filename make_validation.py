import os
import random
import shutil

PERCENTAGE_TO_MOVE = 10

# Define the source and destination directories
source_dir = 'data/training/'
destination_dir = 'data/validation/'

# Get a list of all files in the source directory
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Calculate 10% of the total number of files
num_files_to_move = (len(all_files) * PERCENTAGE_TO_MOVE)//100

# Randomly select 10% of the files
files_to_move = random.sample(all_files, num_files_to_move)

# Move the selected files to the destination directory
for file_name in files_to_move:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(destination_dir, file_name))

print(f'Moved {num_files_to_move} files from {source_dir} to {destination_dir}')