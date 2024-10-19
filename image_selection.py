import os


directory = 'dataset/test/images'  # Replace with your directory path
list = []
# List all files in the directory
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        list.append(filename)
hurricane_files = [filename for filename in list if 'hurricane' in filename]
print(hurricane_files)