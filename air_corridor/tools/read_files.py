import os
import torch
# Define the folder path where the models are stored
folder_path = ''
exps=['multi-deeper-annel_hor:32_batch:8_curFalse_boidFalse_seed:1']
folder_path=f"{os.getcwd()}/result/d2move__20231112212944/{exps[0]}"
# List all files in the folder
files = os.listdir(folder_path)


# Filter out files that match the pattern "critic*.pth"
model_files = [f for f in files if f.startswith('ppo_critic') and f.endswith('.pth')]

# Extract the number 'n' from each model file name
model_numbers = []
for file in model_files:
    try:
        # Extract the number part from the filename (assuming the format is critic'n'.pth)
        number = int(file.split('ppo_critic')[1].split('.pth')[0])
        model_numbers.append((number, file))
    except ValueError:
        # If the filename does not follow the expected format, ignore it
        pass

# Find the file with the largest number 'n'
if model_numbers:
    latest_model_file = max(model_numbers, key=lambda x: x[0])[1]
    # Load the model
    model_path = os.path.join(folder_path, latest_model_file)
    model = torch.load(model_path)
    model_name = latest_model_file
else:
    model = None
    model_name = "No model found following the specified pattern."

model_name, model