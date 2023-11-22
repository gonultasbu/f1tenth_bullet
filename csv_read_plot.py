import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime

start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
# Specify the directory where your CSV files are located
folder_path = './data/friction'

# Get a list of CSV files in the specified directory
# csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files = [f for f in os.listdir(folder_path) if f.endswith('092_2.csv') or f.endswith('092_1.csv')]
# Initialize a plot
plt.figure(figsize=(10, 5))

# Loop through the list of CSV files
for file in csv_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['x','y','z'])
    
    # For Labeling 
    parts = file.split('_')
    # modified_part = parts[-2][0] + '.' + parts[-2][1:]
    modified_part = parts[-2] + "_" + parts[-1]
    # Plot x and z values
    plt.plot(df['x'], df['z'], label=modified_part[:-4])

# Set plot labels and title
plt.xlabel('X coordinate')
plt.ylabel('Z coordinate')
plt.title('X and Z Coordinates from Multiple CSV Files')
plt.legend()
plt.grid(True)
# Display the plot
plt.show()
