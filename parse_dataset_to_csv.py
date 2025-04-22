import os
import re
import csv

def parse_dataset(root_dir="datasets"):
    """
    Parses data files in the specified directory and saves them as CSV files.

    Args:
        root_dir (str): The root directory containing the dataset folders.
    """
    for dataset_dir in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_dir)

        # Skip if it's not a directory
        if not os.path.isdir(dataset_path):
            continue

        for filename in os.listdir(dataset_path):
            if filename.endswith(".dat"):
                filepath = os.path.join(dataset_path, filename)
                species_name = filename[:-4]  # Remove ".dat" extension

                output_csv_path = os.path.join(dataset_path, f"{species_name}.csv")

                print(f"Parsing {filepath} to {output_csv_path}")

                with open(filepath, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
                    csv_writer = csv.writer(outfile)
                    csv_writer.writerow(['position', 'temperature', 'value'])  # Write header

                    # Read the file and extract data
                    for line in infile:
                        # Skip comment lines
                        if line.startswith(('Number', '---')):
                            continue

                        # Use regex to find the numbers in each line
                        match = re.findall(r"([-+]?\d*\.\d+|[-+]?\d+)", line)
                        if match and len(match) == 3:
                            position, temperature, value = map(float, match)
                            csv_writer.writerow([position, temperature, value])

if __name__ == "__main__":
    parse_dataset()