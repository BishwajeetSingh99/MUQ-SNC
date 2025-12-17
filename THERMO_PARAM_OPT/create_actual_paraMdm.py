'''
#works for one mechanism
import yaml
import csv
import os

# --- Configuration for a SINGLE Test File ---
# ⚠️ CHANGE THIS to the path of your specific YAML file
TEST_FILE_PATH = '/home/user/Desktop/RANA/MUQ-SAC/Database/SYNGAS_COMPLETE/FFCM1.yaml' 
OUTPUT_FILE = 'dm_origional_params.csv'

# Define the FULL list of 20 species labels in the EXACT required order
TARGET_SPECIES_LABELS = [
    'modified_AR_Low_dm_zeta', 'modified_AR_High_dm_zeta',
    'modified_H2_Low_dm_zeta', 'modified_H2_High_dm_zeta',
    'modified_O_Low_dm_zeta', 'modified_O_High_dm_zeta',
    'modified_O2_Low_dm_zeta', 'modified_O2_High_dm_zeta',
    'modified_H2O_Low_dm_zeta', 'modified_H2O_High_dm_zeta',
    'modified_CO_Low_dm_zeta', 'modified_CO_High_dm_zeta',
    'modified_C_Low_dm_zeta', 'modified_C_High_dm_zeta',
    'modified_HCO_Low_dm_zeta', 'modified_HCO_High_dm_zeta',
    'modified_OH*_Low_dm_zeta', 'modified_OH*_High_dm_zeta', 
    'modified_H_Low_dm_zeta', 'modified_H_High_dm_zeta'
]

# --- Helper Function for Coefficient Extraction ---

def get_species_info(label):
    """
    Determines the base species name (e.g., 'AR') and the required thermo data index 
    (0 for High-T/High-zeta, 1 for Low-T/Low-zeta).
    """
    if label.endswith('_High_dm_zeta'):
        thermo_index = 1
        suffix_len = len('_High_dm_zeta')
    elif label.endswith('_Low_dm_zeta'):
        thermo_index = 0
        suffix_len = len('_Low_dm_zeta')
    else:
        return None, None

    base_name = label[:-suffix_len]
    species_name = base_name[len('modified_'):] if base_name.startswith('modified_') else base_name

    return species_name, thermo_index

# --- Main Extraction Logic ---

def extract_nasa7_coefficients_test(filepath, output_filename):
    """
    Processes a single YAML file and writes a single row of 100 coefficients to a CSV (NO FILENAME, NO HEADER).
    """
    print(f"Starting TEST extraction for file: {filepath}")

    # Initialize the row data: [Coeff1, Coeff2, ..., Coeff100]
    # 🚨 FILENAME IS INTENTIONALLY OMITTED HERE
    row_data = [] 
    extraction_successful = True
    
    try:
        # Load the YAML file
        with open(filepath, 'r') as file:
            mechanism_data = yaml.safe_load(file)

        species_list = mechanism_data.get('species', [])
        species_map = {species.get('name'): species for species in species_list if isinstance(species, dict)}
        
        # 3. Iterate through the required species labels (20 total)
        for target_label in TARGET_SPECIES_LABELS:
            species_name, thermo_index = get_species_info(target_label)
            
            found_coeffs = False
            species_entry = species_map.get(species_name) 
            
            # Attempt to find coefficients
            if species_entry and 'thermo' in species_entry and 'data' in species_entry['thermo']:
                nasa7_data = species_entry['thermo']['data']
                
                # Check if the required coefficient set exists
                if isinstance(nasa7_data, list) and len(nasa7_data) > thermo_index and isinstance(nasa7_data[thermo_index], list):
                    coefficients = nasa7_data[thermo_index]
                    
                    # Extract the first 5 coefficients
                    if isinstance(coefficients, list) and len(coefficients) >= 5:
                        row_data.extend(coefficients[:5])
                        found_coeffs = True
            
            if not found_coeffs:
                # Log the missing data and fill the row with N/A
                print(f"⚠️ Warning: Missing data for {target_label} (Base Species: {species_name}, Index: {thermo_index}). Filling with 'N/A'.")
                row_data.extend(['N/A'] * 5)

    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"❌ An error occurred with {filepath}: {e}")
        extraction_successful = False
        # If any major error occurs, ensure the row still has 100 elements
        # (Assuming you want to keep the 100 columns filled with error markers)
        row_data = [f'ERROR_{type(e).__name__}'] * (len(TARGET_SPECIES_LABELS) * 5)


    # 4. Write the result to the CSV file (NO FILENAME, NO HEADER)
    expected_length = len(TARGET_SPECIES_LABELS) * 5
    if len(row_data) == expected_length: 
        try:
            # Use 'w' to overwrite the file for the test
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile: 
                writer = csv.writer(csvfile)
                # 🚨 NO HEADER IS WRITTEN
                writer.writerow(row_data) # 🚨 NO FILENAME IS INCLUDED

            print(f"\n✅ TEST Extraction complete! Data saved to {output_filename}")
            print(f"The row has {len(row_data)} data points (100 expected).")
        except Exception as e:
            print(f"❌ Error writing to CSV file: {e}")
            
    else:
        print(f"\n❌ TEST Extraction FAILED. Final row data size ({len(row_data)}) is incorrect. CSV not created.")

if __name__ == "__main__":
    extract_nasa7_coefficients_test(TEST_FILE_PATH, OUTPUT_FILE)
    '''
    
    
    
    
    
import yaml
import csv
import os

# --- Configuration ---
# 🚨 REQUIRED: Set this to the path where your 36050 YAML files are stored
FOLDER_PATH = '/home/user/Desktop/thermo_param_opt_new/full_run_1/Perturbed_Mech/' 
OUTPUT_FILE = 'dm_origional_params.csv'
NUM_FILES = 36050

# Define the FULL list of 20 species labels in the EXACT required order
TARGET_SPECIES_LABELS = [
    'modified_AR_Low_dm_zeta', 'modified_AR_High_dm_zeta',
    'modified_H2_Low_dm_zeta', 'modified_H2_High_dm_zeta',
    'modified_O_Low_dm_zeta', 'modified_O_High_dm_zeta',
    'modified_O2_Low_dm_zeta', 'modified_O2_High_dm_zeta',
    'modified_H2O_Low_dm_zeta', 'modified_H2O_High_dm_zeta',
    'modified_CO_Low_dm_zeta', 'modified_CO_High_dm_zeta',
    'modified_C_Low_dm_zeta', 'modified_C_High_dm_zeta',
    'modified_HCO_Low_dm_zeta', 'modified_HCO_High_dm_zeta',
    'modified_OH*_Low_dm_zeta', 'modified_OH*_High_dm_zeta', 
    'modified_H_Low_dm_zeta', 'modified_H_High_dm_zeta'
]

# --- Helper Function (from your verified working code) ---

def get_species_info(label):
    """
    Determines the base species name (e.g., 'AR') and the required thermo data index 
    (0 for Low-T/Low-zeta, 1 for High-T/High-zeta, based on user's verified index swap).
    """
    if label.endswith('_High_dm_zeta'):
        thermo_index = 1
        suffix_len = len('_High_dm_zeta')
    elif label.endswith('_Low_dm_zeta'):
        thermo_index = 0
        suffix_len = len('_Low_dm_zeta')
    else:
        return None, None

    base_name = label[:-suffix_len]
    species_name = base_name[len('modified_'):] if base_name.startswith('modified_') else base_name
    
    # Handle the 'OH*' case, which might be stored as 'OH' in some mechanisms
    if species_name == 'OH*':
        species_name = 'OH' 

    return species_name, thermo_index

# --- Bulk Extraction Logic ---

def bulk_extract_nasa7_coefficients():
    """
    Iterates through all 36,050 mechanism files (0 to 36049) in order 
    and writes 100 coefficients per row to the CSV (NO FILENAME, NO HEADER).
    """
    if not os.path.isdir(FOLDER_PATH):
        print(f"❌ Error: Folder path '{FOLDER_PATH}' does not exist. Please update FOLDER_PATH.")
        return

    print(f"Starting bulk extraction for {NUM_FILES} files from '{FOLDER_PATH}'...")
    expected_length = len(TARGET_SPECIES_LABELS) * 5 # Should be 100

    # Open CSV in WRITE mode ('w') to create/overwrite the file
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for i in range(NUM_FILES):
            filename = f"mechanism_{i}.yaml"
            filepath = os.path.join(FOLDER_PATH, filename)
            
            # Initialize the row data (should contain 100 coefficients)
            row_data = [] 
            
            try:
                # 1. Load the YAML file
                with open(filepath, 'r') as file:
                    mechanism_data = yaml.safe_load(file)

                species_list = mechanism_data.get('species', [])
                # Map species by their 'name' key for quick lookup
                species_map = {species.get('name'): species for species in species_list if isinstance(species, dict)}
                
                # 2. Extract data for all target species (20 labels)
                for target_label in TARGET_SPECIES_LABELS:
                    species_name, thermo_index = get_species_info(target_label)
                    
                    found_coeffs = False
                    species_entry = species_map.get(species_name) 
                    
                    if species_entry and 'thermo' in species_entry and 'data' in species_entry['thermo']:
                        nasa7_data = species_entry['thermo']['data']
                        
                        # Check if the required coefficient set exists
                        if isinstance(nasa7_data, list) and len(nasa7_data) > thermo_index:
                            coefficients = nasa7_data[thermo_index]
                            
                            # Extract the first 5 coefficients
                            if isinstance(coefficients, list) and len(coefficients) >= 5:
                                row_data.extend(coefficients[:5])
                                found_coeffs = True
                    
                    if not found_coeffs:
                        # Missing data: fill with 'N/A'
                        row_data.extend(['N/A'] * 5)
                        
                # 3. Write Row if structure is correct
                if len(row_data) == expected_length:
                    writer.writerow(row_data)
                else:
                    print(f"❌ Critical structure error in {filename}. Row has {len(row_data)} elements. Writing ERROR marker.")
                    writer.writerow([f'ROW_SIZE_ERROR_{len(row_data)}'] * expected_length)

            except (FileNotFoundError, yaml.YAMLError, Exception) as e:
                # Handle any error by writing a full row of error markers
                error_marker = f'ERROR_{type(e).__name__}'
                print(f"Error processing {filename}: {error_marker}.")
                error_row = [error_marker] * expected_length
                writer.writerow(error_row)


            # Optional: Print progress every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{NUM_FILES} files...")

    print(f"\n✅ Bulk extraction complete! Data for {NUM_FILES} mechanisms saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    bulk_extract_nasa7_coefficients()
