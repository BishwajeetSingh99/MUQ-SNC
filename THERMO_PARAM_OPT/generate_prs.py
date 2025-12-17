'''
# use this for creating prs
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the PRS functions from the module
#import ResponseSurface_v2 as prs   # <-- module we created earlier
from ResponseSurface_v2  import run_prs_trials


def read_data(input_csv_path, output_folder_path):
    """
    Reads the input design matrix and all 80 simulation outputs.
    """
    X = pd.read_csv(input_csv_path, header=None).values  # numpy array
    
    outputs = {}
    for i in range(80):
        file_name = f"sim_data_case-{i}.lst"
        file_path = os.path.join(output_folder_path, file_name)
        df = pd.read_csv(file_path, sep=r"\s+", header=None, usecols=[1])
        outputs[f"case-{i}"] = df.values.flatten()
    
    return X, outputs


def process_case(case_number, X, y, outdir="PRS_Results"):
    """
    Runs PRS for a single case.
    Returns (case_number, max_err, mean_err).
    """
    coeff, (y_test, y_pred_test, max_err, mean_err) = prs.train_prs(
        X, y, order=2, case_index=case_number, outdir=outdir
    )
    prs.plot_parity(y_test, y_pred_test, case_number,
                    max_err, mean_err, outdir=os.path.join(outdir, "Plots"))
    return case_number, max_err, mean_err


def run_prs_for_all_cases(X, outputs, outdir="PRS_Results", n_jobs=8):
    """
    Runs PRS in parallel for all 80 cases.
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)

    results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for case_idx, y in outputs.items():
            case_number = int(case_idx.split("-")[1])
            futures.append(executor.submit(process_case, case_number, X, y, outdir))

        for f in as_completed(futures):
            case_number, max_err, mean_err = f.result()
            results.append([case_number, max_err, mean_err])
            print(f"[Case {case_number}] Max error: {max_err:.3f}% | Mean error: {mean_err:.3f}%")

    # Save summary CSV
    results = sorted(results, key=lambda x: x[0])
    np.savetxt(os.path.join(outdir, "PRS_error_summary.csv"),
               results, delimiter=",", header="Case,MaxError(%),MeanError(%)", comments="")


if __name__ == "__main__":
    input_csv = "/home/user/Desktop/thermo_param_opt_new/full_run_1/DesignMatrix.csv"
    output_folder = "/home/user/Desktop/thermo_param_opt_new/full_run_1/Opt/Data/Simulations"
    
    X, outputs = read_data(input_csv, output_folder)
    
    # Run in parallel (change n_jobs depending on CPU cores)
    run_prs_for_all_cases(X, outputs, outdir="PRS_Results", n_jobs=1)

    print("All PRS models trained successfully.")


'''






import os
import pandas as pd
import numpy as np

# Import the PRS trial runner
from  ResponseSurface_v2 import run_prs_trials

'''
def read_data(input_csv_path, output_folder_path, n_cases=80):
    """
    Reads the input design matrix (X) and all simulation outputs (Y).
    Returns:
        X : (N, d)
        Y : (N, n_cases)
    """
    # Design matrix
    X = pd.read_csv(input_csv_path, header=None).values  # numpy array
    # --------------------------------------------------------------------------
    # 🔍 DIAGNOSTIC STEP: Identify and print rows containing NaN (missing data)
    # --------------------------------------------------------------------------
    nan_rows = X_numeric.isnull().any(axis=1)
    nan_row_indices = X_numeric.index[nan_rows].tolist()

    if nan_row_indices:
        print(f"\n⚠️ Found missing data (NaN) in {len(nan_row_indices)} rows.")
        print("These correspond to the following mechanism indices (Row Number):")
        # Print the indices as a comma-separated list
        print(", ".join(map(str, nan_row_indices)))
        print("Mechanism 'i' corresponds to row index 'i'.\n")
    else:
        print("\n✅ No missing data (NaN) found in the input file.\n")
    # --------------------------------------------------------------------------

    # Collect outputs for all cases
    outputs = []
    for i in range(n_cases):
        file_name = f"sim_data_case-{i}.lst"
        file_path = os.path.join(output_folder_path, file_name)
        df = pd.read_csv(file_path, sep=r"\s+", header=None, usecols=[1])
        outputs.append(df.values.flatten())

    Y = np.column_stack(outputs)  # shape (N, n_cases)
    return X, Y
'''



def read_data(input_csv_path, output_folder_path, n_cases=80):
    """
    Reads the input design matrix (X) and all simulation outputs (Y).
    Handles missing data (NaN) in the design matrix X via mean imputation.
    
    Returns:
        X : (N, d) - Cleaned design matrix (NumPy array)
        Y : (N, n_cases) - Output matrix (NumPy array)
    """
    
    # 1. Load the Design Matrix into a Pandas DataFrame
    X_df = pd.read_csv(input_csv_path, header=None)
    
    # 2. Convert all columns to numeric, coercing non-numeric values ('N/A') to NaN
    X_numeric = X_df.apply(pd.to_numeric, errors='coerce')

    # --------------------------------------------------------------------------
    # 🔍 DIAGNOSTIC STEP: Identify and print rows containing NaN (missing data)
    # --------------------------------------------------------------------------
    nan_rows = X_numeric.isnull().any(axis=1)
    nan_row_indices = X_numeric.index[nan_rows].tolist()

    if nan_row_indices:
        print(f"\n⚠️ Found missing data (NaN) in {len(nan_row_indices)} rows.")
        print("These correspond to the following mechanism indices (Row Number):")
        # Print the indices as a comma-separated list
        # Row index 'i' corresponds to mechanism_i.yaml
        print(", ".join(map(str, nan_row_indices)))
        print(f"Mechanism 'i' corresponds to row index 'i' in '{os.path.basename(input_csv_path)}'.\n")
    else:
        print("\n✅ No missing data (NaN) found in the input coefficient file.\n")
    # --------------------------------------------------------------------------
    
    # 3. Impute the NaN values using the mean of the respective column
    # This step replaces 'N/A' entries with a numeric placeholder (the column mean)
    X_imputed = X_numeric.fillna(X_numeric.mean())

    # 4. Convert the clean DataFrame back to a NumPy array for modeling
    X = X_imputed.values
    
    # Collect outputs for all cases
    outputs = []
    
    # Ensure the output folder path exists before attempting to read files
    if not os.path.isdir(output_folder_path):
        print(f"❌ Error: Output folder path not found: {output_folder_path}")
        # Depending on your main script flow, you might raise an error or return here
        return X, None 

    for i in range(n_cases):
        file_name = f"sim_data_case-{i}.lst"
        file_path = os.path.join(output_folder_path, file_name)
        
        try:
            # Note: The 'sep=r"\s+"' is used correctly here to handle varying whitespace
            df = pd.read_csv(file_path, sep=r"\s+", header=None, usecols=[1])
            outputs.append(df.values.flatten())
        except FileNotFoundError:
            print(f"❌ Warning: Output file not found: {file_name}. Skipping case {i}.")
            # If you expect all cases to exist, you should handle this more robustly
            continue # Skip to the next case
        except pd.errors.EmptyDataError:
            print(f"❌ Warning: Output file is empty: {file_name}. Skipping case {i}.")
            continue
            

    # 5. Stack outputs if successful
    if outputs:
        # Check if all output arrays have the same length as X
        if all(len(arr) == X.shape[0] for arr in outputs):
            Y = np.column_stack(outputs)  # shape (N, n_cases)
        else:
            print("❌ Error: Output file lengths do not match the number of rows in the input matrix (X).")
            Y = np.empty((X.shape[0], 0)) # Return empty Y array
    else:
        print("❌ Error: Failed to load any output data (Y).")
        Y = np.empty((X.shape[0], 0))
        
    return X, Y



if __name__ == "__main__":
    input_csv = "/home/user/Desktop/thermo_param_opt_new/prs-and_opti_with_actual_param/dm_origional_params.csv"           #input data
    output_folder = "/home/user/Desktop/thermo_param_opt_new/prs-and_opti_with_actual_param/Opt/Data/Simulations"   # output data

    # Load all data
    X, Y = read_data(input_csv, output_folder, n_cases=80)

    # Run all PRS trials (creates trial_1 ... trial_8 folders automatically)
    run_prs_trials(X, Y)

    print("All PRS trials completed successfully.")

