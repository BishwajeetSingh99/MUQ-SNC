import os
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import multiprocessing
from check_derivative_constraint import check_derivative_positive_constraints
# -------------------------------
# PRS Loader + Evaluator
# -------------------------------
def build_polynomial_basis(X, order=2):
    """Builds 2nd-order polynomial basis (Legendre-like terms)."""
    N, d = X.shape
    B = []

    for n in range(N):
        row = [1.0]  # constant
        x = X[n, :]

        # linear terms
        row.extend(x)

        # quadratic terms (including cross terms)
        for i in range(d):
            for j in range(i, d):
                row.append(x[i] * x[j])

        B.append(row)
    return np.array(B)

def load_prs_models(prs_folder, valid_cases=None):
    """
    Load PRS coefficients + normalization for all cases.
    Only include cases in valid_cases (list) if provided.
    """
    models = {}
    for i in range(80):
        if valid_cases and i not in valid_cases:
            continue

        coef_file = os.path.join(prs_folder, f"responsecoef_case-{i}.csv")

        coeff = pd.read_csv(coef_file)["# Coefficients"].values
        norm_data = pd.read_csv(norm_file).values
        X_mean, X_std = norm_data[0], norm_data[1]

        models[i] = {"coeff": coeff, "mean": X_mean, "std": X_std}
    return models

def evaluate_prs(x, model, order=2):
    """Evaluate PRS model at given x."""
    X_mean, X_std, coeff = model["mean"], model["std"], model["coeff"]
    x_scaled = (x - X_mean) / X_std
    B = build_polynomial_basis(x_scaled.reshape(1, -1), order=order)
    return float(B @ coeff)

# -------------------------------
# Objective Function
# -------------------------------
def make_objective(prs_models, exp_values):
    """
    Build objective function:
    minimize sum of squared errors between experimental and PRS predictions.
    """

    def objective(individual):
        x = np.array(individual)
        preds = []
        for case_id, model in prs_models.items():
            y_pred = evaluate_prs(x, model)
            preds.append(y_pred)

        preds = np.array(preds)
        errors = exp_values - preds
        return (np.sum(errors**2),)  # single-objective minimization

    return objective

# -------------------------------
# Genetic Algorithm Setup
# -------------------------------
def run_ga(prs_folder="PRS_Results", ngen=20, pop_size=30, n_dim=5):
    # ---- Step 1: Load PRS models ----
    # Dummy: assume all 80 are valid, replace with your filtered list
    prs_models = load_prs_models(prs_folder, valid_cases=list(range(80))) ### this takes all the 80 cases by default
    #prs_models = load_prs_models(prs_folder, valid_cases=[0, 2, 5, 7, 11, ...])  ## specify the cases for which u want to run optimization..leave cases with mre >2%


    # Dummy experimental values (replace with your actual 80 numbers)
    #exp_values = np.array [] # paste the actual experimental values here for each case
    exp_values = np.random.rand(len(prs_models)) * 10.0

    # ---- Step 2: Build objective ----
    objective = make_objective(prs_models, exp_values)

    # ---- Step 3: GA definition ----
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Define parameter bounds (example: all between -2, 2)
    #BOUNDS_LOW = [-2.0] * n_dim
    #BOUNDS_HIGH = [2.0] * n_dim
    BOUNDS_LOW =np.array([-11.962742863218066, -2.11835162207328, -37.138235937223506, -453.4487916369844, -145.41486587267738, -2.846109700157162, -1.9309655450643661, -15.668817461107675, -0.6662585957426544, -5.839823452403772, -5.664693963658331, -9.529676109582008, -724.4069950105511, -2526.50942373263, -4863.280995936893, -0.5635363619483925, -2.5657152459906607, -10.660928476173272, -15.234519517268366, -5208.9187839999495, -10.18378905664729, -1.9458251644586013, -1138.6756779057905, -66.10055088064676, -2778.870003684608, -10.91044044249068, -4.146067767573143, -32.40270564770441, -9.700807631826494, -44.86379315412115, -2.4674872302164252, -1.677413522274072, -2838.4186551518496, -5.677778467391434, -13523.926034301867, -21.778361846735738, -178.32329764983444, -8.219373656473707, -57.65866807276478, -48.00838490049371, -2.5605802903619623, -1.4140417407254242, -3186.043547162469, -21.628145196512488, -20703.312876763543, -3.345053832964678, -28.072025828582397, -73.80017085264885, -3.674068312317091, -18.823565214387084, -2.37648133619456, -1.619294269129153, -3650.568953579708, -8.327062941180085, -14628.82967400944, -2.551643603260309, -3.745362857356871, -3.15122523958079, -13.312507782471535, -110.4566094598308, -11.728660461765903, -2.110227254806684, -22.882208924167276, -444.6227102971251, -349.06028560537254, -2.8819809831543513, -132.93518658899032, -1.7914781027299531, -134.1272763848271, -9.707642106016527, -3.099415140306358, -1.56863616099015, -3987.1258907537485, -24.376172535850912, -26282.420492639238, -3.495254025097583, -1.7767089835494576, -0.8235394002231864, -15.725527980696238, -392.6302027881407, -2.6279165577450287, -1.1048798297872113, -743.1421012374548, -5.0046518919472565, -6168.6044401804365, -23.839338968640018, -1.8638930399199602, -23.570127810301557, -1.0100546623713385, -7.286777046733126, -11.951720709004821, -2.1201256167223175, -36.92424385571537, -451.8785096592115, -140.4347142689185, -2.846071264402503, -2.257606734787236, -15.49209300432676, -0.6768211110172035, -5.835635260859572] )

    BOUNDS_HIGH =np.array( [2.7707509505433103, 28.6885692618926, 760.8281625827445, 30.438881938007864, 222.40888567210723, 10.873495239422002, 20.873738196711788, 6.879579403332835, 25.13158833417049, 31.62591986832099, 16.340409010888127, 33.42797599243832, 427.4776356272128, 3066.5454917700663, 6051.663735893007, 15.190419746133761, 4.4191140547344805, 235.4161038695377, 137.63153332820409, 0.4500858835486544, 2.3611944901278217, 23.460986243291103, 18.883896036287773, 2054.341503881178, 414.9162543671907, 2.419434638411956, 4.148206382004917, 14.909969196555858, 13.648293156822298, 41.97381754478761, 19.096882557002605, 45.2210610498377, 12.126612682150038, 8273.529988525694, 500.398218991086, 3.398692166509423, 3.9483960388426516, 302.1270913737865, 53.19066449420499, 17.322380017394956, 26.330948820289997, 79.26352317678098, 11.187502885598363, 11667.775101430747, 539.9783461391781, 25.357069264613948, 1.8215578871063103, 12.588921781884137, 178.69462906734168, 53.294699095383116, 23.84038086915374, 57.18564504316488, 15.069571011961818, 9778.748192424948, 713.7549810910845, 10.880507654572224, 3.5733957119033115, 4.246494615093695, 31.705165019073366, 85.36400347227848, 2.581569940844692, 27.864573142234345, 740.1559132374439, 53.546360339171386, 206.9548462225239, 16.615233263471755, 0.03257578749856131, 90.15721468709854, 12.703177961509748, 4.09186370122712, 42.67278517258837, 736.7744819186446, 4.543438121991287, 14048.343244266482, 627.8098473981039, 30.22832992559521, 152.97815313157298, 1257.4574215938994, 187.42581888446523, 3.5403398263279433, 15.0694595303151, 27.969605283707292, 14.167361999176713, 3367.945481142056, 338.7218195248067, 3.3380724252798664, 79.23617310265098, 1.9649189705446428, 278.4041200943391, 73.9471901048495, 2.7667649265381313, 28.55041053712844, 757.833800443525, 30.32382066415379, 221.80134932939757, 10.53551788365713, 19.827943039049877, 6.845555009959876, 24.296807314581233, 30.23320447429651] )
    #print(len(BOUNDS_LOW))
    #print(len(BOUNDS_HIGH))
    # Attribute generator
    for i in range(n_dim):
        toolbox.register(f"attr_float_{i}", np.random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i])

    # Individual and population
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [toolbox.__getattribute__(f"attr_float_{i}") for i in range(n_dim)],
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", objective)

    # ---- Step 4: Parallelization ----
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # ---- Step 5: Run GA ----
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3,
                        ngen=ngen, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    print("Best solution found:", hof[0])
    print("Best fitness:", hof[0].fitness.values)

    return hof[0]

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    best = run_ga(prs_folder="PRS_Results",ngen=20, pop_size=30, n_dim=100)






'''












import csv
import math
import argparse
import sys

def calculate_csv_stats(path, delimiter=','):
    """
    Reads a CSV file, calculates Min, Max, Mean, and Std Dev for all columns.
    """
    try:
        f = open(path, 'r', newline='', encoding='utf-8', errors='replace')
        reader = csv.reader(f, delimiter=delimiter)
    except FileNotFoundError:
        print(f"Error: File not found at path: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening file: {e}")
        sys.exit(1)

    ncols = None
    
    # Lists to store statistics calculated cumulatively
    min_vals, max_vals = [], []
    sums, sum_sqs = [], []
    counts = []

    for row_idx, row in enumerate(reader, start=1):
        # Initialize lists on the first row
        if ncols is None:
            ncols = len(row)
            min_vals = [float('inf')] * ncols
            max_vals = [-float('inf')] * ncols
            sums = [0.0] * ncols
            sum_sqs = [0.0] * ncols
            counts = [0] * ncols

        # Process each column in the row
        for col_idx, token in enumerate(row):
            if col_idx >= ncols:
                continue

            try:
                # Simple float conversion for machine-generated data
                val = float(token.strip())
            except ValueError:
                # Skip tokens that are not numbers (e.g., text, empty strings)
                continue
                
            # Update min/max
            if val < min_vals[col_idx]: min_vals[col_idx] = val
            if val > max_vals[col_idx]: max_vals[col_idx] = val
            
            # Update sums for mean/std dev calculation
            sums[col_idx] += val
            sum_sqs[col_idx] += val * val
            counts[col_idx] += 1
    
    f.close()

    # --- Final Calculation of Mean and Std Dev ---
    mean_vals = [None] * ncols
    std_vals = [None] * ncols
    
    for i in range(ncols):
        if counts[i] > 0:
            # Handle columns with no numeric data
            if min_vals[i] == float('inf'): min_vals[i] = None
            if max_vals[i] == -float('inf'): max_vals[i] = None

            # Calculate Mean
            mean = sums[i] / counts[i]
            mean_vals[i] = mean
            
            # Calculate Variance and Standard Deviation
            # Formula: E[x^2] - (E[x])^2
            variance = (sum_sqs[i] / counts[i]) - (mean * mean)
            
            # Handle tiny negative numbers due to float precision
            if variance < 0 and variance > -1e-12:
                std_vals[i] = 0.0
            elif variance >= 0:
                std_vals[i] = math.sqrt(variance)
            else:
                std_vals[i] = float('nan')
        else:
            min_vals[i] = None
            max_vals[i] = None

    return {
        'ncols': ncols,
        'min_values': min_vals,
        'max_values': max_vals,
        'mean_values': mean_vals,
        'std_values': std_vals,
    }

def main():
    p = argparse.ArgumentParser(description='Compute full statistics (min/max/mean/std) per column from a CSV file.')
    p.add_argument('file', help='Path to the input CSV file, e.g., dm_origional_params.csv')
    p.add_argument('--delimiter', '-d', default=',', help='(optional) delimiter, default is ","')
    args = p.parse_args()

    result = calculate_csv_stats(args.file, delimiter=args.delimiter)
    
    if result['ncols'] is None:
        print("No columns found.")
        return

    # --- Printing the Report ---
    print("\n" + "=" * 115)
    print(f"STATISTICAL DATA QUALITY REPORT | File: {args.file}")
    print("=" * 115)

    # Use a clean, fixed-width format for the header
    HEADER = f"{'COL':<5} | {'MIN VALUE':<20} | {'MAX VALUE':<20} | {'MEAN VALUE':<20} | {'STD DEV':<20} | {'ISSUE'}"
    print(HEADER)
    print("-" * 115)
    
    ISSUE_THRESHOLD = 1e-10

    for i in range(result['ncols']):
        col_idx = i + 1
        std_dev = result['std_values'][i]
        
        is_issue = False
        issue_text = ""
        
        # Check for issue: Standard Deviation is essentially zero
        if std_dev is not None and std_dev < ISSUE_THRESHOLD:
            is_issue = True
            issue_text = "!!! CONSTANT COLUMN (STD DEV ≈ 0) - MUST REMOVE FOR SCALING !!!" 
        elif std_dev is None:
            issue_text = "NON-NUMERIC/EMPTY DATA"

        # Format output string
        output = (
            f"{col_idx:<5} | "
            f"{str(result['min_values'][i]):<20} | "
            f"{str(result['max_values'][i]):<20} | "
            f"{str(result['mean_values'][i]):<20} | "
            f"{str(result['std_values'][i]):<20} | "
            f"{issue_text}"
        )
        
        # Highlight problematic columns using simple console color (ANSI codes)
        if is_issue:
            # Red highlight
            print(f"\033[41m{output}\033[0m") 
        else:
            print(output)

    print("=" * 115)


if __name__ == '__main__':
    main()
