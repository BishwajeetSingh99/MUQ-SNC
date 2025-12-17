
# ga_with_constraints_updated_v2.py
import os
import random
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from typing import List, Dict, Any, Tuple


# -------------------------------
# Polynomial Basis for PRS
# -------------------------------
import numpy as np
import os
def build_basis(X, basis_type="hybrid", degree=2):
    """
    Construct polynomial basis.
    basis_type = 'hybrid' or 'orthogonal'
    degree = 2, 2.5 (quadratic + x^3 independents)
    """
    n_samples, n_dim = X.shape
    Phi = [np.ones(n_samples)]

    # Univariate terms
    for j in range(n_dim):
        col = X[:, j]
        if basis_type == "orthogonal":
            # Legendre-like P1, P2, P3
            Phi.append(col)                          # P1
            Phi.append(0.5 * (3 * col**2 - 1))        # P2
            if degree == 2.5:
                Phi.append(0.5 * (5 * col**3 - 3 * col)) # P3
        else:
            # Hybrid → raw powers
            Phi.append(col)
            Phi.append(col**2)
            if degree == 2.5:
                Phi.append(col**3)

    # Quadratic cross terms
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            Phi.append(X[:, i] * X[:, j]) # same for both bases

    return np.vstack(Phi).T  # shape (n_samples, n_terms)

def evaluate_prs_all_cases(x, folder_path, valid_cases ,basis_type="hybrid", degree=2):
    """
    Evaluate all PRS response surfaces (cases 0..79) for a given raw input vector x.
    
    Parameters
    ----------
    x : array-like, shape (100,)
        Raw input vector (no normalization required).
    folder_path : str
        Path to folder containing responsecoef_case-<i>.csv files.
    basis_type : str
        Type of basis used during PRS creation ('hybrid' or 'orthogonal').
    degree : float
        Polynomial degree used (e.g., 2 or 2.5).
    
    Returns
    -------
    predictions : np.ndarray, shape (80,)
        Predicted PRS values for each case (0..79).
    """
    x = np.asarray(x).reshape(1, -1)  # ensure shape (1, n_params)
    
    # Build polynomial basis for the raw input
    Phi_x = build_basis(x, basis_type=basis_type, degree=degree).flatten()  # shape (n_basis_terms,)
    
    predictions = []
    
    #for case_idx in range(80):
    for case_idx in valid_cases:
        coef_file = os.path.join(folder_path, f"responsecoef_case-{case_idx}.csv")
        
        if not os.path.exists(coef_file):
            raise FileNotFoundError(f"Missing coefficient file: {coef_file}")
        
        # Load coefficients (they are for raw x already)
        coef = np.loadtxt(coef_file, delimiter=",", skiprows=1)  # skip header "Coefficients"
        
        # Handle both 1D and 2D load cases safely
        coef = np.ravel(coef)
        
        # Ensure basis and coefficient lengths match
        if Phi_x.shape[0] != coef.shape[0]:
            raise ValueError(
                f"Basis length ({Phi_x.shape[0]}) != coef length ({coef.shape[0]}) for case {case_idx}"
            )
        
        
        # Predicted value = dot product of basis and coefficients
        y_pred = np.dot(Phi_x, coef)
        predictions.append(y_pred)
    
    return np.array(predictions)


def cons_6_derivative_positive_definition(lenth_five_array, key):
    # Select temperature range
    if key == 'low':
        T_min, T_max = 300, 1000
    elif key == 'high':
        T_min, T_max = 1000, 5000
    else:
        raise ValueError("key must be 'low' or 'high'")

    # Generate 100 evenly spaced points in the temperature range
    linspace_points = np.linspace(T_min, T_max, 100)

    # Initialize an array to store the gradient at each point
    gradients = []
    for T in linspace_points:
        Theta_derivative = np.array([0, 1, 2*T, 3*T**2, 4*T**3])  # same as [0, T/T, ...]
        derivative = float(np.dot(Theta_derivative ,  lenth_five_array))
        gradients.append(derivative)
    # Ensure all gradient values are >= 0
    gradients = np.array(gradients)
    return gradients.min()
    
    
def derivative_positive_constraint(dm_zeta_vector):
    if len(dm_zeta_vector) != 100:
        raise AssertionError(f"Input vector must have exactly 100 elements, but found {len(dm_zeta_vector)}.")
    
    # Slicing and Assignment
    
    # --- Species 1: AR (Indices 0 to 9)
    AR_Low_dm_zeta  = dm_zeta_vector[0:5]
    AR_High_dm_zeta = dm_zeta_vector[5:10]

    AR_low  = cons_6_derivative_positive_definition(AR_Low_dm_zeta, "low" )
    AR_high = cons_6_derivative_positive_definition(AR_High_dm_zeta, "high" )

    # --- Species 2: H2 (Indices 10 to 19)
    H2_Low_dm_zeta  = dm_zeta_vector[10:15]
    H2_High_dm_zeta = dm_zeta_vector[15:20]

    H2_low  = cons_6_derivative_positive_definition(H2_Low_dm_zeta, "low" )
    H2_high = cons_6_derivative_positive_definition(H2_High_dm_zeta, "high" )
    # O is not required to satisfy the constraints
    
    # --- Species 3: O (Indices 20 to 29)  
    #O_Low_dm_zeta  = dm_zeta_vector[20:25]
    #O_High_dm_zeta = dm_zeta_vector[25:30]

    #O_low  = cons_6_derivative_positive_definition(O_Low_dm_zeta, "low" )
    #O_high = cons_6_derivative_positive_definition(O_High_dm_zeta, "high" )

    # --- Species 4: O2 (Indices 30 to 39)
    O2_Low_dm_zeta  = dm_zeta_vector[30:35]
    O2_High_dm_zeta = dm_zeta_vector[35:40]

    O2_low  = cons_6_derivative_positive_definition(O2_Low_dm_zeta, "low" )
    O2_high = cons_6_derivative_positive_definition(O2_High_dm_zeta, "high" )

    # --- Species 5: H2O (Indices 40 to 49)
    H2O_Low_dm_zeta  = dm_zeta_vector[40:45]
    H2O_High_dm_zeta = dm_zeta_vector[45:50]

    H2O_low  = cons_6_derivative_positive_definition(H2O_Low_dm_zeta, "low" )
    H2O_high = cons_6_derivative_positive_definition(H2O_High_dm_zeta, "high" )

    # --- Species 6: CO (Indices 50 to 59)
    CO_Low_dm_zeta  = dm_zeta_vector[50:55]
    CO_High_dm_zeta = dm_zeta_vector[55:60]

    CO_low  = cons_6_derivative_positive_definition(CO_Low_dm_zeta, "low" )
    CO_high = cons_6_derivative_positive_definition(CO_High_dm_zeta, "high" )

    # --- Species 7: C (Indices 60 to 69)
    C_Low_dm_zeta  = dm_zeta_vector[60:65]
    C_High_dm_zeta = dm_zeta_vector[65:70]

    C_low  = cons_6_derivative_positive_definition(C_Low_dm_zeta, "low" )
    C_high = cons_6_derivative_positive_definition(C_High_dm_zeta, "high" )

    # --- Species 8: HCO (Indices 70 to 79)
    HCO_Low_dm_zeta  = dm_zeta_vector[70:75]
    HCO_High_dm_zeta = dm_zeta_vector[75:80]

    HCO_low  = cons_6_derivative_positive_definition(HCO_Low_dm_zeta, "low" )
    HCO_high = cons_6_derivative_positive_definition(HCO_High_dm_zeta, "high" )

    # --- Species 9: OH* (Indices 80 to 89) (Using OHstar)
    OHstar_Low_dm_zeta  = dm_zeta_vector[80:85]
    OHstar_High_dm_zeta = dm_zeta_vector[85:90]

    OHstar_low  = cons_6_derivative_positive_definition(OHstar_Low_dm_zeta, "low" )
    OHstar_high = cons_6_derivative_positive_definition(OHstar_High_dm_zeta, "high" )

    # --- Species 10: H (Indices 90 to 99)
    H_Low_dm_zeta  = dm_zeta_vector[90:95]
    H_High_dm_zeta = dm_zeta_vector[95:100]

    H_low  = cons_6_derivative_positive_definition(H_Low_dm_zeta, "low" )
    H_high = cons_6_derivative_positive_definition(H_High_dm_zeta, "high" )
    gredient_vector = np.array([ AR_low, AR_high, H2_low, H2_high, O2_low, O2_high, H2O_low, H2O_high, CO_low , CO_high,C_low, C_high, HCO_low, HCO_high, 
                OHstar_low, OHstar_high,H_low,H_high ])
    
    return gredient_vector
    


def enforce_b0_from_a(block, T= 1000):  # block contains 10 zeta for a species (5 low, 5 "high" )..we need to transform zeta to 
    a = block[:5].copy()
    b = block[5:].copy()
    A_T = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
    new_b0 = A_T - (b[1]*T + b[2]*T**2 + b[3]*T**3 + b[4]*T**4)
    b[0] = new_b0
    return np.concatenate([a, b])


def enforce_scaling_of_b(block, T= 1000):
    a = block[:5].copy()
    b = block[5:].copy()
    A_T = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
    B_T = b[0] + b[1]*T + b[2]*T**2 + b[3]*T**3 + b[4]*T**4
    if abs(B_T) < 1e-12:
        B_T = 1e-6
    c = A_T / B_T
    if c <= 0:
        c = abs(c) + 1e-6
    b *= c
    return np.concatenate([a, b])


def worker_evaluate(ind, prs_coeff_folder_path,valid_cases, exp_values,  use_scaling, T_fixed=1000):
    """
    Enforce equality once per block, check derivative constraints once.
    If derivative check fails -> assign large penalty objective and return (no iterative repair).
    Otherwise evaluate PRS and return normal objective.
    Returns: obj, passed_bool, mins_vector, repaired_ind
    """
    ga_iteration = np.array(ind, dtype=float)
    # Copy and enforce equality once per block
    ga_itr_copy = ga_iteration.copy()
    n_blocks = len(ga_itr_copy) // 10  ## diving the itretion copy in blocks..each block with 10 params .... 10 blocks in total 
    for bidx in range(n_blocks):
        start = bidx * 10                               ## take a block and force equality at 1000using either enforce_scaling_of_b or enforce_b0_from_a
        block = ga_itr_copy[start:start+10]
        if use_scaling is True:                            # if use _scaling tag == "True" use enforce_scaling_of_b
            repaired = enforce_scaling_of_b(block , T_fixed)
            ga_itr_copy[start:start+10] = repaired
        else:                         #if use_scaling tag =="False " use enforce_b0_from_a
            repaired = enforce_b0_from_a(block , T_fixed)
            ga_itr_copy[start:start+10] = repaired            
    # with the above block ga_itr_copy is enforced equality at 1000 for each species
    # Check derivative constraint with the updated ga_itr_copy
    try:
        mins = derivative_positive_constraint(ga_itr_copy)
    except Exception:
        # If derivative check fails unexpectedly, treat as infeasible
        print("constraints can not be checked:: check the constraints definition")
        return 1e30, False, np.full(18 , -1.0), ga_itr_copy

    if not np.all(mins >= 0.0):
        # Constraint violated -> assign large penalty
        return 1e30, False, mins, ga_itr_copy
    else:
        # Constraints satisfied -> evaluate PRS models
        preds = []
        try:
            preds = evaluate_prs_all_cases(ga_itr_copy, prs_coeff_folder_path, valid_cases, basis_type="hybrid", degree=2) # returns the predicted values list for valid cases
        except Exception as e:
            print("PRS evalution not working..debug the code")
            raise RuntimeError ("PRS evalution failed") from e

        preds = np.array(preds, dtype=float)
        if not np.all(np.isfinite(preds)):
            obj = 1e30
        else:
            obj = float(np.sum((exp_values - preds) ** 2))

        return obj, True, mins, ga_itr_copy



# -------------------------------
# Main GA runner
# -------------------------------
def run_ga(prs_coeff_folder_path,  experimental_values,
           valid_cases=None, ngen=50, pop_size=50,
           use_scaling=True, outdir="GA_Results",
           n_workers=None, seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(outdir, exist_ok=True)
    # Files for iteration indices (kept for backward compatibility)
    iter_index_file = os.path.join(outdir, "GA_iteration_index.txt")
    open(iter_index_file, "w").close()

    # New files requested by user: iterations (vectors) and corresponding objectives
    iterations_file = os.path.join(outdir, "GA_iterations.txt")
    iter_obj_file = os.path.join(outdir, "GA_iteration_objective.txt")

    # Ensure files exist and are empty
    open(iterations_file, "w").close()
    open(iter_obj_file, "w").close()

    #prs_models = load_prs_models(prs_folder, valid_cases)  # seems like this is not required
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    n_dim = 100
    
    # --- BOUNDS DEFINITION ---
    # Note: These were at the end of the original function body. Keeping them here.
    BOUNDS_LOW = 1.25*np.array([-0.05901997357855526, -0.0005131463660410394, -7.0503926978563736e-06, -5.831025715752709e-10, -1.3651344841402066e-13, -0.19823411896541598, -0.0002045418793882784, -1.0726904167045634e-06, -2.9024923457250843e-11, -9.8859978919662e-15, -2.7635013472486407, 0.0011828060494265, -5.291645391715452e-05, 2.548107158497108e-09, -1.1303132867627044e-11, -2.9339787296809763, -0.0005603215828130713, -3.178530021608644e-06, -1.5951697774448935e-10, -3.431088754355534e-14, 0.6979050576119805, -0.0035533128741537683, -2.2210674319494823e-06, -6.506431831682806e-09, -7.791746399450946e-13, -0.31961866422562935, -0.0017694995073527593, -1.135290425245315e-06, -2.371153585240227e-10, -2.059759108194156e-14, -2.167555441589433, -0.0035333935006132246, -4.546440743903281e-05, -9.773541775305196e-09, -9.557689233910429e-12, -3.00210289075633, 0.00045200904329975285, -3.269943997432286e-06, 2.9689785205344345e-12, -3.337683479157974e-14, -5.1782904319024805, -0.0024228425386267254, -8.216111504852208e-05, -6.206099383438286e-09, -1.761039566049008e-11, -6.101018849287062, 0.0025140386966256085, -5.123506484032738e-06, 7.77790607778357e-11, -5.133680969859997e-14, -4.423732220905963, -0.0009077451683515995, -6.703745065403879e-05, 6.6531744856873e-10, -1.4319485142237179e-11, -1.6466348833587166, -0.0004666482121528086, -2.314480838279122e-06, -1.426859014911745e-10, -2.4252012938519928e-14, 0.029940203592325965, -0.0008230971654880571, -6.067058875600656e-06, -1.2712209747059886e-09, -5.567807440992424e-14, -0.8940890434354589, -0.00020429594543279106, -1.4547229870560184e-06, -2.5578812486634222e-11, -1.4205705681520924e-14, -7.273693375289578, -0.004144432272708414, -9.787337443514998e-05, -1.3870906955130086e-08, -2.033672875112691e-11, -6.520490587888853, 0.0020054814832559547, -5.913181697178305e-06, 2.0715996402928138e-11, -6.192653282410684e-14, 3.99198424, -0.00240106655, 4.61664033e-06, -3.87916306e-09, 1.36319502e-12, 2.83853033, 0.00110741289, -2.94000209e-07, 4.20698729e-11, -2.4228989e-15, -0.056662160399247874, -0.0005135760958828863, -7.0169202895777645e-06, -5.504435105503195e-10, -1.296176464062885e-13, -0.11436577555949512, -0.0002045391171189628, -1.0135671300900427e-06, -2.9024531485417525e-11, -9.277667803721699e-15])

    BOUNDS_HIGH = 1.25*np.array([3.0927074672860373, 0.008645112069102175, 4.0952038305842684e-07, 7.554115276078946e-10, 7.973596858252956e-13, 3.2062559122149112, 0.0030574971155194015, 1.2110542688830498e-07, 1.6816541404626026e-10, 2.422044282416377e-15, 4.115052264453702, 0.031232456879954802, -2.2231686232271187e-06, 3.9937660808150423e-08, -1.0951975672880342e-12, 3.1505147846843715, 0.00849429121913863, 5.805084592977758e-07, 5.39286036311066e-10, 1.37828436026476e-14, 3.7410406728481247, 0.004513837055367864, 6.897421052125972e-06, 2.12562850680969e-09, 2.757159817688963e-12, 3.1785755773448243, 0.0032358642340308825, 1.01257901113099e-06, 2.4714985946460297e-10, 1.9628111807604505e-14, 4.551250852830736, 0.02703711283037737, 1.0171751265839817e-05, 3.4040819727949993e-08, 4.480496441562348e-12, 4.700786419279214, 0.008926556414061945, -1.72990352279764e-08, 5.394840119308857e-10, -6.2195547546068804e-18, 5.110504009632016, 0.04841590901312735, 6.6861246557700965e-06, 6.211595959836517e-08, 3.348032635427277e-12, 3.835022711686568, 0.014204000853532297, -7.045909346343789e-07, 8.294392688855228e-10, -2.1644064275359344e-15, 4.377323255068366, 0.039566898983530344, 1.2938033879420954e-06, 5.0595314409075606e-08, 8.314668202733394e-13, 4.149562733336672, 0.006616661989689422, 5.061556264010208e-07, 3.628460102601451e-10, 1.2767174523682497e-14, 3.1098575270155324, 0.008109711362940216, 1.1354052244818877e-06, 1.9321269775774242e-10, 1.0213262438899558e-12, 3.2126156017409433, 0.004043609149430383, 1.1459708944882775e-07, 2.3475298235127033e-10, 2.0461260514740457e-15, 5.0736319043154685, 0.05852937632078628, 1.4209417678625374e-05, 7.285476251795792e-08, 6.30196826973643e-12, 5.127234632505352, 0.015982043159395456, -4.216606054349988e-07, 9.881738696156598e-10, 2.0570154367034628e-15, 3.99198424, -0.00240106655, 4.61664033e-06, -3.87916306e-09, 1.36319502e-12, 2.83853033, 0.00110741289, -2.94000209e-07, 4.20698729e-11, -2.4228989e-15, 3.0918547936842518, 0.008605352677426698, 4.098633322462044e-07, 7.505410420197408e-10, 7.942161939192896e-13, 3.206246374466255, 0.002903415223321245, 1.2110379140022964e-07, 1.5835450147428803e-10, 2.4220115735226168e-15])
    
    
    # --- Population Initialization using Bounds ---
    # Create the initial population by sampling uniformly between the bounds
    population = [np.random.uniform(low= BOUNDS_LOW, high = BOUNDS_HIGH, ) for _ in range(pop_size)]
    # Quick sanity check: basis length must match coefficient length for the first valid case
    if valid_cases is None or len(valid_cases) == 0:
        raise RuntimeError("valid_cases must be a non-empty list")
    # use the first individual (raw) to build a test basis vector
    test_x = population[0].copy().reshape(1, -1)
    Phi_test = build_basis(test_x, basis_type="hybrid", degree=2).flatten()

    # load first available coefficient file and compare lengths
    first_case = valid_cases[0]
    coef_test_file = os.path.join(prs_coeff_folder_path, f"responsecoef_case-{first_case}.csv")
    if not os.path.exists(coef_test_file):
        raise FileNotFoundError(f"Sanity check failed — missing coefficient file {coef_test_file}")

    coef_test = np.atleast_1d(np.ravel(np.genfromtxt(coef_test_file, delimiter=",", skip_header=1)))
    if Phi_test.shape[0] != coef_test.shape[0]:
        raise RuntimeError(
            f"Basis length ({Phi_test.shape[0]}) != coef length ({coef_test.shape[0]}) for case {first_case}. "
            "Check build_basis ordering/degree used at training and here."
        )
    
    # Initial Hall of Fame
    hall_of_fame = {"best_obj": 1e31, "best_ind": None}


    # --- Sanity: ensure experimental_values length matches valid_cases (user said they pass sliced list) ---
    experimental_values = np.asarray(experimental_values, dtype=float)
    if valid_cases is None or len(valid_cases) == 0:
        raise ValueError("valid_cases must be a non-empty list")
    if len(experimental_values) != len(valid_cases):
        raise ValueError(f"experimental_values length ({len(experimental_values)}) != len(valid_cases) ({len(valid_cases)})")

    # --- GA Evolution Loop ---
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for gen in range(ngen):

            # 1. Submit evaluation tasks to the pool
            # Pass: ind, prs_coeff_folder_path, valid_cases, experimental_values (already sliced), use_scaling, T_fixed
            futures = [
                executor.submit(
                    worker_evaluate,
                    ind,
                    prs_coeff_folder_path,
                    valid_cases,
                    experimental_values,
                    use_scaling,
                    1000.0,
                )
                for ind in population
            ]

            # 2. Collect results sequentially in the main process
            # Note: future.result() waits for the worker to complete.
            results = [future.result() for future in futures]

            # 3. Perform synchronous, safe writing in the main process
            current_generation_objectives = []

            # Use 'a' (append) mode to add to the files created earlier
            with open(iterations_file, "a") as f_ind, open(iter_obj_file, "a") as f_obj:
                for obj, passed, mins, repaired_ind in results:

                    # Write objective
                    f_obj.write(f"{obj}\n")

                    # Write individual vector (comma-separated)
                    ind_str = ",".join(map(str, repaired_ind.tolist()))
                    f_ind.write(f"{ind_str}\n")

                    current_generation_objectives.append(obj)

                    # Update Hall of Fame
                    if obj < hall_of_fame["best_obj"]:
                        hall_of_fame["best_obj"] = obj
                        hall_of_fame["best_ind"] = repaired_ind.copy()

            # --- (11) SAFETY CHECK: ensure all workers returned (run after writing all results) ---
            if len(current_generation_objectives) != len(population):
                raise RuntimeError(
                    f"Worker mismatch: got {len(current_generation_objectives)} results "
                    f"but expected {len(population)}. Check for crashed or hung processes."
                )

            # --- Placeholder for GA Mechanics (Selection, Crossover, Mutation) ---
            # Only create a new population if not on the last generation
            if gen < ngen - 1:  # Don't generate a new population after the last generation

                # --- Simple GA step (elitism + tournament + one-point crossover + gaussian mutation) ---
                elitism = 2
                mut_prob = 0.08      # probability to mutate a child
                mut_sigma = 0.005    # gaussian mutation scale

                # rank individuals by objective (ascending)
                sorted_idx = np.argsort(current_generation_objectives)
                new_population = []

                # keep elites (copy to avoid aliasing)
                for e in range(elitism):
                    new_population.append(population[sorted_idx[e]].copy())

                # --- (12) Robust tournament selection helper ---
                def tournament_select(pop, objs, k=3):
                    k_sel = min(k, len(pop))
                    idxs = np.random.choice(len(pop), size=k_sel, replace=False)
                    best_local = idxs[np.argmin([objs[i] for i in idxs])]
                    return pop[best_local]

                # make offspring until population full
                while len(new_population) < pop_size:
                    p1 = tournament_select(population, current_generation_objectives)
                    p2 = tournament_select(population, current_generation_objectives)
                    cx_pt = np.random.randint(1, n_dim)
                    child = np.concatenate([p1[:cx_pt], p2[cx_pt:]])
                    # mutation
                    if np.random.rand() < mut_prob:
                        child += np.random.normal(0, mut_sigma, size=n_dim)
                    # enforce bounds elementwise
                    child = np.minimum(np.maximum(child, BOUNDS_LOW), BOUNDS_HIGH)
                    new_population.append(child)

                population = new_population

            # End if gen < ngen - 1

            print(f"Generation {gen+1}/{ngen}: Best Objective = {hall_of_fame['best_obj']:.4e}")
    # end ProcessPoolExecutor






def main():

    experimental_values = [ 9539, 7503, 3144, 1539, 1064, 753, 332, 326, 6076, 1944, 684, 51, 19, 3123,1309, 874, 523, 390, 253, 3231, 1085, 115, 105, 84, 55, 4340, 3310,1020,6497, 5478, 3950, 2349, 1873, 1349, 1085, 920, 802, 615, 477, 302, 398, 2865, 1051, 559, 329, 197, 167, 89, 75, 56, 470, 3577, 1575, 1069, 1111, 725, 612, 498, 10776, 355, 1150, 763, 455, 364, 294, 179, 102, 85, 87]

    # give actual experimental values for all 80 cases   
    # list of cases for which the prs accuracy is good..out of 80 11 PRS are not good.. [21,22,23,43,45,55,57,72,74,75,79]      
    valid_cases=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,46,47,48,49,50,51,52,53,54,
    56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,73,76,77,78]  
    if len(valid_cases) != len(experimental_values):
        raise AssertionError("valid prs and experimental values mismatch")

    best_ind, best_obj = run_ga(
        prs_coeff_folder_path="/home/user/Desktop/thermo_param_opt_new/prs-and_opti_with_actual_param/PRS_SURROGATE_MODEL/trial_3_split_90_10",   # <-- your PRS folder
        experimental_values=experimental_values,
        valid_cases=valid_cases,
        ngen=20000,
        pop_size=300,
        use_scaling=True,
        outdir="GA_Run",
        n_workers=4,
        seed=42
    )
    print("Best objective:", best_obj)


if __name__ == "__main__":  ### use scaling-= true (uses C scalinf method, false -use hardcore b0 formula)
    main()























    '''
min_values list (column order):
[-0.05901997357855526, -0.0005131463660410394, -7.0503926978563736e-06, -5.831025715752709e-10, -1.3651344841402066e-13, -0.19823411896541598, -0.0002045418793882784, -1.0726904167045634e-06, -2.9024923457250843e-11, -9.8859978919662e-15, -2.7635013472486407, 0.0011828060494265, -5.291645391715452e-05, 2.548107158497108e-09, -1.1303132867627044e-11, -2.9339787296809763, -0.0005603215828130713, -3.178530021608644e-06, -1.5951697774448935e-10, -3.431088754355534e-14, 0.6979050576119805, -0.0035533128741537683, -2.2210674319494823e-06, -6.506431831682806e-09, -7.791746399450946e-13, -0.31961866422562935, -0.0017694995073527593, -1.135290425245315e-06, -2.371153585240227e-10, -2.059759108194156e-14, -2.167555441589433, -0.0035333935006132246, -4.546440743903281e-05, -9.773541775305196e-09, -9.557689233910429e-12, -3.00210289075633, 0.00045200904329975285, -3.269943997432286e-06, 2.9689785205344345e-12, -3.337683479157974e-14, -5.1782904319024805, -0.0024228425386267254, -8.216111504852208e-05, -6.206099383438286e-09, -1.761039566049008e-11, -6.101018849287062, 0.0025140386966256085, -5.123506484032738e-06, 7.77790607778357e-11, -5.133680969859997e-14, -4.423732220905963, -0.0009077451683515995, -6.703745065403879e-05, 6.6531744856873e-10, -1.4319485142237179e-11, -1.6466348833587166, -0.0004666482121528086, -2.314480838279122e-06, -1.426859014911745e-10, -2.4252012938519928e-14, 0.029940203592325965, -0.0008230971654880571, -6.067058875600656e-06, -1.2712209747059886e-09, -5.567807440992424e-14, -0.8940890434354589, -0.00020429594543279106, -1.4547229870560184e-06, -2.5578812486634222e-11, -1.4205705681520924e-14, -7.273693375289578, -0.004144432272708414, -9.787337443514998e-05, -1.3870906955130086e-08, -2.033672875112691e-11, -6.520490587888853, 0.0020054814832559547, -5.913181697178305e-06, 2.0715996402928138e-11, -6.192653282410684e-14, 3.99198424, -0.00240106655, 4.61664033e-06, -3.87916306e-09, 1.36319502e-12, 2.83853033, 0.00110741289, -2.94000209e-07, 4.20698729e-11, -2.4228989e-15, -0.056662160399247874, -0.0005135760958828863, -7.0169202895777645e-06, -5.504435105503195e-10, -1.296176464062885e-13, -0.11436577555949512, -0.0002045391171189628, -1.0135671300900427e-06, -2.9024531485417525e-11, -9.277667803721699e-15]

max_values list (column order):
[3.0927074672860373, 0.008645112069102175, 4.0952038305842684e-07, 7.554115276078946e-10, 7.973596858252956e-13, 3.2062559122149112, 0.0030574971155194015, 1.2110542688830498e-07, 1.6816541404626026e-10, 2.422044282416377e-15, 4.115052264453702, 0.031232456879954802, -2.2231686232271187e-06, 3.9937660808150423e-08, -1.0951975672880342e-12, 3.1505147846843715, 0.00849429121913863, 5.805084592977758e-07, 5.39286036311066e-10, 1.37828436026476e-14, 3.7410406728481247, 0.004513837055367864, 6.897421052125972e-06, 2.12562850680969e-09, 2.757159817688963e-12, 3.1785755773448243, 0.0032358642340308825, 1.01257901113099e-06, 2.4714985946460297e-10, 1.9628111807604505e-14, 4.551250852830736, 0.02703711283037737, 1.0171751265839817e-05, 3.4040819727949993e-08, 4.480496441562348e-12, 4.700786419279214, 0.008926556414061945, -1.72990352279764e-08, 5.394840119308857e-10, -6.2195547546068804e-18, 5.110504009632016, 0.04841590901312735, 6.6861246557700965e-06, 6.211595959836517e-08, 3.348032635427277e-12, 3.835022711686568, 0.014204000853532297, -7.045909346343789e-07, 8.294392688855228e-10, -2.1644064275359344e-15, 4.377323255068366, 0.039566898983530344, 1.2938033879420954e-06, 5.0595314409075606e-08, 8.314668202733394e-13, 4.149562733336672, 0.006616661989689422, 5.061556264010208e-07, 3.628460102601451e-10, 1.2767174523682497e-14, 3.1098575270155324, 0.008109711362940216, 1.1354052244818877e-06, 1.9321269775774242e-10, 1.0213262438899558e-12, 3.2126156017409433, 0.004043609149430383, 1.1459708944882775e-07, 2.3475298235127033e-10, 2.0461260514740457e-15, 5.0736319043154685, 0.05852937632078628, 1.4209417678625374e-05, 7.285476251795792e-08, 6.30196826973643e-12, 5.127234632505352, 0.015982043159395456, -4.216606054349988e-07, 9.881738696156598e-10, 2.0570154367034628e-15, 3.99198424, -0.00240106655, 4.61664033e-06, -3.87916306e-09, 1.36319502e-12, 2.83853033, 0.00110741289, -2.94000209e-07, 4.20698729e-11, -2.4228989e-15, 3.0918547936842518, 0.008605352677426698, 4.098633322462044e-07, 7.505410420197408e-10, 7.942161939192896e-13, 3.206246374466255, 0.002903415223321245, 1.2110379140022964e-07, 1.5835450147428803e-10, 2.4220115735226168e-15]
  
  '''
