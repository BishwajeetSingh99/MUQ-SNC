'''
# this uses othogonal and monomial both:: note only orthogonal worked better.. this is used on dm zeta entries 
import os
import math
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.polynomial.legendre import legval

# -------------------------------
# Utility: Build 2nd-order basis
# -------------------------------
def build_polynomial_basis(X, order=2, use_legendre=True):
    """
    Construct polynomial basis (monomials + optional Legendre orthogonalization).
    X: input array (N, d)
    order: polynomial order (default 2)
    use_legendre: if True, apply Legendre polynomials instead of raw powers
    Returns: Design matrix B of shape (N, n_terms)
    """
    N, d = X.shape
    B = []

    for n in range(N):
        row = [1.0]  # constant term
        x = X[n, :]

        # Linear terms
        for xi in x:
            if use_legendre:
                row.append(legval(xi, [0, 1]))  # P1(x) = x
            else:
                row.append(xi)

        # Quadratic terms (cross terms included)
        for i in range(d):
            for j in range(i, d):
                if use_legendre:
                    # Legendre P2(x) = (3x^2 - 1)/2
                    if i == j:
                        row.append(legval(x[i], [0, 0, 1]))  
                    else:
                        row.append(x[i] * x[j])  # cross term stays raw
                else:
                    row.append(x[i] * x[j])

        B.append(row)

    return np.array(B)


# -------------------------------
# Train PRS
# -------------------------------
def train_prs(X, Y, order=2, case_index=0, outdir="ResponseSurface"):
    """
    Train polynomial response surface of given order.
    Saves coefficients and returns trained model.
    """
    # Normalize inputs for stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.1, random_state=42
    )

    # Build design matrix
    B_train = build_polynomial_basis(X_train, order=order, use_legendre=True)
    B_test = build_polynomial_basis(X_test, order=order, use_legendre=True)

    # Solve least squares
    Q, R = np.linalg.qr(B_train)
    y_proj = np.dot(Q.T, y_train)
    coeff, _, _, _ = np.linalg.lstsq(R, y_proj, rcond=None)

    # Predict
    y_pred_train = B_train @ coeff
    y_pred_test = B_test @ coeff

    # Errors
    error_test = np.abs(y_test - y_pred_test)
    rel_error_test = (error_test / np.abs(y_test)) * 100
    max_err = np.max(rel_error_test)
    mean_err = np.mean(rel_error_test)

    # Save coefficients
    os.makedirs(outdir, exist_ok=True)
    coef_file = os.path.join(outdir, f"responsecoef_case-{case_index}.csv")
    np.savetxt(coef_file, coeff, delimiter=",", header="Coefficients")

    # Save normalization (needed for optimization later)
    norm_file = os.path.join(outdir, f"normalization_case-{case_index}.csv")
    norm_data = np.vstack([X_mean, X_std])
    np.savetxt(norm_file, norm_data, delimiter=",", header="Mean,Std")

    return coeff, (y_test, y_pred_test, max_err, mean_err)


# -------------------------------
# Evaluate PRS
# -------------------------------
def evaluate_prs(x, coeff, X_mean, X_std, order=2):
    """
    Evaluate polynomial response surface at a single point.
    """
    # Normalize
    x_scaled = (x - X_mean) / X_std
    B = build_polynomial_basis(x_scaled.reshape(1, -1), order=order, use_legendre=True)
    return float(B @ coeff)


# -------------------------------
# Plot Parity
# -------------------------------
def plot_parity(y_true, y_pred, case_index, max_err, mean_err, outdir="Plots"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.7)

    plt.xlabel("Direct Simulation")
    plt.ylabel("Response Surface estimation")

    ax.plot(y_true, y_pred, "k.", ms=6,
            label=f"Testing (max error = {max_err:.3f}%, mean error = {mean_err:.3f}%)")

    min_val, max_val = min(y_true), max(y_true)
    ax.plot([min_val, max_val], [min_val, max_val], "r-", label="Parity line")

    plt.legend(loc="upper left")
    plt.savefig(os.path.join(outdir, f"Parity_plot_case_{case_index}_TESTING.pdf"),
                bbox_inches="tight")
    plt.close()


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example: assume X and outputs are already loaded (as per your previous script)
    # Here I just create dummy data for illustration
    N, d = 1000, 5
    X = np.random.rand(N, d)
    Y = X[:, 0] + 0.5*X[:, 1]**2 - 0.2*X[:, 2]*X[:, 3] + np.random.normal(0, 0.01, N)

    coeff, (y_test, y_pred_test, max_err, mean_err) = train_prs(X, Y, order=2, case_index=0)

    plot_parity(y_test, y_pred_test, case_index=0,
                max_err=max_err, mean_err=mean_err)
'''

# expermiment with diferent combinations of prs setting to find out optimal setup
# again used on dm zeta entries



'''
# responsesurface_trials.py
import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# Basis Generator
# -------------------------------
def build_basis(X, basis_type="hybrid", degree=2.5):
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
            # Legendre P1, P2, P3
            Phi.append(col)                               # P1
            Phi.append(0.5 * (3 * col**2 - 1))            # P2
            if degree == 2.5:
                Phi.append(0.5 * (5 * col**3 - 3 * col))  # P3
        else:
            # Hybrid → raw powers
            Phi.append(col)
            Phi.append(col**2)
            if degree == 2.5:
                Phi.append(col**3)

    # Quadratic cross terms
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            Phi.append(X[:, i] * X[:, j])  # same for both bases

    return np.vstack(Phi).T  # shape (n_samples, n_terms)


# -------------------------------
# Plot Parity
# -------------------------------
def plot_parity(y_true, y_pred, case_index, max_err, mean_err, outdir="Plots"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.7)

    plt.xlabel("Direct Simulation")
    plt.ylabel("Response Surface estimation")

    ax.plot(
        y_true,
        y_pred,
        "k.",
        ms=6,
        label=f"Testing (max error = {max_err:.3f}%, mean error = {mean_err:.3f}%)",
    )

    min_val, max_val = min(y_true), max(y_true)
    ax.plot([min_val, max_val], [min_val, max_val], "r-", label="Parity line")

    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(outdir, f"Parity_plot_case_{case_index}_TESTING.pdf"),
        bbox_inches="tight",
    )
    plt.close()


# -------------------------------
# Run Multiple PRS Trials
# -------------------------------
def run_prs_trials(X, Y):
    """
    Run multiple PRS setups (basis + degree + regression) and save results.
    X : input array (N,d)
    Y : output array (N,) or (N,k)
    """
    trials = [
        {"basis": "hybrid", "degree": 2, "reg": LinearRegression(), "name": "trial_1"},
        {"basis": "orthogonal", "degree": 2, "reg": LinearRegression(), "name": "trial_2"},
        {"basis": "hybrid", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_3"},
        {"basis": "orthogonal", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_4"},
        {"basis": "hybrid", "degree": 2, "reg": Lasso(alpha=1e-4), "name": "trial_5"},
        {"basis": "orthogonal", "degree": 2, "reg": Lasso(alpha=1e-4), "name": "trial_6"},
        {"basis": "hybrid", "degree": 2.5, "reg": Ridge(alpha=1e-3), "name": "trial_7"},
        {"basis": "orthogonal", "degree": 2.5, "reg": Ridge(alpha=1e-3), "name": "trial_8"},
    ]

    # Split train/test once globally
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.1, random_state=42
    )

    for t in trials:
        folder = t["name"]
        os.makedirs(folder, exist_ok=True)

        # --- Build basis ---
        Phi_train = build_basis(X_train, t["basis"], t["degree"])
        Phi_test = build_basis(X_test, t["basis"], t["degree"])

        # --- Fit model ---
        model = t["reg"]
        model.fit(Phi_train, y_train)
        y_pred = model.predict(Phi_test)

        # --- Save coefficients ---
        coef_file = os.path.join(folder, "responsecoef_case-0.csv")
        np.savetxt(coef_file, model.coef_.reshape(1, -1), delimiter=",", header="Coefficients")

        # --- Save normalization ---
        norm_file = os.path.join(folder, "normalization_case-0.csv")
        norm_data = np.vstack([X_mean, X_std])
        np.savetxt(norm_file, norm_data, delimiter=",", header="Mean,Std")

        # --- Error metrics ---
        errors = np.abs((y_pred - y_test) / y_test) * 100
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)

        max_err = np.max(errors, axis=0)   # per output case
        min_err = np.min(errors, axis=0)
        mean_err = np.mean(errors, axis=0)

        count_above_2 = np.sum(max_err > 2)
        count_above_5 = np.sum(max_err > 5)

        # --- Save accuracy file ---
        with open(os.path.join(folder, "prs_accuracy_file.txt"), "w") as f:
            for k in range(len(max_err)):
                f.write(
                    f"Case {k+1}: min={min_err[k]:.3f}%, max={max_err[k]:.3f}%, mean={mean_err[k]:.3f}%\n"
                )
            f.write("\n")
            f.write(f"Count max_err >2%: {count_above_2}\n")
            f.write(f"Count max_err >5%: {count_above_5}\n")

        # --- Save plots (per output case) ---
        for k in range(y_test.shape[1] if y_test.ndim > 1 else 1):
            y_true_k = y_test[:, k] if y_test.ndim > 1 else y_test
            y_pred_k = y_pred[:, k] if y_pred.ndim > 1 else y_pred
            plot_parity(
                y_true_k,
                y_pred_k,
                case_index=k,
                max_err=max_err[k],
                mean_err=mean_err[k],
                outdir=folder,
            )


'''

'''
## this is for using prs directly on the actual dm parameters and not zeta..here the params are P0 + l zeta
import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# Basis Generator
# -------------------------------
def build_basis(X, basis_type="hybrid", degree=2.5):
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
            # Legendre P1, P2, P3
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


# -------------------------------
# Plot Parity
# -------------------------------
def plot_parity(y_true, y_pred, case_index, max_err, mean_err, outdir="Plots"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.7)

    plt.xlabel("Direct Simulation")
    plt.ylabel("Response Surface estimation")

    ax.plot(
        y_true,
        y_pred,
        "k.",
        ms=6,
        label=f"Testing (max error = {max_err:.3f}%, mean error = {mean_err:.3f}%)",
    )

    min_val, max_val = min(y_true), max(y_true)
    ax.plot([min_val, max_val], [min_val, max_val], "r-", label="Parity line")

    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(outdir, f"Parity_plot_case_{case_index}_TESTING.pdf"),
        bbox_inches="tight",
    )
    plt.close()


# -------------------------------
# Run Multiple PRS Trials
# -------------------------------
def run_prs_trials(X, Y):
    """
    Run multiple PRS setups (basis + degree + regression) and save results.
    X : input array (N,d)
    Y : output array (N,) or (N,k)
    """
    trials = [
        {"basis": "hybrid", "degree": 2, "reg": LinearRegression(), "name": "trial_1"},
        {"basis": "orthogonal", "degree": 2, "reg": LinearRegression(), "name": "trial_2"},
        {"basis": "hybrid", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_3"}, ## seesm best
        {"basis": "orthogonal", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_4"},  # seems best
        {"basis": "hybrid", "degree": 2, "reg": Lasso(alpha=1e-4), "name": "trial_5"},
        {"basis": "orthogonal", "degree": 2, "reg": Lasso(alpha=1e-4), "name": "trial_6"},
        {"basis": "hybrid", "degree": 2.5, "reg": Ridge(alpha=1e-3), "name": "trial_7"},
        {"basis": "orthogonal", "degree": 2.5, "reg": Ridge(alpha=1e-3), "name": "trial_8"},
    ]

    # Split train/test once globally
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    # --- START OF MODIFICATION FOR SAFE SCALING ---
    # Define a small numerical threshold for zero detection
    STD_SAFE_THRESHOLD = 1e-10

    # Create a safe version of X_std for the division.
    # If a column's standard deviation is less than the threshold (meaning it's constant), 
    # we replace it with 1.0. This makes the column's scaled value effectively 0.
    X_safe_std = np.where(X_std < STD_SAFE_THRESHOLD, 1.0, X_std)

    # Perform the scaling using the safe standard deviation
    X_scaled = (X - X_mean) / X_safe_std
    # --- END OF MODIFICATION ---

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.1, random_state=42
    )

    for t in trials:
        folder = t["name"]
        os.makedirs(folder, exist_ok=True)

        # --- Build basis ---
        Phi_train = build_basis(X_train, t["basis"], t["degree"])
        Phi_test = build_basis(X_test, t["basis"], t["degree"])

        # --- Fit model ---
        model = t["reg"]
        model.fit(Phi_train, y_train)
        y_pred = model.predict(Phi_test)

        # --- Save coefficients (UNCHANGED) ---
        coef_file = os.path.join(folder, "responsecoef_case-0.csv")
        np.savetxt(coef_file, model.coef_.reshape(1, -1), delimiter=",", header="Coefficients")

        # --- Save normalization (UNCHANGED, uses original X_std with potential zeros) ---
        norm_file = os.path.join(folder, "normalization_case-0.csv")
        norm_data = np.vstack([X_mean, X_std]) # X_std here preserves the actual statistics
        np.savetxt(norm_file, norm_data, delimiter=",", header="Mean,Std")

        # --- Error metrics (UNCHANGED) ---
        errors = np.abs((y_pred - y_test) / y_test) * 100
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)

        max_err = np.max(errors, axis=0)   # per output case
        min_err = np.min(errors, axis=0)
        mean_err = np.mean(errors, axis=0)

        count_above_2 = np.sum(max_err > 2)
        count_above_5 = np.sum(max_err > 5)

        # --- Save accuracy file (UNCHANGED) ---
        with open(os.path.join(folder, "prs_accuracy_file.txt"), "w") as f:
            for k in range(len(max_err)):
                f.write(
                    f"Case {k+1}: min={min_err[k]:.3f}%, max={max_err[k]:.3f}%, mean={mean_err[k]:.3f}%\n"
                )
            f.write("\n")
            f.write(f"Count max_err >2%: {count_above_2}\n")
            f.write(f"Count max_err >5%: {count_above_5}\n")

        # --- Save plots (per output case) (UNCHANGED) ---
        for k in range(y_test.shape[1] if y_test.ndim > 1 else 1):
            y_true_k = y_test[:, k] if y_test.ndim > 1 else y_test
            y_pred_k = y_pred[:, k] if y_pred.ndim > 1 else y_pred
            plot_parity(
                y_true_k,
                y_pred_k,
                case_index=k,
                max_err=max_err[k],
                mean_err=mean_err[k],
                outdir=folder,
            )
            '''









import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# Basis Generator (same as yours)
# -------------------------------
def build_basis(X, basis_type="hybrid", degree=2.5):
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


# -------------------------------
# Plot Parity (same as yours)
# -------------------------------
def plot_parity(y_true, y_pred, case_index, max_err, mean_err, outdir="Plots"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.7)

    plt.xlabel("Direct Simulation")
    plt.ylabel("Response Surface estimation")

    ax.plot(
        y_true,
        y_pred,
        "k.",
        ms=6,
        label=f"Search Iteration (max error = {max_err:.3f}%, mean error = {mean_err:.3f}%)",
    )

    min_val, max_val = min(y_true), max(y_true)
    ax.plot([min_val, max_val], [min_val, max_val], "r-", label="Parity line")

    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(outdir, f"Parity_plot_case_{case_index}_TESTING.pdf"),
        bbox_inches="tight",
    )
    plt.close()


# -------------------------------
# Runner for the selected two trials
# -------------------------------
'''
## this def is saving the normalized coeffs.. 
def run_prs_trials(X, Y, random_state=42):
    """
    Runs only the two Ridge trials:
      - trial_3: hybrid, degree=2, Ridge(alpha=1e-3)
      - trial_4: orthogonal, degree=2, Ridge(alpha=1e-3)

    Runs each trial for two train/test splits: 90/10 and 80/20.
    Saves responsecoef_case-<i>.csv and normalization_case-<i>.csv for every response case (0..n_cases-1)
    EXACT file formats mimic the originals.
    Writes a single prs_accuracy_file.txt per folder containing info for all cases.
    """
    # Trials (only the two you specified)
    trials = [
        {"basis": "hybrid", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_3"},
        {"basis": "orthogonal", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_4"},
    ]

    # Two split setups (test_size values)
    splits = [
        {"name": "split_90_10", "test_size": 0.1},
        {"name": "split_80_20", "test_size": 0.2},
    ]

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Ensure Y is 2D: (N, n_cases)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_cases = Y.shape[1]

    # Compute global mean/std (keeps the same behavior as your current script)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # Safe scaling for performing fitting (same as your current modified code)
    STD_SAFE_THRESHOLD = 1e-10
    X_safe_std = np.where(X_std < STD_SAFE_THRESHOLD, 1.0, X_std)
    X_scaled = (X - X_mean) / X_safe_std

    # Loop over trials and splits
    for t in trials:
        for sp in splits:
            folder = f"{t['name']}_{sp['name']}"
            os.makedirs(folder, exist_ok=True)

            # perform a single global split according to this setup (deterministic)
            X_train, X_test, y_train_all, y_test_all = train_test_split(
                X_scaled, Y, test_size=sp["test_size"], random_state=random_state
            )

            # Build basis once per split (train/test)
            Phi_train = build_basis(X_train, t["basis"], t["degree"])
            Phi_test = build_basis(X_test, t["basis"], t["degree"])

            coef_folder = folder  # coefficients go into this folder (same as before)

            # --- Initialize accuracy tracking ---
            accuracy_lines = []
            count_above_2_total = 0
            count_above_5_total = 0

            # Loop over each response case
            for case_idx in range(n_cases):
                # prepare target vectors for this case
                y_train = y_train_all[:, case_idx]
                y_test = y_test_all[:, case_idx]

                model = Ridge(alpha=1e-3)  # explicitly use Ridge as in trial definition
                # fit on features (Phi_train) and single-target y_train
                model.fit(Phi_train, y_train)
                y_pred = model.predict(Phi_test)

                # --- Save coefficients (same filename format as before) ---
                coef_file = os.path.join(coef_folder, f"responsecoef_case-{case_idx}.csv")
                np.savetxt(coef_file, model.coef_.reshape(1, -1), delimiter=",", header="Coefficients")

                # --- Save normalization exactly as before (Mean,Std stacked) ---
                norm_file = os.path.join(coef_folder, f"normalization_case-{case_idx}.csv")
                norm_data = np.vstack([X_mean, X_std])
                np.savetxt(norm_file, norm_data, delimiter=",", header="Mean,Std")

                # --- Error metrics (same computation as before) ---
                errors = np.abs((y_pred - y_test) / y_test) * 100
                if errors.ndim == 1:
                    errors = errors.reshape(-1, 1)

                max_err = np.max(errors, axis=0)
                min_err = np.min(errors, axis=0)
                mean_err = np.mean(errors, axis=0)

                # Update total counts
                if max_err[0] > 2:
                    count_above_2_total += 1
                if max_err[0] > 5:
                    count_above_5_total += 1

                # Store formatted line for this case
                accuracy_lines.append(
                    f"Case {case_idx+1}: min={min_err[0]:.3f}%, max={max_err[0]:.3f}%, mean={mean_err[0]:.3f}%\n"
                )

                # --- Save parity plot (same filename scheme and style) ---
                plot_parity(
                    y_true=y_test,
                    y_pred=y_pred,
                    case_index=case_idx,
                    max_err=max_err[0],
                    mean_err=mean_err[0],
                    outdir=coef_folder,
                )

            # --- Write a single aggregated accuracy file per folder ---
            acc_file = os.path.join(coef_folder, "prs_accuracy_file.txt")
            with open(acc_file, "w") as f:
                for line in accuracy_lines:
                    f.write(line)
                f.write("\n")
                f.write(f"Count max_err >2%: {count_above_2_total}\n")
                f.write(f"Count max_err >5%: {count_above_5_total}\n")

            print(f"Finished {t['name']} with {sp['name']} — results in folder: {folder}")

    print("All selected trials and splits completed.")


# Example usage:
# run_selected_prs(X, Y)
# where X.shape == (N, d) and Y.shape == (N, 80)
'''

# this is saving the de normalized coeffs..which can be used directly to the GA without the need for denormalization
def run_prs_trials(X, Y, random_state=42):
    """
    Runs only the two Ridge trials:
      - trial_3: hybrid, degree=2, Ridge(alpha=1e-3)
      - trial_4: orthogonal, degree=2, Ridge(alpha=1e-3)

    Runs each trial for two train/test splits: 90/10 and 80/20.
    Saves responsecoef_case-<i>.csv and normalization_case-<i>.csv for every response case (0..n_cases-1)
    EXACT file formats mimic the originals.
    Writes a single prs_accuracy_file.txt per folder containing info for all cases.
    """
    # Trials (only the two you specified)
    trials = [
        {"basis": "hybrid", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_3"},
        #{"basis": "orthogonal", "degree": 2, "reg": Ridge(alpha=1e-3), "name": "trial_4"},
    ]

    # Two split setups (test_size values)
    splits = [
        #{"name": "split_90_10", "test_size": 0.1},
        {"name": "split_80_20", "test_size": 0.2},
    ]

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Ensure Y is 2D: (N, n_cases)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_cases = Y.shape[1]

    # Compute global mean/std (keeps the same behavior as your current script)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # Safe scaling for performing fitting (same as your current modified code)
    STD_SAFE_THRESHOLD = 1e-10
    X_safe_std = np.where(X_std < STD_SAFE_THRESHOLD, 1.0, X_std)
    X_scaled = (X - X_mean) / X_safe_std

    # Loop over trials and splits
    for t in trials:
        for sp in splits:
            folder = f"{t['name']}_{sp['name']}"
            os.makedirs(folder, exist_ok=True)

            # perform a single global split according to this setup (deterministic)
            X_train, X_test, y_train_all, y_test_all = train_test_split(
                X_scaled, Y, test_size=sp["test_size"], random_state=random_state
            )

            # Build basis once per split (train/test)
            Phi_train = build_basis(X_train, t["basis"], t["degree"])
            Phi_test = build_basis(X_test, t["basis"], t["degree"])

            coef_folder = folder  # coefficients go into this folder (same as before)

            # --- Initialize accuracy tracking ---
            accuracy_lines = []
            count_above_2_total = 0
            count_above_5_total = 0

            # Loop over each response case
            for case_idx in range(n_cases):
                # prepare target vectors for this case
                y_train = y_train_all[:, case_idx]
                y_test = y_test_all[:, case_idx]

                model = Ridge(alpha=1e-3)  # explicitly use Ridge as in trial definition
                # fit on features (Phi_train) and single-target y_train
                model.fit(Phi_train, y_train)
                y_pred = model.predict(Phi_test)

                # --------------------------
                # NEW: compute de-normalized coefficients (coeffs for RAW x)
                # --------------------------
                # 1) predicted values on the training set (includes intercept effect via predictions)
                y_hat_train = model.predict(Phi_train)   # shape (n_train,)

                # 2) reconstruct raw X_train from scaled X_train using same safe-std rule
                X_train_raw = X_train * X_safe_std + X_mean

                # 3) build raw-basis matrix (same basis_type and degree used for Phi)
                Phi_raw_train = build_basis(X_train_raw, t["basis"], t["degree"])

                # 4) solve least-squares to get coefficients in raw-variable basis
                #     Phi_raw_train @ coef_raw ≈ y_hat_train
                coef_raw, *_ = np.linalg.lstsq(Phi_raw_train, y_hat_train, rcond=None)
                coef_raw = coef_raw.reshape(-1)  # ensure 1D

                # --- Save de-normalized coefficients (these are for RAW x) ---
                coef_file = os.path.join(coef_folder, f"responsecoef_case-{case_idx}.csv")
                np.savetxt(coef_file, coef_raw.reshape(1, -1), delimiter=",", header="Coefficients")

                # --- Save normalization exactly as before (Mean,Std stacked) ---
                norm_file = os.path.join(coef_folder, f"normalization_case-{case_idx}.csv")
                norm_data = np.vstack([X_mean, X_std])
                np.savetxt(norm_file, norm_data, delimiter=",", header="Mean,Std")

                # --- Error metrics (same computation as before) ---
                errors = np.abs((y_pred - y_test) / y_test) * 100
                if errors.ndim == 1:
                    errors = errors.reshape(-1, 1)

                max_err = np.max(errors, axis=0)
                min_err = np.min(errors, axis=0)
                mean_err = np.mean(errors, axis=0)

                # Update total counts
                if max_err[0] > 2:
                    count_above_2_total += 1
                if max_err[0] > 5:
                    count_above_5_total += 1

                # Store formatted line for this case
                accuracy_lines.append(
                    f"Case {case_idx+1}: min={min_err[0]:.3f}%, max={max_err[0]:.3f}%, mean={mean_err[0]:.3f}%\n"
                )

                # --- Save parity plot (same filename scheme and style) ---
                plot_parity(
                    y_true=y_test,
                    y_pred=y_pred,
                    case_index=case_idx,
                    max_err=max_err[0],
                    mean_err=mean_err[0],
                    outdir=coef_folder,
                )

            # --- Write a single aggregated accuracy file per folder ---
            acc_file = os.path.join(coef_folder, "prs_accuracy_file.txt")
            with open(acc_file, "w") as f:
                for line in accuracy_lines:
                    f.write(line)
                f.write("\n")
                f.write(f"Count max_err >2%: {count_above_2_total}\n")
                f.write(f"Count max_err >5%: {count_above_5_total}\n")

            print(f"Finished {t['name']} with {sp['name']} — results in folder: {folder}")

    print("All selected trials and splits completed.")





'''
x = "/home/user/Desktop/thermo_param_opt_new/prs-and_opti_with_actual_param/dm_origional_params.csv"
y = "/home/user/Desktop/thermo_param_opt_new/prs-and_opti_with_actual_param/Opt/Data/Simulations" 
run_prs_trials(x , y)
'''
