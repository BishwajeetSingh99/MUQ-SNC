
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
original_loc = sys.argv[1]
optimized_loc = sys.argv[2]
plot_loc = sys.argv[3] 
start = os.getcwd()
os.chdir(original_loc)
ORIGINAL_Files = os.listdir()
os.chdir(start)
os.chdir(optimized_loc)
OPTIMIZED_Files = os.listdir()
os.chdir(start)
#Dataset_loc = os.getcwd()
#file_list = os.listdir()

for file_ in ORIGINAL_Files:
	#Reading simulation data
	df_sim = pd.read_csv(original_loc+"/"+file_)
	df_sim = pd.DataFrame(df_sim)
	dT = 1000/df_sim["T"].to_numpy()
	Obs = df_sim["Obs(us)"].to_numpy()
	Sim = df_sim["Nominal"].to_numpy()

	#Reading simulation data
	df_sim_OPT = pd.read_csv(optimized_loc+"/"+file_)
	df_sim_OPT = pd.DataFrame(df_sim_OPT)
	dT_OPT = 1000/df_sim_OPT["T"].to_numpy()
	Obs_OPT = df_sim_OPT["Obs(us)"].to_numpy()
	Sim_OPT = df_sim_OPT["Nominal"].to_numpy()
	
	
	n_model_prior = np.log(np.asarray(Sim)*10)
	n_model_opt = np.log(np.asarray(Sim_OPT)*10)
	n_exp = np.log(np.asarray(Obs)*10)
	s_exp = 0.1*np.asarray(Obs)/np.asarray(Obs)

	obj_prior = (n_model_prior-n_exp)/s_exp
	op = 0
	for dif in obj_prior:
		op+=dif**2
	
	obj_opt = (n_model_opt-n_exp)/s_exp
	oo = 0
	for dif in obj_opt:
		oo+=dif**2	
	
	fig = plt.figure()
	plt.plot(dT,Sim,"b--",label="H2/CO (FFCM1)")
	plt.plot(dT,Sim_OPT,"r-",label="H2/CO ( Optimization)")
	#plt.plot(dT,Sim_PARTIAL_OPT,"y--",label="MB (HTC optimized \n     with p-PRS)")
	plt.errorbar(dT,Obs,yerr =0.1*(np.asarray(Obs)),fmt='k.',ecolor="black",markerfacecolor='black',markeredgecolor='black',markersize=8,capsize=2,elinewidth=0.7,
    markeredgewidth=0.5,label = f"Mech. obj (FFCM1) = {op:.2f},\n Mech. obj Optimized (this work) = {oo:.2f}")
	plt.plot(dT,Obs,"ro",label=f"{file_}")
	#plt.plot(dT_sim_lele,Sim_lele,"r--",label="Lele simulation")
	plt.legend(loc='best')
	plt.yscale("log")
	plt.xlabel("1000/T")
	plt.ylabel("Ignition delay time (ms)")
	#plt.title(r"P = 4 bar, $\phi$ = 0.25")
	#plt.show()
	plt.savefig(f"{plot_loc}/{file_.strip('.csv')}.pdf",bbox_inches="tight")
	
'''




def main():
    """
    Main function to parse command-line arguments and run the plotting script.
    """
    if len(sys.argv) != 4:
        print("Usage: python script.py <original_data_dir> <optimized_data_dir> <plot_output_dir>")
        sys.exit(1)

    original_loc = sys.argv[1]
    optimized_loc = sys.argv[2]
    plot_loc = sys.argv[3] 
    
    start = os.getcwd()
    os.chdir(original_loc)
    ORIGINAL_Files = os.listdir()
    os.chdir(start)
    os.chdir(optimized_loc)
    OPTIMIZED_Files = os.listdir()
    os.chdir(start)

    # Ensure the plot directory exists
    os.makedirs(plot_loc, exist_ok=True)

    for file_ in ORIGINAL_Files:
        if file_ not in OPTIMIZED_Files:
            print(f"Warning: Corresponding optimized file for '{file_}' not found. Skipping.")
            continue
            
        # Reading simulation data
        df_sim = pd.read_csv(os.path.join(original_loc, file_))
        df_sim = pd.DataFrame(df_sim)
        dT = 1000 / df_sim["T"].to_numpy()
        Obs = df_sim["Obs(us)"].to_numpy()
        Sim = df_sim["Nominal"].to_numpy()

        # Reading optimized simulation data
        df_sim_OPT = pd.read_csv(os.path.join(optimized_loc, file_))
        df_sim_OPT = pd.DataFrame(df_sim_OPT)
        dT_OPT = 1000 / df_sim_OPT["T"].to_numpy()
        Obs_OPT = df_sim_OPT["Obs(us)"].to_numpy()
        Sim_OPT = df_sim_OPT["Nominal"].to_numpy()
        
        # Calculating objective function values
        n_model_prior = np.log(np.asarray(Sim) * 10)
        n_model_opt = np.log(np.asarray(Sim_OPT) * 10)
        n_exp = np.log(np.asarray(Obs) * 10)
        s_exp = 0.1 * np.asarray(Obs) / np.asarray(Obs)

        obj_prior = (n_model_prior - n_exp) / s_exp
        op = np.sum(obj_prior**2)

        obj_opt = (n_model_opt - n_exp) / s_exp
        oo = np.sum(obj_opt**2)
        
        # Plotting section
        fig = plt.figure()
        plt.plot(dT, Sim, "b--", label="H2/CO (FFCM1)")
        plt.plot(dT, Sim_OPT, "r-", label="H2/CO (Optimization)")
        
        plt.errorbar(
            dT, Obs,
            yerr=0.1 * np.asarray(Obs),
            fmt='k.',
            ecolor="black",
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=8,
            capsize=2,
            elinewidth=0.7,
            markeredgewidth=0.5,
            label=f"Mech. obj (FFCM1) = {op:.2f},\nMech. obj Optimized (this work) = {oo:.2f}"
        )
        
        plt.plot(dT, Obs, "ro", label=f"{file_}")
        plt.legend(loc='best')
        plt.yscale("log")
        plt.xlabel("1000/T")
        plt.ylabel("Ignition delay time (ms)")
        plt.savefig(os.path.join(plot_loc, f"{os.path.splitext(file_)[0]}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print("plots created succesfully..return 0")

if __name__ == "__main__":
    main()	
	
	
	
	
	
	
	
	
