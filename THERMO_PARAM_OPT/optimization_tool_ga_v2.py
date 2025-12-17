import numpy as np
from solution import Solution
from scipy.optimize import minimize 
from scipy import optimize as spopt
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pygad
import matplotlib as mpl
mpl.use('Agg')
style.use("fivethirtyeight")
import os
from scipy.optimize import rosen, differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds
from scipy.optimize import shgo
from scipy.optimize import BFGS
import pickle

class OptimizationTool(object):
	def __init__(self,
		target_list=None,frequency = None,opt_method = "GA"):
		
		self.target_list = target_list
		self.objective = 0
		self.frequency = frequency
		self.count = 0
		self.opt_method = opt_method
	
	def obj_func_of_selected_PRS_thermo(self,Z):	
		
		self.count +=1
		string_x = "{self.count},"
		for i in Z:
			string_x+=f"{i},"
		string_x+="\n"
		note = open("guess_values.txt","+a").write(string_x)

		###Algorithm:
		#1] Get the guess values into saperate bins corresponding to species
			# eg: Z_{Ar: Low} = [x1,x2,x3,x4,x5] -> store x5 in self.unsrt[Ar:High].first
			# pass Z_{Ar: Low} to DECODER, get X_{Ar:Low}
		V_of_Z = {}
		count = 0
		for ind,speciesID in enumerate(self.unsrt):
			if self.unsrt[speciesID].a1_star is None or np.all(self.unsrt[speciesID].a1_star) == None:
				species_name = self.unsrt[speciesID].species
				lim = self.unsrt[speciesID].temp_limit
				p_o = self.species_dict[species_name][lim]
				zeta_max = self.zeta_dict[species_name][lim]
				#print(zeta_max)
				
				cov = self.cov_dict[species_name][lim]
				Z_species = Z[count:count+len(p_o)]
				T_low = self.T[count:count+len(p_o)]
				
				Cp_T = []
				for index in range(5):
					t = T_low[index]
					Theta = np.asarray([t/t,t,t**2,t**3,t**4])
					p_ = p_o + Z_species[index]*np.asarray(np.dot(cov,zeta_max)).flatten()
					Cp_T.append(Theta.dot(p_))
				x_species = self.unsrt[speciesID].DECODER(np.asarray(Cp_T),np.asarray(T_low),species_name,self.count,tag="Low")
				count+=len(p_o)
				V_of_Z[speciesID] = x_species
				for spec in self.unsrt:
					if self.unsrt[spec].species == species_name:
						self.unsrt[spec].a1_star = Z_species
						self.unsrt[spec].a2_star = Z_species[-1]
						self.unsrt[spec].Cp_T_mid_star = Cp_T[-1]
			else:
				species_name = self.unsrt[speciesID].species
				lim = self.unsrt[speciesID].temp_limit
				p_o = self.species_dict[species_name][lim]
				zeta_max = self.zeta_dict[species_name][lim]
				cov = self.cov_dict[species_name][lim]
				Z_species = Z[count:count+len(p_o)-1]
				Cp_T = [self.unsrt[speciesID].Cp_T_mid_star]  ##Start with mid Cp value 1000k
				T_high = self.T[count:count+len(p_o)-1]
				for index in range(4):  ### Start from the next point after 1000 K
					t = T_high[index]
					Theta = np.asarray([t/t,t,t**2,t**3,t**4])
					p_ = p_o + Z_species[index]*np.asarray(np.dot(cov,zeta_max)).flatten()
					Cp_T.append(Theta.dot(p_))
				x_species = self.unsrt[speciesID].DECODER(np.asarray(Cp_T),np.asarray(T_high),species_name,self.count,tag="High")
				count+=len(p_o)-1
				V_of_Z[speciesID] = x_species
				for spec in self.unsrt:
					if self.unsrt[spec].species == species_name:
						self.unsrt[spec].a1_star = None
						self.unsrt[spec].a2_star = None
						self.unsrt[spec].Cp_T_mid_star = None
		
		string = ""
		V = []
		for spec in self.unsrt:
			V.extend(list(V_of_Z[spec]))
			for k in V_of_Z[spec]:
				string+=f"{k},"
			#x_transformed.extend(temp)
		string+=f"\n"
		V = np.asarray(V)
		#print(V)
		zeta_file = open("zeta_guess_values.txt","+a").write(string)	
		#2] Use X_{Ar:Low} to generate obj. function value from response surface
			#eta_{case_0}= eta([X_{Ar:Low},X_{Ar:High},X_{H2:Low},...])
			#Make sure V_{case_0} = [X_{Ar:Low},X_{Ar:High},X_{H2:Low},...] is in the order of self.unsrt
		"""
		Just for simplicity
		"""
		x = V
		
		obj = 0.0
		rejected_PRS = []
		rejected_PRS_index = []
		target_value = []
		target_stvd = []
		direct_target_value = []
		direct_target_stvd = []
		target_value_2 = []
		case_stvd = []
		case_systematic_error = []
		response_value = []
		response_stvd = []	
		target_weights = []	
		COUNT_Tig = 0
		COUNT_Fls = 0
		COUNT_All = 0	
		COUNT_Flw = 0
		frequency = {}
		diff = []
		diff_2 = []
		diff_3 = {}
		for i,case in enumerate(self.target_list):
			if self.ResponseSurfaces[i].selection == 1:	
				if case.target == "Tig":
					if case.d_set in frequency:
						frequency[case.d_set] += 1
					else:
						frequency[case.d_set] = 1
					COUNT_All +=1
					COUNT_Tig +=1
					#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
					#dataset_weights = (1/COUNT_All)
					
					val = self.ResponseSurfaces[i].evaluate(x)
					#print(val)
					#val,grad = case.evaluateResponse(x)
					f_exp = np.log(case.observed*10)
					#print(f_exp,val)
					#w = 1/(np.log(case.std_dvtn*10))
					w_ = case.std_dvtn/case.observed
					w = 1/w_
					diff.append((val-f_exp)*w)
					#diff.append((val - f_exp)/f_exp)
					diff_2.append(val - f_exp)
					diff_3[case.uniqueID] = (val-f_exp)/f_exp
					response_value.append(val)
					#response_stvd.append(grad)
					#diff.append(val - np.log(case.observed*10))
					target_value.append(np.log(case.observed*10))
					target_value_2.append(np.log(case.observed))
					target_stvd.append(1/(np.log(case.std_dvtn*10)))
					case_stvd.append(np.log(case.std_dvtn*10))
					case_systematic_error.append(abs(np.log(case.observed*10)-val))
					#target_weights.append(dataset_weights)				
				elif case.target == "Fls":
					if case.d_set in frequency:
						frequency[case.d_set] += 1
					else:
						frequency[case.d_set] = 1
					COUNT_All +=1
					COUNT_Fls +=1
					#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
					#dataset_weights = (1/COUNT_All)
					
					val = np.exp(self.ResponseSurfaces[i].evaluate(x))
					response_value.append(val)
					f_exp = case.observed
					w = 1/(case.std_dvtn)
					diff.append((val - f_exp)*w)
					#response_stvd.append(grad)
					target_value.append(case.observed)
					target_value_2.append(case.observed)
					target_stvd.append(1/(case.std_dvtn))
					case_stvd.append(case.std_dvtn)
					case_systematic_error.append(abs(case.observed)-val)
					#target_weights.append(dataset_weights)	
				elif case.target == "Flw":
					if case.d_set in frequency:
						frequency[case.d_set] += 1
					else:
						frequency[case.d_set] = 1
					COUNT_All +=1
					COUNT_Flw +=1
					#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
					#dataset_weights = (1/COUNT_All)
					
					val = self.ResponseSurfaces[i].evaluate(x)
					response_value.append(val)
					f_exp = case.observed
					w = 1/(case.std_dvtn)
					diff.append((val - f_exp)*w)
					#response_stvd.append(grad)
					target_value.append(np.log(case.observed))
					target_value_2.append(np.log(case.observed))
					target_stvd.append(1/(np.log(case.std_dvtn)+abs(np.log(case.observed)-val)))
					case_stvd.append(np.log(case.std_dvtn))
					case_systematic_error.append(abs(np.log(case.observed)-val))
					#target_weights.append(dataset_weights)	
			
		
		diff = np.asarray(diff)
		multiplicating_factors = []
		#multiplicating_factors = np.asarray(target_weights)*np.asarray(target_stvd)
		for i,case in enumerate(self.target_list):
			if self.ResponseSurfaces[i].selection == 1:	
				if case.target == "Tig":
					multiplicating_factors.append(1/COUNT_Tig)
		
				elif case.target == "Fls":
					multiplicating_factors.append(1/COUNT_Fls)
		
		multiplicating_factors= np.asarray(multiplicating_factors)		
		#Giving all datapoints equal weights
		#multiplicating_factor = 1/COUNT_All
		
		for i,dif in enumerate(diff):
			#obj+= multiplicating_factors[i]*(dif)**2
			#obj+= multiplicating_factor*(dif)**2
			#obj+= multiplicating_factors[i]*(dif)**2
			obj+=dif**2		
		
		Diff_3 = open("Dataset_based_obj","+a").write(f"{diff_3}\n")
		get_opt = open("Objective.txt","+a").write(f"{obj}\n")
		string_v = f"{self.count},"
		for i in x:
			string_v+=f"{x},"
		string_v+="\n"
		note = open("guess_values_TRANSFORMED.txt","+a").write(string_v)
		#string_response="#CaseID,Obs,Xvector...\n"
		#for i in range(len(target_value)):
		#	string_response+=f"{target_value[i]},"
		string_response = ""
		for i in range(len(response_value)):
			string_response+=f"{response_value[i]},"
		string_response+="\n"	
		get_target_value = open("response_values.txt","+a").write(string_response)
		#print(obj)
		#if self.count > 3:
		#	raise AssertionError("Stop!")
		return obj
	
	def fitness_function_for_T_DIPENDENT(self):
		global obj_func_of_selected_PRS_thermo
		def obj_func_of_selected_PRS_thermo(Z,solution_idx):	
		
			self.count +=1
			string_x = "{self.count},"
			for i in Z:
				string_x+=f"{i},"
			string_x+="\n"
			note = open("guess_values.txt","+a").write(string_x)

			###Algorithm:
			#1] Get the guess values into saperate bins corresponding to species
				# eg: Z_{Ar: Low} = [x1,x2,x3,x4,x5] -> store x5 in self.unsrt[Ar:High].first
				# pass Z_{Ar: Low} to DECODER, get X_{Ar:Low}
			V_of_Z = {}
			count = 0
			for ind,speciesID in enumerate(self.unsrt):
				if self.unsrt[speciesID].a1_star is None or np.all(self.unsrt[speciesID].a1_star) == None:
					species_name = self.unsrt[speciesID].species
					lim = self.unsrt[speciesID].temp_limit
					p_o = self.species_dict[species_name][lim]
					zeta_max = self.zeta_dict[species_name][lim]
					#print(zeta_max)
					
					cov = self.cov_dict[species_name][lim]
					Z_species = Z[count:count+len(p_o)]
					T_low = self.T[count:count+len(p_o)]
					
					Cp_T = []
					for index in range(5):
						t = T_low[index]
						Theta = np.asarray([t/t,t,t**2,t**3,t**4])
						p_ = p_o + Z_species[index]*np.asarray(np.dot(cov,zeta_max)).flatten()
						Cp_T.append(Theta.dot(p_))
					x_species = self.unsrt[speciesID].DECODER(np.asarray(Cp_T),np.asarray(T_low),species_name,self.count,tag="Low")
					count+=len(p_o)
					V_of_Z[speciesID] = x_species
					for spec in self.unsrt:
						if self.unsrt[spec].species == species_name:
							self.unsrt[spec].a1_star = Z_species
							self.unsrt[spec].a2_star = Z_species[-1]
							self.unsrt[spec].Cp_T_mid_star = Cp_T[-1]
				else:
					species_name = self.unsrt[speciesID].species
					lim = self.unsrt[speciesID].temp_limit
					p_o = self.species_dict[species_name][lim]
					zeta_max = self.zeta_dict[species_name][lim]
					cov = self.cov_dict[species_name][lim]
					Z_species = Z[count:count+len(p_o)-1]
					Cp_T = [self.unsrt[speciesID].Cp_T_mid_star]  ##Start with mid Cp value 1000k
					T_high = self.T[count:count+len(p_o)-1]
					for index in range(4):  ### Start from the next point after 1000 K
						t = T_high[index]
						Theta = np.asarray([t/t,t,t**2,t**3,t**4])
						p_ = p_o + Z_species[index]*np.asarray(np.dot(cov,zeta_max)).flatten()
						Cp_T.append(Theta.dot(p_))
					x_species = self.unsrt[speciesID].DECODER(np.asarray(Cp_T),np.asarray(T_high),species_name,self.count,tag="High")
					count+=len(p_o)-1
					V_of_Z[speciesID] = x_species
					for spec in self.unsrt:
						if self.unsrt[spec].species == species_name:
							self.unsrt[spec].a1_star = None
							self.unsrt[spec].a2_star = None
							self.unsrt[spec].Cp_T_mid_star = None
			
			string = ""
			V = []
			for spec in self.unsrt:
				V.extend(list(V_of_Z[spec]))
				for k in V_of_Z[spec]:
					string+=f"{k},"
				#x_transformed.extend(temp)
			string+=f"\n"
			V = np.asarray(V)
			#print(V)
			zeta_file = open("zeta_guess_values.txt","+a").write(string)	
			#2] Use X_{Ar:Low} to generate obj. function value from response surface
				#eta_{case_0}= eta([X_{Ar:Low},X_{Ar:High},X_{H2:Low},...])
				#Make sure V_{case_0} = [X_{Ar:Low},X_{Ar:High},X_{H2:Low},...] is in the order of self.unsrt
			"""
			Just for simplicity
			"""
			x = V
			
			obj = 0.0
			rejected_PRS = []
			rejected_PRS_index = []
			target_value = []
			target_stvd = []
			direct_target_value = []
			direct_target_stvd = []
			target_value_2 = []
			case_stvd = []
			case_systematic_error = []
			response_value = []
			response_stvd = []	
			target_weights = []	
			COUNT_Tig = 0
			COUNT_Fls = 0
			COUNT_All = 0	
			COUNT_Flw = 0
			frequency = {}
			diff = []
			diff_2 = []
			diff_3 = {}
			for i,case in enumerate(self.target_list):
				if self.ResponseSurfaces[i].selection == 1:	
					if case.target == "Tig":
						if case.d_set in frequency:
							frequency[case.d_set] += 1
						else:
							frequency[case.d_set] = 1
						COUNT_All +=1
						COUNT_Tig +=1
						#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
						#dataset_weights = (1/COUNT_All)
						
						val = self.ResponseSurfaces[i].evaluate(x)
						#print(val)
						#val,grad = case.evaluateResponse(x)
						f_exp = np.log(case.observed*10)
						#print(f_exp,val)
						#w = 1/(np.log(case.std_dvtn*10))
						w_ = case.std_dvtn/case.observed
						w = 1/w_
						diff.append((val-f_exp)*w)
						#diff.append((val - f_exp)/f_exp)
						diff_2.append(val - f_exp)
						diff_3[case.uniqueID] = (val-f_exp)/f_exp
						response_value.append(val)
						#response_stvd.append(grad)
						#diff.append(val - np.log(case.observed*10))
						target_value.append(np.log(case.observed*10))
						target_value_2.append(np.log(case.observed))
						target_stvd.append(1/(np.log(case.std_dvtn*10)))
						case_stvd.append(np.log(case.std_dvtn*10))
						case_systematic_error.append(abs(np.log(case.observed*10)-val))
						#target_weights.append(dataset_weights)				
					elif case.target == "Fls":
						if case.d_set in frequency:
							frequency[case.d_set] += 1
						else:
							frequency[case.d_set] = 1
						COUNT_All +=1
						COUNT_Fls +=1
						#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
						#dataset_weights = (1/COUNT_All)
						
						val = np.exp(self.ResponseSurfaces[i].evaluate(x))
						response_value.append(val)
						f_exp = case.observed
						w = 1/(case.std_dvtn)
						diff.append((val - f_exp)*w)
						#response_stvd.append(grad)
						target_value.append(case.observed)
						target_value_2.append(case.observed)
						target_stvd.append(1/(case.std_dvtn))
						case_stvd.append(case.std_dvtn)
						case_systematic_error.append(abs(case.observed)-val)
						#target_weights.append(dataset_weights)	
					elif case.target == "Flw":
						if case.d_set in frequency:
							frequency[case.d_set] += 1
						else:
							frequency[case.d_set] = 1
						COUNT_All +=1
						COUNT_Flw +=1
						#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
						#dataset_weights = (1/COUNT_All)
						
						val = self.ResponseSurfaces[i].evaluate(x)
						response_value.append(val)
						f_exp = case.observed
						w = 1/(case.std_dvtn)
						diff.append((val - f_exp)*w)
						#response_stvd.append(grad)
						target_value.append(np.log(case.observed))
						target_value_2.append(np.log(case.observed))
						target_stvd.append(1/(np.log(case.std_dvtn)+abs(np.log(case.observed)-val)))
						case_stvd.append(np.log(case.std_dvtn))
						case_systematic_error.append(abs(np.log(case.observed)-val))
						#target_weights.append(dataset_weights)	
				
			
			diff = np.asarray(diff)
			multiplicating_factors = []
			#multiplicating_factors = np.asarray(target_weights)*np.asarray(target_stvd)
			for i,case in enumerate(self.target_list):
				if self.ResponseSurfaces[i].selection == 1:	
					if case.target == "Tig":
						multiplicating_factors.append(1/COUNT_Tig)
			
					elif case.target == "Fls":
						multiplicating_factors.append(1/COUNT_Fls)
			
			multiplicating_factors= np.asarray(multiplicating_factors)		
			#Giving all datapoints equal weights
			#multiplicating_factor = 1/COUNT_All
			
			for i,dif in enumerate(diff):
				#obj+= multiplicating_factors[i]*(dif)**2
				#obj+= multiplicating_factor*(dif)**2
				#obj+= multiplicating_factors[i]*(dif)**2
				obj+=dif**2		
			
			Diff_3 = open("Dataset_based_obj","+a").write(f"{diff_3}\n")
			get_opt = open("Objective.txt","+a").write(f"{obj}\n")
			string_v = f"{self.count},"
			for i in x:
				string_v+=f"{x},"
			string_v+="\n"
			note = open("guess_values_TRANSFORMED.txt","+a").write(string_v)
			#string_response="#CaseID,Obs,Xvector...\n"
			#for i in range(len(target_value)):
			#	string_response+=f"{target_value[i]},"
			string_response = ""
			for i in range(len(response_value)):
				string_response+=f"{response_value[i]},"
			string_response+="\n"	
			get_target_value = open("response_values.txt","+a").write(string_response)
			#print(obj)
			#if self.count > 3:
			#	raise AssertionError("Stop!")
			fitness = 1.0 / (np.abs((obj) - 0) + 0.000001)
			#fitness = 1.0 / (np.abs((obj) - 0))
			record =open("samplefile.txt","+a").write(f"{self.ga_instance.generations_completed},{self.count},{self.objective},{fitness}\n")
			
			return fitness
		return obj_func_of_selected_PRS_thermo
	

	def plot_DATA(self,x):	
		kappa_curve = {}
		count = 0
		for i in self.rxn_index:
			temp = []
			for j in range(len(self.T)):
				temp.append(x[count])
				count += 1
			#trial = [temp[0],(temp[0]+temp[1])/2,temp[1]]
			Kappa = self.kappa_0[i] + temp*(self.kappa_max[i]-self.kappa_0[i])
			kappa_curve[i] = np.asarray(Kappa).flatten()
			
		
		zeta = {}
		for rxn in self.rxn_index:
			zeta[rxn] = self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])
	
		x_transformed = []
		string = ""
		for rxn in self.rxn_index:
			temp = list(zeta[rxn])
			for k in temp:
				string+=f"{k},"
			x_transformed.extend(temp)
		string+=f"\n"
		x_transformed = np.asarray(x_transformed)
		zeta_file = open("zeta_guess_values.txt","+a").write(string)
		"""
		Just for simplicity
		"""
		x = x_transformed
		
		obj = 0.0
		rejected_PRS = []
		rejected_PRS_index = []
		target_value = []
		target_stvd = []
		direct_target_value = []
		direct_target_stvd = []
		target_value_2 = []
		case_stvd = []
		case_systematic_error = []
		response_value = []
		response_stvd = []	
		target_weights = []	
		COUNT_Tig = 0
		COUNT_Fls = 0
		COUNT_All = 0	
		COUNT_Flw = 0
		frequency = {}
		diff = []
		diff_2 = []
		diff_3 = {}
		VALUE = []
		EXP = []
		CASE = []
		TEMPERATURE = []
		for i,case in enumerate(self.target_list):
			if self.ResponseSurfaces[i].selection == 1:	
				if case.target == "Tig":					
					t = case.temperature
					val = self.ResponseSurfaces[i].evaluate(x)
					f_exp = np.log(case.observed*10)
					VALUE.append(np.exp(val)/10)
					EXP.append(np.exp(f_exp)/10)
					CASE.append(case.dataSet_id)
					TEMPERATURE.append(t)				
				elif case.target == "Fls":
					t = case.temperature
					val = np.exp(self.ResponseSurfaces[i].evaluate(x))
					f_exp = case.observed
					VALUE.append(val)
					EXP.append(f_exp)	
					CASE.append(case.dataSet_id)
					TEMPERATURE.append(t)
		
		return VALUE,EXP,CASE,TEMPERATURE
	

	def _obj_func(self,x):
		string = ""
		for i in x:
			string+=f"{i},"
		string+=f"\n"
		#x_transformed = np.asarray(x_transformed)
		zeta_file = open("zeta_guess_values.txt","+a").write(string)
		"""
		Just for simplicity
		"""
		
		obj = 0.0
		rejected_PRS = []
		rejected_PRS_index = []
		target_value = []
		target_stvd = []
		direct_target_value = []
		direct_target_stvd = []
		target_value_2 = []
		case_stvd = []
		case_systematic_error = []
		response_value = []
		response_stvd = []	
		target_weights = []	
		COUNT_Tig = 0
		COUNT_Fls = 0
		COUNT_All = 0	
		COUNT_Flw = 0
		frequency = {}
		
		
		for i,case in enumerate(self.target_list):
			if self.ResponseSurfaces[i].selection == 1:	
				if case.target == "Tig":
					if case.d_set in frequency:
						frequency[case.d_set] += 1
					else:
						frequency[case.d_set] = 1
					COUNT_All +=1
					COUNT_Tig +=1
					#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
					#dataset_weights = (1/COUNT_All)
					
					val = self.ResponseSurfaces[i].evaluate(x)
					#print(val)
					#val,grad = case.evaluateResponse(x)
					response_value.append(val)
					#response_stvd.append(grad)
					target_value.append(np.log(case.observed*10))
					
					target_value_2.append(np.log(case.observed))
					target_stvd.append(1/(np.log(case.std_dvtn*10)))
					case_stvd.append(case.std_dvtn/case.observed)
					case_systematic_error.append(abs(np.log(case.observed*10)-val))
					#target_weights.append(dataset_weights)				
				elif case.target == "Fls":
					if case.d_set in frequency:
						frequency[case.d_set] += 1
					else:
						frequency[case.d_set] = 1
					COUNT_All +=1
					COUNT_Fls +=1
					#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
					#dataset_weights = (1/COUNT_All)
					
					val = np.exp(self.ResponseSurfaces[i].evaluate(x))
					response_value.append(val)
					#response_stvd.append(grad)
					target_value.append(case.observed)
					target_value_2.append(case.observed)
					target_stvd.append(1/(case.std_dvtn))
					case_stvd.append(case.std_dvtn)
					case_systematic_error.append(abs(case.observed)-val)
					#target_weights.append(dataset_weights)	
				elif case.target == "Flw":
					if case.d_set in frequency:
						frequency[case.d_set] += 1
					else:
						frequency[case.d_set] = 1
					COUNT_All +=1
					COUNT_Flw +=1
					#dataset_weights = (1/len(self.frequency))*float(self.frequency[case.d_set])
					#dataset_weights = (1/COUNT_All)
					
					val = self.ResponseSurfaces[i].evaluate(x)
					response_value.append(val)
					#response_stvd.append(grad)
					target_value.append(np.log(case.observed))
					target_value_2.append(np.log(case.observed))
					target_stvd.append(1/(np.log(case.std_dvtn)+abs(np.log(case.observed)-val)))
					case_stvd.append(np.log(case.std_dvtn))
					case_systematic_error.append(abs(np.log(case.observed)-val))
					#target_weights.append(dataset_weights)	
			
		self.count +=1
		diff = np.asarray(response_value)-np.asarray(target_value)
		multiplicating_factors = []
		#multiplicating_factors = np.asarray(target_weights)*np.asarray(target_stvd)
		for i,case in enumerate(self.target_list):
			if self.ResponseSurfaces[i].selection == 1:	
				if case.target == "Tig":
					multiplicating_factors.append(1/COUNT_Tig)
		
				elif case.target == "Fls":
					multiplicating_factors.append(0.05*(1/COUNT_Fls))
		
		multiplicating_factors= np.asarray(multiplicating_factors)		
		#Giving all datapoints equal weights
		#multiplicating_factor = 1/COUNT_All
		
		for i,dif in enumerate(diff):
			#obj+= multiplicating_factors[i]*(dif)**2
			#obj+= multiplicating_factor*(dif)**2
			obj+= multiplicating_factors[i]*(dif)**2
						
		
		note = open("guess_values.txt","+a").write(f"{self.count},{x}\n")
		get_target_value = open("response_values.txt","+a").write(f"\t{target_value},{response_value}\n")
		get_opt = open("Objective.txt","+a").write(f"{obj}\n")
		return obj
	
		

		
	def run_optimization_with_selected_PRS_thermo(self,unsrt_data,ResponseSurfaces,Input_data):
		
		self.unsrt = unsrt_data
		self.ResponseSurfaces = ResponseSurfaces
		self.Input_data = Input_data
		algorithm = Input_data["Type"]["Algorithm"]
		self.species_index = np.asarray([species for species in unsrt_data]).flatten()
		self.species_index = np.asarray([species for species in unsrt_data]).flatten()
		self.init_guess = np.ones(len(self.species_index))	
		bounds = tuple([(-1,1) for _ in self.init_guess ])
		
		if Input_data["Stats"]["Design_of_PRS"] == "A-facto":
		
			opt_output = minimize(self._obj_func,
					    self.init_guess,
					    bounds=bounds,
					    method='L-BFGS-B',  # Replace 'algorithm' with specific method if known
					    options={"maxiter": 500000}  # Replace 'maxfev' with 'maxiter'
					)
			#opt_output = minimize(self._obj_func,self.init_guess,bounds=bounds,method=algorithm,options={"maxiter":500000})
			print(opt_output)
			optimal_parameters = np.asarray(opt_output.x)
			optimal_parameters_zeta = np.asarray(opt_output.x)
			cov = []
		
		else:
			self.species_index = []
			self.cp_0 = {}
			self.cp_max = {}
			self.T = []
			for species in self.unsrt:
				self.species_index.append(species)
				lim = self.unsrt[species].temp_limit
				T = self.unsrt[species].temperatures
				#print("line 1179 Optimization tool T \n " , T)
				if lim == "Low":
					self.T.extend(list(np.linspace(T[0],T[-1],5)))
				else:
					self.T.extend(list(np.linspace(T[0],T[-1],5)[1:])) #skip the common temperature
			self.T = np.asarray(self.T).flatten() #DOF = 9
			#print(self.T)
			all_species_list = []  # List 1: All species with descriptors- low and high
			key_species_list = []  # List 2: Unique species names
			paired_data_list = []  # List 3: Paired High and Low data for each species


			self.species_dict = {}
			self.zeta_dict = {}
			self.cov_dict = {}
			for species_descriptor in self.unsrt:
				all_species_list.append(species_descriptor)
				base_species = species_descriptor.split(":")[0]
				if base_species not in key_species_list:
					key_species_list.append(base_species)
				if base_species not in self.species_dict:
					self.species_dict[base_species] = {}
					self.zeta_dict[base_species] = {}
					self.cov_dict[base_species] = {}
				self.species_dict[base_species][species_descriptor.split(":")[1]] = self.unsrt[species_descriptor].nominal[0:5]
				self.zeta_dict[base_species][species_descriptor.split(":")[1]] = self.unsrt[species_descriptor].zeta_max.x
				self.cov_dict[base_species][species_descriptor.split(":")[1]] = self.unsrt[species_descriptor].cov
			
			#print(len(self.T))
			#self.init_guess = np.zeros(len(self.T[1:])*len(self.species_index))
			self.init_guess = np.array([-2.96893286,0.76861981,17.4261154,45.4912758,10.0504961,-1.50947742,1.07985437,1.65623474,-0.339644179,5.14789303,1.41980216,-2.7868245,18.5771791,6.38730047,2.03608777,0.735639342,0.635280217,4.44022129,2.46345247,1.08053651,-1.0320216,2.28972601,11.2433774,8.84643232,2.6845374,-3.32188469,-5.5823844,-4.73678754,-24.621364,4.49043001,0.109785722,-3.61280577,72.268569,37.1318507,8.58929532,4.56276155,16.72694,-20.3222704,-22.4714367,11.6492897,2.83900928,2.98787738,-35.8113613,-14.0451364,-2.11520168,11.8108703,-14.8673447,-61.017312,117.928411,27.3207054,-0.512431326,-0.395861123,14.5678042,8.11913429,2.5010311,-1.00499721,-0.540682717,-0.323527367,-0.0752015412,1.21455104,0.192446569,0.694958936,34.9679687,4.32339462,-78.2844219,-0.266273536,-0.171549093,-6.13088537,-6.90337571,5.25188327,18.8685928,346.641921,-1778.20726,3886.08529,1763.37944,-9.38066425,-32.1146073,-181.403055,-11.5461109,56.3705034,3.11977583,1.62205554,-21.3389812,-5.59264694,-0.230593311,6.90737712,-40.2357966,-12.208475,-151.828192,-23.9850608,1.09325274,-0.405419652,-8.22169336,5.77094053,1.35098526,1.10570741,4.32684794,2.39707177,1.62472607,11.7328336])
			#self.init_guess = np.ones(len(self.T)*len(self.species_index))
			
		cons = []
		# Cp(T)-based monotonicity and equality constraints
		Tgrid_low = [500,600,700,800,900,1000]
		Tgrid_high = [1000,1100,1200,1300,1400,1500]

		def Cp_eval(coeffs, T):
			return coeffs[0] + coeffs[1]*T + coeffs[2]*T**2 + coeffs[3]*T**3 + coeffs[4]*T**4

		for base_species, blocks in self.species_dict.items():
			for lim, p0 in blocks.items():
				cov = self.cov_dict[base_species][lim]
				grid = Tgrid_low if lim.lower() == 'low' else Tgrid_high
				for i in range(len(grid)-1):
					T1, T2 = grid[i], grid[i+1]
					cons.append({
						'type': 'ineq',
						'fun': lambda z, p0=p0, cov=cov, T1=T1, T2=T2: float(Cp_eval(p0 + cov.dot(z), T2) - Cp_eval(p0 + cov.dot(z), T1))
					})

		for base_species, blocks in self.species_dict.items():
			if 'Low' in blocks and 'High' in blocks:
				p0_low = blocks['Low']; cov_low = self.cov_dict[base_species]['Low']
				p0_high = blocks['High']; cov_high = self.cov_dict[base_species]['High']
				cons.append({
					'type': 'eq',
					'fun': lambda z, p0_low=p0_low, cov_low=cov_low, p0_high=p0_high, cov_high=cov_high: float(Cp_eval(p0_low + cov_low.dot(z), 1000) - Cp_eval(p0_high + cov_high.dot(z), 1000))
				})
bounds = tuple([(-1,1) for _ in self.init_guess ])
			
			self.theta_global = np.array([self.T/self.T,self.T,self.T**2,self.T**3,self.T**4])
			#self.theta_inv = np.linalg.inv(theta.T)
			start = time.time()
			
			if "GD" in self.opt_method:
				opt_output = minimize(self.obj_func_of_selected_PRS_thermo,self.init_guess,bounds=bounds,constraints=cons,method='SLSQP',options={"maxiter":500000})
				stop = time.time()
				final=print(f"Time taken for optimization {stop-start}")
				print(opt_output)
				optimal_parameters = np.asarray(opt_output.x)
			
else:
    # GA branch with feasibility checks and optional user-provided gen0 population
    base_fitness = self.fitness_function_for_T_DIPENDENT()
    # wrapper fitness function that rejects infeasible candidates (returns very low fitness)
    def fitness_wrapper(solution, solution_idx):
        # solution is a 1D numpy array (zeta)
        z = np.array(solution)
        # check feasibility: Cp monotonicity on specified grids and low/high equality at 1000K
        feasible = True
        tol = 1e-9
        # Temperature grids
        T_low_grid = [500,600,700,800,900,1000]
        T_high_grid = [1000,1100,1200,1300,1400,1500]
        # helper to compute Cp values for a species block using existing decoding pattern
        def Cp_vals_for_species(z, speciesID, lim):
            p_o = self.species_dict[speciesID][lim]
            cov = self.cov_dict[speciesID][lim]
            # reconstruct Z_species from z by same ordering used elsewhere: assume contiguous blocks of length 5 per species' Low block
            # build start index by summing lengths of prior species' Low blocks (safe if all Low blocks length 5)
            start = 0
            for sp in self.species_dict:
                if sp == speciesID:
                    break
                if 'Low' in self.species_dict[sp]:
                    start += len(self.species_dict[sp]['Low'])
            Z_species = z[start:start+len(p_o)]
            grid = T_low_grid if lim.lower()=='low' else T_high_grid
            Cpvals = []
            for idx,t in enumerate(grid):
                Theta = np.array([t/t, t, t**2, t**3, t**4])
                p_ = p_o + Z_species[idx]*np.asarray(np.dot(cov, self.zeta_dict[speciesID][lim])).flatten()
                Cpvals.append(Theta.dot(p_))
            return np.array(Cpvals)
        # check all species blocks
        for speciesID, blocks in self.species_dict.items():
            for lim in blocks:
                try:
                    Cpvals = Cp_vals_for_species(z, speciesID, lim)
                except Exception:
                    feasible = False
                    break
                diffs = Cpvals[1:] - Cpvals[:-1]
                if np.any(diffs < -tol):
                    feasible = False
                    break
            if not feasible:
                break
            # check low-high equality at 1000K if both exist
            if 'Low' in blocks and 'High' in blocks:
                Cp_low = Cp_vals_for_species(z, speciesID, 'Low')
                Cp_high = Cp_vals_for_species(z, speciesID, 'High')
                if abs(Cp_low[-1] - Cp_high[0]) > 1e-6:
                    feasible = False
                    break
        # If infeasible, return a very low fitness so it's rejected by selection
        if not feasible:
            return -1e12
        # otherwise compute the actual fitness (pygad expects a scalar to maximize)
        return base_fitness(solution, solution_idx)

    # Prepare gene_space and GA parameters (reuse original settings)
    gene_space = [{'low': -1, 'high': 1} for _ in self.init_guess ]
    # If user provided initial population rows in self.gen0 (numpy array shape (pop_size, num_genes)), use them
    init_pop = None
    if hasattr(self, 'gen0') and self.gen0 is not None:
        init_pop = np.asarray(self.gen0, dtype=float)
        # ensure shape matches
        if init_pop.ndim == 1:
            init_pop = init_pop.reshape(1, -1)

    self.ga_instance = pygad.GA(num_generations=2000,
                   num_parents_mating=300,
                   fitness_func=fitness_wrapper,
                   init_range_low=-1,
                   init_range_high=1,
                   sol_per_pop=400,
                   num_genes=len(self.init_guess),
                   crossover_type="uniform",
                   crossover_probability=0.6,
                   mutation_type="adaptive",
                   mutation_probability=(0.03, 0.008),
                   gene_type=float,
                   allow_duplicate_genes=False,
                   gene_space=gene_space,
                   keep_parents = -1,
                   save_best_solutions=True,
                   save_solutions=True,
                   stop_criteria=["reach_200"],
                   initial_population=init_pop if init_pop is not None else None
                   )

    # Run GA
    self.ga_instance.run()
    self.ga_instance.plot_fitness()
    # retrieve best solution
    optimal_parameters = self.ga_instance.best_solution()[0]
Z_star = optimal_parameters
			V_of_Z = {}
			count = 0
			for ind,speciesID in enumerate(self.unsrt):
				if self.unsrt[speciesID].a1_star is None or np.all(self.unsrt[speciesID].a1_star) == None:
					species_name = self.unsrt[speciesID].species
					lim = self.unsrt[speciesID].temp_limit
					p_o = self.species_dict[species_name][lim]
					#print(p_o)
					zeta_max = self.zeta_dict[species_name][lim]
					cov = self.cov_dict[species_name][lim]
					Z_species = Z_star[count:count+len(p_o)]
					T_low = self.T[count:count+len(p_o)]
					#print("T_low \t", T_low)
					Cp_T = []
					for index in range(5):
						t = T_low[index]
						Theta = np.array([t/t,t,t**2,t**3,t**4])
						p_ = p_o + Z_species[index]*np.asarray(np.dot(cov,zeta_max)).flatten()
						Cp_T.append(Theta.dot(p_))
					x_species = self.unsrt[speciesID].DECODER(np.asarray(Cp_T),np.asarray(T_low),species_name,self.count,tag="Low")
					count+=len(p_o)
					V_of_Z[speciesID] = x_species
					for spec in self.unsrt:
						if self.unsrt[spec].species == species_name:
							self.unsrt[spec].a1_star = Z_species
							self.unsrt[spec].a2_star = Z_species[-1]
							self.unsrt[spec].Cp_T_mid_star = Cp_T[-1]
				else:
					species_name = self.unsrt[speciesID].species
					lim = self.unsrt[speciesID].temp_limit
					p_o = self.species_dict[species_name][lim]
					zeta_max = self.zeta_dict[species_name][lim]
					cov = self.cov_dict[species_name][lim]
					Z_species = Z_star[count:count+len(p_o)-1]
					Cp_T = [self.unsrt[speciesID].Cp_T_mid_star]  ##Start with mid Cp value 1000k
					T_high = self.T[count:count+len(p_o)-1]
					for index in range(4):  ### Start from the next point after 1000 K
						t = T_high[index]
						Theta = np.array([t/t,t,t**2,t**3,t**4])
						p_ = p_o + Z_species[index]*np.asarray(np.dot(cov,zeta_max)).flatten()
						Cp_T.append(Theta.dot(p_))
					x_species = self.unsrt[speciesID].DECODER(np.asarray(Cp_T),np.asarray(T_high),species_name,self.count,tag="High")
					count+=len(p_o)-1
					V_of_Z[speciesID] = x_species
			
			string = ""
			#print(V_of_Z)
			V_star = []
			for spec in self.unsrt:
				#V_star.extend(list(V_of_Z[spec]))
				for k in V_of_Z[spec]:
					string+=f"{k},"
					V_star.append(k)
				#x_transformed.extend(temp)
				string+=f"\n"
			V_star = np.asarray(V_star)
			zeta_file = open("OPTIMIZED_ZETA.csv","w").write(string)	
			
			#########################
			### For plotting purposes
			########################
			##NEED TO ADD LINES FOR THAT
		
						
		return np.asarray(Z_star),np.asarray(V_star),cov
	
	
