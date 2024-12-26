import sys
import numpy as np
import math
import time
import pickle
import random

from tqdm import tqdm

import flory

sys.path.append('../..')
print(np.__version__)


import matplotlib.pyplot as plt
import mpltern
#import matplotlib as mpl
import matplotlib as mpl

from mpltern.datasets import get_shanon_entropies

import importlib

from scipy import cluster, spatial

from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution

#sys.path.append('/Users/calvinsmith/dufresne_lab/lab_work/llps_classification')
import flory_generate as gen
import flory_plots as plots


from flory_phase_space import PhaseSpace

importlib.reload(gen)
importlib.reload(plots)
importlib.reload(flory_phase_space)

class FloryClassifier:
    
    def __init__(self):
        self.phi_in_thresh = 0.2
        self.phi_H_thresh = 0.3
        self.phi_out = 0.1
        self.num_sol = 1
        self.phi_in_range = [0.02,0.2]
        self.phi_in_comps = 2
        self.phi_out_vector = [0.1,0.1]
        self.num_of_points = 6
        self.chi_strength = 4
        self.H_dim = 2
        self.num_comps = self.H_dim + len(self.phi_out_vector)  + self.phi_in_comps + self.num_sol
        #self.num_comps_sol = self.num_comps + 1
        self.phi_in_min = 0.1
        self.chi_matrix = gen.chi_matrix_w_sol(self.num_comps,self.chi_strength)
        #self.chi_matrix_multi = gen.chi_matrix_w_sol(self.num_comps + 1,self.chi_strength)
        self.chi_matrix_custom = [[],[]]
        #print(self.chi_matrix)

    def update_vars(self):
        self.num_comps = self.H_dim + len(self.phi_out_vector)  + self.phi_in_comps + self.num_sol


    def reset_rand_chi_matrix(self, num_comps,chi_strength):
        ''' Change the chi_matrix'''
        self.chi_matrix = gen.chi_matrix_w_sol(num_comps, chi_strength)
        self.chi_strength = chi_strength
        print(f"Reset CHI_MATRIX: \n {self.chi_matrix}")

    def set_chi_matrix_multi(self, chi_matrix):
        self.chi_matrix_multi = chi_matrix
        self.chi_matrix = chi_matrix
    
    def get_info(self):
        ''' Returns information about the FloryClassifier item'''
        print(f"num_comps: {self.num_comps} ")
        print(f"H_DIM: {self.H_dim}")
        print(f"Phi_out_vector: {self.phi_out_vector}")
        print(f"Phi_in_comps: {self.phi_in_comps}")
        print(f"Phi_in_range: {self.phi_in_range}")
        print(f"num_of_points: {self.num_of_points}")
        print(f"CHI_STRENGTH: {self.chi_strength} \n CHI_MATRIX: \n {self.chi_matrix}")
        sol_remains = 1 - (sum(self.phi_out_vector) + self.phi_in_comps*max(self.phi_in_range)) 
        print(f"REMAINING SOLVENT: {sol_remains}")
        print(f"H component {sol_remains/self.H_dim}")

        

    
    def step_func(self,x):
        '''Goal Function '''
        if x >= 0.5:
            return 0
        elif 0 <= x < 0.5:
            return 1
        else: 
            raise Exception("Sorry, no numbers below zero")
            
   
    def calc_loss(self,points):
        '''Calculates the loss which is the difference between the readout function and the inputed points
        Takes in a set of points to compare to a defined readout_func in the class'''
        loss = 0
        for point in points:
            loss += (self.step_func(point[0]) - point[1])**2
        return loss
    
    ''' Readout Function
    def output_readout_func(self, phi_in, phi_out):
        x = math.exp(-(phi_in/phi_out))
        return x
    ''' 
    def phi_out_readout_func(self,phi_out):
        """A readout function that compares the value of phi_out[0] to phi_out[1] and returns the difference
        ed and then inputed into x = 0.5*(math.tanh(0.1*x) + 1).
        """
        print('TEST')
        x = phi_out[0] - phi_out[1]
        x = x/sum(phi_out) 
        x = 0.5*(math.tanh(0.1*x) + 1)
        return x

    
    def pick_point(self, num_comps, phi_H = [], auto_chi = False):
        """ Input: num_comps, phi_H, and auto_chi flag.
        If only num_comps it picks a random point with num_comps components 
        Note: this will use the chi_matrix defined in the class
        If you want to pick a random chi_matrix then turn on auto_chi and it will automatically pick a chi_matrix
        with the correct se
        If you select phi_H it will evolve a point with phi_H as part of it. If you len(phi_H) == num_comps then you can
        specify the point exactly. 

        """
        #self.H_dim = len(phi_H)
        self.num_comps = num_comps 
        h_dim = len(phi_H)
        if h_dim > num_comps:
            raise Exception("Num_comps too small")
        if auto_chi:
            self.reset_rand_chi_matrix(num_comps, self.chi_strength)
        if h_dim == 0:
            
            point = np.random.rand(1,num_comps)
            point = point/(np.sum(point))
            point = np.squeeze(point)
        else:

            #print(point)
            if h_dim == num_comps:
                point = phi_H
            elif h_dim < num_comps:
                #phi_H = np.array(phi_H)
                point = np.random.rand(num_comps-h_dim)
                #point = np.squeeze(point)
                point = (1-sum(phi_H))*point/(np.sum(point))
                point = np.concatenate([point[0:1], phi_H, point[1:]])
        print(f"Evolved Point: {point}")
        evolved_point = flory.find_coexisting_phases(num_comps, self.chi_matrix, point, progress = False)
            
        return evolved_point
        
    def evolve_flory_simple(self,point):
        '''Takes a point and evolves it returning the flory object phases '''
        self.chi_matrix = gen.chi_matrix_w_sol(len(point),self.chi_strength)
        phases = flory.find_coexisting_phases(len(point),self.chi_matrix,point, progress = False)
        return phases

   
    def evolve_components_flory(self, phi_H):
        ''' 
        Main Function
        - Takes a phi_H and phi_in and generates num_points:
                point = [phi_in, phi_H_1 ... , phi_H_n, phi_out, phi_sol]
        - phi_in points = [phi_in_min , ... , phi_in_max] -> [0,1]
        - Evolves each of the points 
        - Returns a numpy array of the most dilute phase for each phi_in point
        '''
        num_comps = len(phi_H) + 2
        #generate comp_points
        
        # picks all the phi_in points given the threshold
        phi_in_points = np.linspace(self.phi_in_min,self.phi_in_thresh, self.num_of_points)
        phi_in_points = phi_in_points.reshape(self.num_of_points,1)
        
        # generates phi_H cords and phi_out cords
        phi_H_points = np.tile(phi_H, [self.num_of_points,1])
        out_array = self.phi_out*np.ones((self.num_of_points,1))
        
        # comp_points is all the points in the phase space that we are going to evolve
        comp_points = np.concatenate((phi_in_points, phi_H_points, out_array), axis = 1)
        
        #add solvent component
        phi_sol_points = []
        for phi_in in phi_in_points:
            phi_sol= 1 - (phi_in + self.phi_out + sum(phi_H))
            if phi_sol < 0:
                raise Exception("No Negative Solvent")
            phi_sol_points.append(phi_sol)
            
        #adds the solvent to the point    
        phi_sol_points = np.array(phi_sol_points)
        comp_points = np.concatenate((comp_points,phi_sol_points), axis = 1)
        #print(comp_points)
        
        #norm_points = np.linspace(0,1, self.num_of_points)

        #Evolves all the points and adds them to evolvedd_comp_points
        evolved_comp_points = []
        for point in tqdm(comp_points):
            phase = flory.find_coexisting_phases(self.num_comps+1,self.chi_matrix,point, progress = False)
            evolved_comp_points.append(phase)

        #Calculates the dilute phases for each point
        dilute_comp_points = []
        for phase in evolved_comp_points:
            x = gen.most_dilute_flory(phase)
            dilute_comp_points.append(x)

        #Plots the most dilute phases for each point 
        dilute_out_comp_arr = np.array(dilute_comp_points)
        test_plot = plots.plot_comp_volume_fraction_v2(dilute_out_comp_arr,self.output_readout_func)
    
        
        return dilute_out_comp_arr
    
    def plot_data_v3(self, data, readoutFlag = True, normFlag = False):
        '''
        Plots each of the components of the volume fraction as a scatter plot 
        Data is a numpy array
        '''
        num_comps = data.shape[1]
        num_of_points = data.shape[0]
        norm_points = norm_points = np.linspace(0,1, self.num_of_points)
    
        
        # Label the Plot
        plt.figure
        plt.title('Plot of Component Volume Fractions')
        plt.xlabel('Initial Phi_In')
        plt.ylabel('Volume Fraction')
        plt.ylim(0, 1)
        
        # Create scatter plot for the different components and labels them
        for i in range(1,self.H_dim+1):
            plt.scatter(norm_points, data[:,i], color='green', marker='o', label = f'Phi_H {i}')
        plt.scatter(norm_points,data[:,0], color='blue', marker='o',label = 'Phi_In')

        # Create scatter plot for the phi_out_vector components
        for i in range(len(self.phi_out_vector)):
            plt.scatter(norm_points, data[:,-i-2], color='red', marker='o', label = f'Phi_Out {i+1}')
        plt.scatter(norm_points, data[:,-1], color='yellow', marker='o', label = 'Phi_Sol')

        # Calculates the readout of each point 
        output_comp_points = []
        for point in data:
            out_point = point[-len(self.phi_out_vector)-1:-1]
            print(out_point)
            if readoutFlag == True:
                x = self.phi_out_readout_func(out_point)
            else:
                x = 0
            output_comp_points.append(x)
        output_comp_points = np.array(output_comp_points)
        print(output_comp_points)
        if normFlag:
            norm_factor = np.amax(output_comp_points)
            norm_factor = 1/norm_factor
            print(f"NORM FACTOR {norm_factor}")
            output_comp_points = output_comp_points * norm_factor

        #Plots the readout on the plot with purple
        
        plt.scatter(norm_points,output_comp_points, color = 'purple', marker='x')
        plt.plot(norm_points,output_comp_points, color = 'purple', linewidth = 2.0, label = 'Readout Value')
        plt.legend()
        # Show plot
        plt.show()
            
        
    

    def evolve_components_flory_multi(self,phi_H):
        ''' 
        Same as evolve_components_flory but allows for N-dim phi_out
        NOTE: This prints lots of values. Good for debugging
        - Takes a phi_H and phi_in_range and generates num_points:
                point = [phi_in, phi_H_1 ... , phi_H_n, phi_out, phi_sol]
        - phi_in points = [phi_in_min , ... , phi_in_max] -> [0,1]
        - Evolves each of the points 
        - Returns a numpy array of the most dilute phase for each phi_in point
        '''
        #generate comp_points: ->
        # picks all the phi_in cords given the threshold
        #print(self.phi_out_vector)
        phi_in_min = self.phi_in_range[0]
        phi_in_max = self.phi_in_range[1]
        phi_in_points = np.linspace(phi_in_min,phi_in_max, self.num_of_points)
        phi_in_points = phi_in_points.reshape(self.num_of_points,1)
        
        # Generates the phi_H and phi_out_vector cords
        phi_H_points = np.tile(phi_H, [self.num_of_points,1])
        out_array = np.tile(self.phi_out_vector, [self.num_of_points, 1])
    
        # comp_points is all the points in the phase space that we are going to evolve
        comp_points = np.concatenate((phi_in_points, phi_H_points, out_array), axis = 1)
        
        #add solvent component
        phi_sol_points = []
        for phi_in in phi_in_points:
            phi_sol= 1 - (phi_in + sum(self.phi_out_vector) + sum(phi_H)) 
            if phi_sol < 0:
                raise Exception("No Negative Solvent")
            phi_sol_points.append(phi_sol)

        # Adds solvent component to points
        phi_sol_points = np.array(phi_sol_points)
        comp_points = np.concatenate((comp_points,phi_sol_points), axis = 1)
        print(comp_points)
        
        norm_points = np.linspace(0,1, self.num_of_points)

        # Evolves all the points
        evolved_comp_points = []
        for point in tqdm(comp_points):
            #print(self.chi_matrix)
            #print(self.num_comps)
            phase = flory.find_coexisting_phases(self.num_comps,self.chi_matrix,point, progress = False)
            #print(f"Number of Phases {len(phase.fractions)}")
            #print(f"Phase Fractions + {phase.fractions}")
            #print(f"Phase Volumes + {phase.volumes}")
            evolved_comp_points.append(phase)

        # Returns the dilute phases for each point
        dilute_comp_points = []
        for phase in evolved_comp_points:
            x = gen.most_dilute_flory(phase)
            dilute_comp_points.append(x)

        dilute_out_comp_arr = np.array(dilute_comp_points)

        #Peaks the first few data points
        print(f"Data: {dilute_out_comp_arr[0:4]}")
        
        #Plots the data using plot_data_v3
        test_plot = self.plot_data_v3(dilute_out_comp_arr)
    
        
        return evolved_comp_points
           

    def evolve_components_flory_multi_clean(self,phi_H,plotFlag = False, readoutFlag = True, normFlag = False):
        ''' 
        Same as evolve_components_flory_multi but removes the print statements and adds the plot flag
        NOTE: plotFlag allows you to for each point plot the volume fractions of the components 
        for all the phasesand the normalized volume fractions.
        - Takes a phi_H and phi_in_range and generates num_points:
                point = [phi_in, phi_H_1 ... , phi_H_n, phi_out, phi_sol]
        - phi_in points = [phi_in_min , ... , phi_in_max] -> [0,1]
        - Evolves each of the points 
        - Returns a numpy array of the most dilute phase for each phi_in point
        '''
    
        #generate comp_points: ->
        # picks all the phi_in cords given the threshold
        phi_in_min = self.phi_in_range[0]
        phi_in_max = self.phi_in_range[1]
        phi_in_points = np.linspace(phi_in_min,phi_in_max, self.num_of_points)
        phi_in_points = phi_in_points.reshape(self.num_of_points,1)

        # Creates phi_H and phi_out_vector cords
        phi_H_points = np.tile(phi_H, [self.num_of_points,1])
        out_array = np.tile(self.phi_out_vector, [self.num_of_points, 1])

        # comp_points is all the points in the phase space that we are going to evolve
        comp_points = np.concatenate((phi_in_points, phi_H_points, out_array), axis = 1)

        #Calculate solvent component
        phi_sol_points = []
        for phi_in in phi_in_points:
            phi_sol= 1 - (phi_in + sum(self.phi_out_vector) + sum(phi_H)) 
            if phi_sol < 0:
                raise Exception("No Negative Solvent")
            phi_sol_points.append(phi_sol)

        #Add the solvent component to the points
        phi_sol_points = np.array(phi_sol_points)
        comp_points = np.concatenate((comp_points,phi_sol_points), axis = 1)

        #Evolves all the points and plots if plotFlag == True
        evolved_comp_points = []
        for point in tqdm(comp_points):
            phase = flory.find_coexisting_phases(self.num_comps,self.chi_matrix,point, progress = False)
            #print(f"Number of Phases {len(phase.fractions)}")
            #print(f"Phase Fractions + {phase.fractions}")
            #print(f"Phase Volumes + {phase.volumes}")
            if plotFlag:
                print(f"POINT: {point}")
                plots.plot_phases(phase)
                plots.plot_phases_vol_norm(phase)
            evolved_comp_points.append(phase)

        #Finds the dilute phases
        dilute_comp_points = []
        for phase in evolved_comp_points:
            x = gen.most_dilute_flory(phase)
            dilute_comp_points.append(x)
            
        #Plots the Data
        dilute_out_comp_arr = np.array(dilute_comp_points)
        
        test_plot = self.plot_data_v3(dilute_out_comp_arr,readoutFlag = readoutFlag, normFlag = normFlag )
        
        return dilute_out_comp_arr
    
    def evolve_components_flory_all_phases(self,phi_H,plotFlag = False, readoutFlag = True, normFlag = False):
        ''' 
        Same as evolve_components_flory_multi but removes the print statements and adds the plot flag
        NOTE: plotFlag allows you to for each point plot the volume fractions of the components 
        for all the phasesand the normalized volume fractions.
        - Takes a phi_H and phi_in_range and generates num_points:
                point = [phi_in, phi_H_1 ... , phi_H_n, phi_out, phi_sol]
        - phi_in points = [phi_in_min , ... , phi_in_max] -> [0,1]
        - Evolves each of the points 
        - Returns a numpy array of the most dilute phase for each phi_in point
        '''
    
        #generate comp_points: ->
        # picks all the phi_in cords given the threshold
        phi_in_min = self.phi_in_range[0]
        phi_in_max = self.phi_in_range[1]
        phi_in_points = np.linspace(phi_in_min,phi_in_max, self.num_of_points)
        phi_in_points = phi_in_points.reshape(self.num_of_points,1)

        # Creates phi_H and phi_out_vector cords
        phi_H_points = np.tile(phi_H, [self.num_of_points,1])
        out_array = np.tile(self.phi_out_vector, [self.num_of_points, 1])

        # comp_points is all the points in the phase space that we are going to evolve
        comp_points = np.concatenate((phi_in_points, phi_H_points, out_array), axis = 1)

        #Calculate solvent component
        phi_sol_points = []
        for phi_in in phi_in_points:
            phi_sol= 1 - (phi_in + sum(self.phi_out_vector) + sum(phi_H)) 
            if phi_sol < 0:
                raise Exception("No Negative Solvent")
            phi_sol_points.append(phi_sol)

        #Add the solvent component to the points
        phi_sol_points = np.array(phi_sol_points)
        comp_points = np.concatenate((comp_points,phi_sol_points), axis = 1)

        #Evolves all the points and plots if plotFlag == True
        evolved_comp_points = []
        for point in tqdm(comp_points):
            phase = flory.find_coexisting_phases(self.num_comps,self.chi_matrix,point, progress = False)
            #print(f"Number of Phases {len(phase.fractions)}")
            #print(f"Phase Fractions + {phase.fractions}")
            #print(f"Phase Volumes + {phase.volumes}")
            if plotFlag:
                print(f"POINT: {point}")
                plots.plot_phases(phase)
                plots.plot_phases_vol_norm(phase)
            evolved_comp_points.append(phase)

        return evolved_comp_points
        
    def evolve_components_flory_all_phases_AND(self,phi_H,plotFlag = False, readoutFlag = True, normFlag = False):
        ''' 
        Same as evolve_components_flory_multi but removes the print statements and adds the plot flag
        NOTE: plotFlag allows you to for each point plot the volume fractions of the components 
        for all the phasesand the normalized volume fractions.
        - Takes a phi_H and phi_in_range and generates num_points:
                point = [phi_in, phi_H_1 ... , phi_H_n, phi_out, phi_sol]
        - phi_in points = [phi_in_min , ... , phi_in_max] -> [0,1]
        - Evolves each of the points 
        - Returns a numpy array of the most dilute phase for each phi_in point
        '''
    
        #generate comp_points: ->
        # picks all the phi_in cords given the threshold
        phi_in_min = self.phi_in_range[0]
        phi_in_max = self.phi_in_range[1]
        phi_in_1_points = np.linspace(phi_in_min,phi_in_max, self.num_of_points)
        phi_in_2_points = np.linspace(phi_in_min,phi_in_max, self.num_of_points)
    
        phi_in_1_v,phi_in_2_v = np.meshgrid(phi_in_1_points, phi_in_2_points)

        phi_in_1_v = np.reshape(phi_in_1_v, (self.num_of_points**2,1))
        phi_in_2_v = np.reshape(phi_in_2_v, (self.num_of_points**2,1))
        phi_in_space = np.concatenate((phi_in_1_v, phi_in_2_v), axis = 1)
        

        # Creates phi_H and phi_out_vector cords
        
        phi_H_points = np.tile(phi_H, [self.num_of_points**2,1])
        out_array = np.tile(self.phi_out_vector, [self.num_of_points**2,1])

        # comp_points is all the points in the phase space that we are going to evolve
        comp_points = np.concatenate((phi_in_space, phi_H_points, out_array), axis = 1)
        
        #Calculate solvent component
        phi_sol_points = []
        for phi_in in phi_in_space:
            phi_sol= 1 - (sum(phi_in) + sum(self.phi_out_vector) + sum(phi_H)) 
            if phi_sol < 0:
                raise Exception("No Negative Solvent")
            phi_sol_points.append(phi_sol)

        #Add the solvent component to the points
        phi_sol_points = np.array(phi_sol_points)
        phi_sol_points = np.reshape(phi_sol_points, (self.num_of_points**2,1))
        #print(comp_points)
       # print(phi_sol_points)
        comp_points = np.concatenate((comp_points,phi_sol_points), axis = 1)
        print(comp_points)
        #Evolves all the points and plots if plotFlag == True
        evolved_comp_points = []
        for point in tqdm(comp_points):
            phase = flory.find_coexisting_phases(self.num_comps,self.chi_matrix,point, progress = False)
            #print(f"Number of Phases {len(phase.fractions)}")
            #print(f"Phase Fractions + {phase.fractions}")
            #print(f"Phase Volumes + {phase.volumes}")
            if plotFlag:
                print(f"POINT: {point}")
                plots.plot_phases(phase)
                plots.plot_phases_vol_norm(phase)
            evolved_comp_points.append(phase)

        return evolved_comp_points
        
        

    def generate_data_plane(self,L_x, L_y, x_range, y_range, AND_flag = False):
        """ Loops through a grid of L_x by L_y points over x_range and y_range for an input phi_H. It then takes each point and runs 
        evolve_components_flory_all_phases for each point. It stores this in a pickle file and puts it into llps_data and labels it
        as llps_data/data_plane_{L_x}_{L_y}_strength_{self.chi_strength}.pickle
        It also returns an np array of objects with the all the plots. By plots I mean it has loops through the phi_in_range for 
        the phi_in components and has taken care of them. 
        Note: The object must be prepared by setting the requiste parameters so that it runs before hand. """

        #Creates the grid of spaced phi_x and phi_y spacings
        phi_H_test_x = np.linspace(x_range[0],x_range[1],L_x)
        phi_H_test_y = np.linspace(y_range[0],y_range[1],L_y)
        
        #Creates the data for numpy array of objects.
        if AND_flag:
            data = np.empty((L_x,L_y,self.num_of_points**2),dtype = object)
        else:
            data = np.empty((L_x,L_y,self.num_of_points),dtype = object)

        #Loops through the x and y spacings
        for idx_x,x in enumerate(phi_H_test_x):
            for idx_y,y in enumerate(phi_H_test_y): 
                try:
                    #Creates a phi_H
                    phi_H = [x,y]
                    #Runs evolve_compontns_flory_all_phases on the phi_H
                    if AND_flag:
                        evolved_points = self.evolve_components_flory_all_phases_AND(phi_H)
                    else:
                        evolved_points = self.evolve_components_flory_all_phases(phi_H, plotFlag = False)
                
                    '''
                    for i, point in enumerate(evolved_points):
                        data[idx_x][idx_y][i] = point
                    '''
                    #Stores the evolved_points object into the data array
                    data[idx_x][idx_y] = evolved_points
                except Exception as e:
                    print(f"Error encountered at ({idx_x}, {idx_y}): {e}")
                    data[idx_x][idx_y] = 0

        #Once the loop is done it stores data into a pickle file at the location below

        phase_space = PhaseSpace(self,data)
        chi_strength_str = str(self.chi_strength).replace('.', '_')
        idx = random.randint(1000,2000)
        ps_file_path = f"TEST_{idx}"
        if AND_flag:
            idx = random.randint(1000,2000)
            file_path = f"llps_data/AND_phase_space_{L_x}_{L_y}_strength_{chi_strength_str}_idx{idx}.pickle"
            ps_file_path = f"llps_data/AND_phase_space_obj{L_x}_{L_y}_strength_{chi_strength_str}_idx{idx}.pickle"
        else:
            idx = random.randint(1000,2000)
            file_path = f"llps_data/phase_space_{L_x}_{L_y}_strength_{chi_strength_str}.pickle"
            ps_file_path = f"llps_data/phase_space_obj{L_x}_{L_y}_strength_{chi_strength_str}_idx{idx}.pickle"
            

        with open(file_path,'wb') as file:
            pickle.dump(data,file)
            print(f'File Created at {file_path} ')

        with open(ps_file_path,'wb') as file:
            pickle.dump(phase_space,file)
            print(f'File Created at {ps_file_path} ')
        return data

    def test_phase_space(self):
        test_data = np.zeros((4,4))
        phase_space = PhaseSpace(self,test_data)
            
        return phase_space