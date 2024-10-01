import numpy as np
import sys
sys.path.append('/Users/calvinsmith/dufresne_lab/multicomponent-mixtures-main')
import multicomp as mm
from datetime import datetime
import flory

def generate_combinations_2(num_components, step_size):
    
    combinations = []

    
    def generate(current, remaining, depth):
        if depth == num_components - 1:
            combinations.append(current + [remaining])
            return
        for i in np.arange(0, remaining + step_size, step_size):
            generate(current + [i], remaining - i, depth + 1)

    generate([], 1.0, 0)
    return np.array(combinations)


def replace_negative(matrix,small_val):
    # Replace negative values in the matrix with 0.01
    modified_matrix = np.where(matrix < 0, small_val, matrix)
    return modified_matrix

def thresh(arr, threshold):
    # Create a boolean mask where True indicates that all elements in the row are above the threshold
    mask = np.all(arr >= threshold, axis=1)

    # Use the mask to select rows that meet the condition
    return arr[mask]

def thresh_2(arr, threshold):
    # Initialize an empty list to hold rows that meet the condition
    filtered_rows = []

    # Loop through each row in the array
    for row in arr:
        # Assume all elements are above the threshold initially
        keep_row = True
        
        # Check each element in the row
        for element in row:
            if element < threshold:
                keep_row = False
                break  # Exit the loop early if any element fails the condition

        # If all elements are above the threshold, add the row to the filtered list
        if keep_row:
            filtered_rows.append(row)

    # Return the list of filtered rows as a NumPy array
    return np.array(filtered_rows)


def random_interaction_matrix(
    num_comp: int, chi_mean: float = None, chi_std: float = 1
) -> np.ndarray:
    """create a random interaction matrix

    Args:
        num_comp (int): The component count
        chi_mean (float): The mean interaction strength
        chi_std (float): The standard deviation of the interactions

    Returns:
        The full, symmetric interaction matrix
    """
    if chi_mean is None:
        chi_mean = 3 + 0.4*num_comp
        #chi_mean = 10+0.4*num_comp
        #chi_mean = 0.0001

    # initialize interaction matrix
    chis = np.zeros((num_comp, num_comp))

    # determine random entries
    num_entries = num_comp * (num_comp - 1) // 2
    chi_vals = np.random.normal(chi_mean, chi_std, num_entries)

    # build symmetric  matrix from this
    i, j = np.triu_indices(num_comp, 1)
    chis[i, j] = chi_vals
    chis[j, i] = chi_vals
    
   # chis = np.array([[0,4,4],[4,0,4],[4,4,0]])
    
    return chis

def most_dilute_flory(phase):
    dilute_phase_vol_index = np.argmax(phase.volumes)
    return phase.fractions[dilute_phase_vol_index]


def chi_matrix_w_sol(num_comps,chi_strength):
    chi_matrix = random_interaction_matrix(num_comps,chi_strength)
    chi_matrix = np.vstack([chi_matrix,np.zeros(num_comps)])
    zero_col = np.zeros(num_comps+1)
    zero_col = zero_col[:, np.newaxis]
    chi_matrix = np.hstack([chi_matrix,zero_col])
    return chi_matrix
    
# This function tells it whether it is in bounds so we don't always have to add constraints
#To the optimization function, makes it easier
def in_bounds(phi_H, phi_H_thresh):
    sum = 0
    for phi in phi_H:
        sum += phi
        if phi < 0:
            return False
    if sum > phi_H_thresh:
        return False
    else:
        return True
    
def get_num_of_phases(phi_vector,fh_energy):
    dynamics_params = {
    "initialization_random_std": 5,  # how random the initial guess is
    "acceptance_Js": 0.0002, # how fast the relative volumes are evolved
    "acceptance_omega": 0.002, # how fast the compositions are evolved
    }

    evolve_params = {
    "t_range": 1000000,  # maximum iteration steps
    "dt": 1,  # always unity for FieldLikeRelaxationDynamics
    "interval": 10000,  # the frequency for checking convergence
    "tolerance": 1e-5,  # criteria for determining convergence
    "progress": False,
    "save_intermediate_data": False,
    }
    num_compartments = 64
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy, 
        np.array(phis), # compositions of each compartment
        np.ones(num_compartments) / num_compartments # relative volume of each compartment
    )    
    
    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return len(unique_phases)
    
    

def get_num_of_phases_fast(phi_vector,fh_energy):
    dynamics_params = {
    "initialization_random_std": 5,  # how random the initial guess is
    "acceptance_Js": 0.0002, # how fast the relative volumes are evolved
    "acceptance_omega": 0.002, # how fast the compositions are evolved
    }

    evolve_params = {
    "t_range": 100000,  # maximum iteration steps
    "dt": 1,  # always unity for FieldLikeRelaxationDynamics
    "interval": 10000,  # the frequency for checking convergence
    "tolerance": 1e-5,  # criteria for determining convergence
    "progress": False,
    "save_intermediate_data": False,
    }
    num_compartments = 64
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy, 
        np.array(phis), # compositions of each compartment
        np.ones(num_compartments) / num_compartments # relative volume of each compartment
    )    
    
    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return len(unique_phases)
    
    
    

def get_uniform_random_composition(num_phases: int, num_comps: int) -> np.ndarray:
    """pick concentrations uniform from allowed simplex (sum of fractions < 1)

    Args:
        num_phases (int): the number of phases to pick concentrations for
        num_comps (int): the number of components to use

    Returns:
        The fractions of num_comps components in num_phases phases
    """
    phis = np.empty((num_phases, num_comps))
    for n in range(num_phases):
        phi_max = 1.0
        for d in range(num_comps):
            x = np.random.beta(1, num_comps - d) * phi_max
            phi_max -= x
            phis[n, d] = x
    return phis


def square_matrix(n, x):
    # Create an nxn matrix of x's
    matrix = np.full((n, n), x)

    # Set the diagonal elements to 0
    np.fill_diagonal(matrix, 0)

    return matrix


def get_num_of_phases_v2(phi_vector, fh_energy, compartments):
    dynamics_params = {
        "initialization_random_std": 5,  # how random the initial guess is
        "acceptance_Js": 0.0002,  # how fast the relative volumes are evolved
        "acceptance_omega": 0.002,  # how fast the compositions are evolved
    }

    evolve_params = {
        "t_range": 1000000,  # maximum iteration steps
        "dt": 1,  # always unity for FieldLikeRelaxationDynamics
        "interval": 10000,  # the frequency for checking convergence
        "tolerance": 1e-5,  # criteria for determining convergence
        "progress": False,
        "save_intermediate_data": False,
    }
    num_compartments = compartments
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy,
        np.array(phis),  # compositions of each compartment
        np.ones(num_compartments) / num_compartments  # relative volume of each compartment
    )

    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return len(unique_phases)

 
    
def evolve_phases(phi_vector,fh_energy, compartments):
    dynamics_params = {
        "initialization_random_std": 5,  # how random the initial guess is
        "acceptance_Js": 0.0002,  # how fast the relative volumes are evolved
        "acceptance_omega": 0.002,  # how fast the compositions are evolved
    }

    evolve_params = {
        "t_range": 1000000,  # maximum iteration steps
        "dt": 1,  # always unity for FieldLikeRelaxationDynamics
        "interval": 10000,  # the frequency for checking convergence
        "tolerance": 1e-5,  # criteria for determining convergence
        "progress": False,
        "save_intermediate_data": False,
    }
    num_compartments = compartments
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy,
        np.array(phis),  # compositions of each compartment
        np.ones(num_compartments) / num_compartments  # relative volume of each compartment
    )

    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return unique_phases

def evolve_phases_result(phi_vector,fh_energy, compartments):
    dynamics_params = {
        "initialization_random_std": 5,  # how random the initial guess is
        "acceptance_Js": 0.0002,  # how fast the relative volumes are evolved
        "acceptance_omega": 0.002,  # how fast the compositions are evolved
    }

    evolve_params = {
        "t_range": 1000000,  # maximum iteration steps
        "dt": 1,  # always unity for FieldLikeRelaxationDynamics
        "interval": 10000,  # the frequency for checking convergence
        "tolerance": 1e-5,  # criteria for determining convergence
        "progress": False,
        "save_intermediate_data": False,
    }
    num_compartments = compartments
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy,
        np.array(phis),  # compositions of each compartment
        np.ones(num_compartments) / num_compartments  # relative volume of each compartment
    )

    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    return result


def timing_chi(comp_range,chi):
    num_comps = 3
    time_data = np.zeros((comp_range,2))
    
    for i in range(comp_range):
    
        print(num_comps)
        chi_matrix = square_matrix(num_comps,chi)
        start_time = datetime.now()  # Record start time

        time_data[i][0] = num_comps
        from tqdm.notebook import tqdm
        print("start: " + str(num_comps))
        print("Init ternary_matrix")
        norm_array = normal_array(5,num_comps)
        for row in tqdm(norm_array):

            # Apply the arbitrary function to the row and add the result to the second column
            fh_energy = mm.FloryHuggins(np.array(chi_matrix))
            test = get_num_of_phases_fast(row,fh_energy)
            #print(ternary_matrix[j,1])

        end_time = datetime.now()  # Record end time
        duration = end_time - start_time  # Calculate duration
        total_seconds = duration.total_seconds()
        time_data[i][1] = total_seconds  # Store total seconds instead of string
        print(duration)
        num_comps += 1
        
    return time_data



def timing_rand_chi(comp_range,avg_chi):
    num_comps = 3
    time_data = np.zeros((comp_range,2))
    
    for i in range(comp_range):
    
        print(num_comps)
        chi_matrix = random_interaction_matrix(num_comps,avg_chi)
        start_time = datetime.now()  # Record start time

        time_data[i][0] = num_comps
        from tqdm.notebook import tqdm
        print("start: " + str(num_comps))
        print("Init ternary_matrix")
        norm_array = normal_array(5,num_comps)
        for row in tqdm(norm_array):

            # Apply the arbitrary function to the row and add the result to the second column
            fh_energy = mm.FloryHuggins(np.array(chi_matrix))
            test = get_num_of_phases_fast(row,fh_energy)
            #print(ternary_matrix[j,1])

        end_time = datetime.now()  # Record end time
        duration = end_time - start_time  # Calculate duration
        total_seconds = duration.total_seconds()
        time_data[i][1] = total_seconds  # Store total seconds instead of string
        print(duration)
        num_comps += 1
        
    return time_data


def normal_array(num_rows, num_comps):
    random_matrix = np.random.rand(num_rows, num_comps)
    
    # Normalize each row to sum to 1
    row_sums = random_matrix.sum(axis=1)[:, np.newaxis]  
    normalized_matrix = random_matrix / row_sums
    return normalized_matrix
