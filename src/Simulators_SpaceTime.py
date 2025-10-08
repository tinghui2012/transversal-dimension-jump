import numpy as np
import matplotlib.pyplot as plt
from graph_tools import Graph
import networkx as nx
import random
import copy
import time
import json

import ldpc
import bposd

from bposd.css_decode_sim import css_decode_sim
from bposd.hgp import hgp
import pickle

import multiprocessing as mp
import random
from ldpc import bp_decoder
from bposd import bposd_decoder

from scipy.optimize import curve_fit
import stim

import sys
sys.path.append("./")
from Decoders_SpaceTime import *
from ErrorPlugin import *
from CircuitScheduling import ColorationCircuit, RandomCircuit, ColorProductCircuit

# Modify the multiprocessing functions
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


## Save the chosen hgps
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)


# function for generating the circuit decoding graph in the circuit level
def GenFaultHyperGraph(detector_error_model:str, num_rounds:int, num_rep:int, num_logicals:int):
    # if the error model string starts with "repeat" block, remove it
    
    items = detector_error_model.split('\n')
    errors = [item for item in items if 'error' in item]
    detectors = [item for item in items if 'detector' in item and 'shift' not in item]
    shifts_indices = np.where(np.array(items) == 'shift_detectors(1) 0')[0] - len(errors) + 1
    num_detectors_each_cycle = []
    for i in range(len(shifts_indices)):
        if i == 0:
            None
        else:
            num_detectors_each_cycle.append(shifts_indices[i] - shifts_indices[i - 1] - 1)
    num_detectors_each_cycle.append(len(detectors) - np.sum(num_detectors_each_cycle))
    
    layered_dectectors = []
    layered_dectectors.append(detectors[:num_detectors_each_cycle[0]]) # first layer
    layered_dectectors.append(detectors[-num_detectors_each_cycle[-1]:]) # last layer
    
    layered_dectectors = [[dectector_str.split()[1] for dectector_str in dectectors_each_layer] for dectectors_each_layer in layered_dectectors]
    
    layered_errors = [[] for i in range(2)]
    for error in errors:
        error_list = error.split()
        #error_p = float(re.findall("\d+\.\d+", error_list[0])[0])
        non_e_pattern = "\d+\.\d+"
        e_pattern = r'([\d]+\.[\d]+e-[\d]+)'
        e_matches = re.findall(e_pattern, error_list[0])
        if e_matches:
            error_p = float(e_matches[0])
        else:
            non_e_matches = re.findall("\d+\.\d+", error_list[0])
            error_p = float(non_e_matches[0])

        detectors = error_list[1:]
        flipped_logicals = [item for item in error_list if 'L' in item]
        occupied_layers = [j for j in range(len(layered_dectectors)) if set(detectors).intersection(set(layered_dectectors[j]))]
        if occupied_layers != []:
            layer = occupied_layers[0]
            detectors = list(set(detectors).intersection(layered_dectectors[layer]))
            error_dict = {'layer':layer, 'p':error_p, 'detectors':detectors, 'logicals':flipped_logicals}
            layered_errors[layer].append(error_dict)
    
    # obtain the check matrix for each layer
    H_list = []
    for error_each_layer, detector_each_layer in zip(layered_errors, layered_dectectors):
        H_each_layer = np.zeros([len(detector_each_layer), len(error_each_layer)])
        for i in range(len(detector_each_layer)):
            for j in range(len(error_each_layer)):
                if detector_each_layer[i] in error_each_layer[j]['detectors']:
                    H_each_layer[i,j] = 1
        H_list.append(H_each_layer)
    
    # obtain the channel probability for each layer
    channel_prob_list = [[error['p'] for error in error_each_layer] for error_each_layer in layered_errors]

    # obtain the matrix for the flipped logicals for each layer
    L_list = []
    logicals = ['L'+str(i) for i in range(num_logicals)]
    for error_each_layer in layered_errors:
        L_each_layer = np.zeros([len(logicals), len(error_each_layer)])
        for i in range(len(logicals)):
            for j in range(len(error_each_layer)):
                if logicals[i] in error_each_layer[j]['logicals']:
                    L_each_layer[i,j] = 1
        L_list.append(L_each_layer)
    
#     return layered_errors, layered_dectectors 
    return H_list, L_list, channel_prob_list
#     return circuit_fault_graph


# function for generating the circuit fault graph in the circuit level
def GenCorrecHyperGraph(detector_error_model:str, num_rounds:int, num_rep:int, num_checks:int, num_logicals:int):
    # if the error model string starts with "repeat" block, remove it
    
    items = detector_error_model.split('\n')
    errors = [item for item in items if 'error' in item]
    detectors = [item for item in items if 'detector' in item and 'shift' not in item]
    shifts_indices = np.where(np.array(items) == 'shift_detectors(1) 0')[0] - len(errors) + 1
    num_detectors_each_cycle = []
    for i in range(len(shifts_indices)):
        if i == 0:
            None
        else:
            num_detectors_each_cycle.append(shifts_indices[i] - shifts_indices[i - 1] - 1)
    num_detectors_each_cycle.append(len(detectors) - np.sum(num_detectors_each_cycle))
    
    layered_dectectors = []
    layered_dectectors.append(detectors[:num_detectors_each_cycle[0]]) # first layer
#     layered_dectectors.append(detectors[:num_detectors_each_cycle[1]]) # second layer
    layered_dectectors.append(detectors[-num_detectors_each_cycle[-1]:]) # last layer
    
    layered_dectectors = [[dectector_str.split()[1] for dectector_str in dectectors_each_layer] for dectectors_each_layer in layered_dectectors]
    relevant_detectors = layered_dectectors[0] + layered_dectectors[1]
    num_detectors_L1, num_detectors_L2 = len(layered_dectectors[0]), len(layered_dectectors[1])
    
    first_layer_errors = []
    for error in errors:
        error_list = error.split()
        error_p = float(re.findall("\d+\.\d+", error_list[0])[0])
        detectors = error_list[1:]
        flipped_logicals = [item for item in error_list if 'L' in item]
        occupied_layers = [j for j in range(len(layered_dectectors)) if set(detectors).intersection(set(layered_dectectors[j]))]
        if occupied_layers != []:
            layer = occupied_layers[0]
            if layer == 0:
                detectors = list(set(detectors).intersection(set(relevant_detectors)))
                error_dict = {'layer':layer, 'p':error_p, 'detectors':detectors, 'logicals':flipped_logicals}
                first_layer_errors.append(error_dict)
    
#     print('first_layer_errors:', first_layer_errors)
    # obtain the check matrix for the first layer
    num_detectors = len(relevant_detectors)
#     print('relevent decoders:', relevant_detectors)
    H = np.zeros([num_detectors, len(first_layer_errors)])
    for i in range(num_detectors):
        for j in range(len(first_layer_errors)):
            if relevant_detectors[i] in first_layer_errors[j]['detectors']:
                H[i,j] = 1
    
    H_space_cor = np.zeros([num_checks, len(first_layer_errors)])
    for i in range(num_rep + 1):
        H_space_cor += H[i*num_checks:(i+1)*num_checks, :]
    H_space_cor = H_space_cor%2
    
    return H_space_cor
#     return H

# Circuit-level simulator
class CodeSimulator_Circuit_SpaceTime():
    def __init__(self, code=None, decoder1_z=None, decoder1_x=None, decoder2_z=None, decoder2_x=None, 
        pi0=1, num_cycles=1, num_rep=1,
                 error_params=None, eval_logical_type='Z', circuit_type = 'coloration', rand_scheduling_seed = 0):
         
        if eval_logical_type == 'X':
            # swap X and Z components of the code and the decoder

            temp = copy.deepcopy(code.hz)
            code.hz = code.hx
            code.hx = temp

            temp = copy.deepcopy(code.lz)
            code.lz = code.lx
            code.lx = temp

            decoder1_z = decoder1_x
            decoder2_z = decoder2_x


        self.eval_code = code
        ## Modify the code to add new varaible nodes for syndrome errors
        self.hx_ext = np.hstack([code.hx, np.identity(np.shape(code.hx)[0])])
        self.hz_ext = np.hstack([code.hz, np.identity(np.shape(code.hz)[0])])
        
        self.decoder1_z = decoder1_z
        self.decoder2_z = decoder2_z
        self.N = code.N
        self.K = code.K
        self.pi0 = pi0
        # self.pz = p
        # self.synd_prob = p
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
        self.num_cycles = num_cycles
        self.num_rep = num_rep
        self.num_rounds = int((self.num_cycles - 1)/self.num_rep)
        assert np.abs((self.num_cycles - 1)/self.num_rep - self.num_rounds) <= 1e-2
        self.circuit = stim.Circuit()
        self.fault_circuit = stim.Circuit()
        self.error_params = error_params
        
        self.max_stab_weight_x = int(np.max(np.sum(code.hx, axis = 1)))
        self.max_stab_weight_z = int(np.max(np.sum(code.hz, axis = 1)))
        
        # Obtain the circuit schedulings based on the circuit_type
        if circuit_type == 'random':
            self.scheduling_X = RandomCircuit(code.hx)
            self.scheduling_Z = RandomCircuit(code.hz)
                
        elif circuit_type == 'coloration':
            self.scheduling_X = ColorationCircuit(code.hx)
            self.scheduling_Z = ColorationCircuit(code.hz)

        elif circuit_type == 'colorproduct':
            self.scheduling_X = ColorProductCircuit(code.hx, 'X')
            self.scheduling_Z = ColorProductCircuit(code.hz, 'Z')
        
        self.num_logicals = self.eval_code.lx.shape[0]
        self.num_checks = self.eval_code.hx.shape[0]
        
        self.detector_sampler = None
        
        self.circuit_graph = None
        
        self.h1_space_cor = None
        
    def _generate_circuit(self):
        '''
        Generate a stim circuit
        '''
        # Set the circuit parameters
        hx = self.eval_code.hx
        hz = self.eval_code.hz
        lx = self.eval_code.lx
        data_indices = list(np.arange(0, np.shape(hx)[1]))
        n = len(data_indices)
        n_Z_ancilla, n_X_ancilla = np.shape(hz)[0], np.shape(hx)[0]
        Z_ancilla_indices = list(np.arange(n, n + n_Z_ancilla))
        X_ancilla_indices = list(np.arange(n + n_Z_ancilla, n + n_Z_ancilla + n_X_ancilla))

        z_stab_weight = int(np.sum(hz[0,:]))
        x_stab_weight = int(np.sum(hx[0,:]))
        
        
        ## Initialization layer
        circuit_init = stim.Circuit()
        circuit_init.append("RX", data_indices)
        circuit_init.append("R", X_ancilla_indices + Z_ancilla_indices)
        
#        circuit_init.append("DEPOLARIZE1", data_indices, (self.pz)) # for debug
         # initialization layer for the fault graph
        circuit_init_f = stim.Circuit()
        circuit_init_f.append("RX", data_indices)
        circuit_init_f.append("R", X_ancilla_indices + Z_ancilla_indices)
        
        circuit_init_f.append("DEPOLARIZE1", data_indices, (self.pi0)) 
       
        ## Repeated stabilizer measurement layer
        circuit_stab_meas_rep1 = stim.Circuit()
#         circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (self.pz)) # for debug
        # measurement the X ancillas
        # # Initialize the X ancillas to the + state
        circuit_stab_meas_rep1.append("H", X_ancilla_indices)
        circuit_stab_meas_rep1.append("DEPOLARIZE1", X_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        circuit_stab_meas_rep1.append("TICK")
        # Apply CX gates for the X stabilizers
        for time_step in range(len(self.scheduling_X)):
            # add idling errors for all the qubits during the ancilla shuffling
            idling_qubits = data_indices + X_ancilla_indices
            idling_data_indices = list(copy.deepcopy(data_indices))
            circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (self.error_params['p_idling_gate'])) 
            for j in self.scheduling_X[time_step]:
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                data_index = self.scheduling_X[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_rep1.append("CX", [X_ancilla_index, data_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_rep1.append("TICK")

        # meausure the Z ancillas
        ## initialize the Z ancillas
#         circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (self.pz)) # for debug
        circuit_stab_meas_rep1.append("DEPOLARIZE1", Z_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        circuit_stab_meas_rep1.append("TICK")
        
        # Appy CX gates for the Z stabilziers
        for time_step in range(len(self.scheduling_Z)):
            idling_qubits = data_indices + Z_ancilla_indices
            idling_data_indices = list(copy.deepcopy(data_indices))
            circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (self.error_params['p_idling_gate']))
            for j in self.scheduling_Z[time_step]:
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                data_index = self.scheduling_Z[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_rep1.append("CX", [data_index, Z_ancilla_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_rep1.append("TICK")

        # Measure the ancillas
        circuit_stab_meas_rep1.append("H", X_ancilla_indices)
        circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*self.error_params['p_m'])) # Add the measurement error
        # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_stab_meas_rep1.append("MR", Z_ancilla_indices + X_ancilla_indices)
        
        circuit_stab_meas_rep1.append("SHIFT_COORDS", [], (1))
        for i in range(len(X_ancilla_indices)):
            circuit_stab_meas_rep1.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i)], (0))
        circuit_stab_meas_rep1.append("TICK")
    
        # rep with difference detectors
        circuit_stab_meas_rep2 = stim.Circuit()
        # measurement the X ancillas
        # # Initialize the X ancillas to the + state
#         circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (self.pz)) # for debug
        circuit_stab_meas_rep2.append("H", X_ancilla_indices)
        circuit_stab_meas_rep2.append("DEPOLARIZE1", X_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        circuit_stab_meas_rep2.append("TICK")
        # Apply CX gates for the X stabilizers
        for time_step in range(len(self.scheduling_X)):
            idling_qubits = data_indices + X_ancilla_indices
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (self.error_params['p_idling_gate']))
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in self.scheduling_X[time_step]:
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                data_index = self.scheduling_X[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_rep2.append("CX", [X_ancilla_index, data_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_rep2.append("TICK")

        # meausure the Z ancillas
        ## initialize the Z ancillas
#         circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (self.pz)) # for debug
        circuit_stab_meas_rep2.append("DEPOLARIZE1", Z_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        circuit_stab_meas_rep2.append("TICK")
        # Appy CX gates for the Z stabilziers
        for time_step in range(len(self.scheduling_Z)):
            idling_qubits = data_indices + Z_ancilla_indices
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (self.error_params['p_idling_gate']))
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in self.scheduling_Z[time_step]:
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                data_index = self.scheduling_Z[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_rep2.append("CX", [data_index, Z_ancilla_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_rep2.append("TICK")
        
        # Measure the ancillas
        circuit_stab_meas_rep2.append("H", X_ancilla_indices)
        circuit_stab_meas_rep2.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*self.error_params['p_m'])) # Add the measurement error
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_stab_meas_rep2.append("MR", Z_ancilla_indices + X_ancilla_indices)
        
        for i in range(len(X_ancilla_indices)):
            circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
                                            stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))], (0))
        circuit_stab_meas_rep2.append("TICK")
        
        
        circuit_stab_meas_rep = circuit_stab_meas_rep1 + (self.num_rep - 1)*circuit_stab_meas_rep2
        
        ## Final projective measurement layer
        circuit_final_meas = stim.Circuit()
        
#         circuit_final_meas.append("DEPOLARIZE1", data_indices, (0*self.pz)) # for debug
        
        circuit_final_meas.append("DEPOLARIZE1",  data_indices, (self.error_params['p_m'])) # Add the measurement error
        circuit_final_meas.append("MX", data_indices)
        
        circuit_final_meas.append("SHIFT_COORDS", [], (1))
        # Obtain the syndroms
        for i in range(len(X_ancilla_indices)):
            supported_data_indices = list(np.where(hx[X_ancilla_indices[i] - n - n_Z_ancilla,:] == 1)[0])

            rec_indices = []
            for data_index in supported_data_indices:
                rec_indices.append(- len(data_indices) + data_index)

#             rec_indices.append(- len(X_ancilla_indices) + i - len(data_indices))

            circuit_final_meas.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices], (0))

        # Obtain the logical measurements result
        for i in range(len(lx)):
            logical_X_qubit_indices = list(np.where(lx[i,:] == 1)[0])
            circuit_final_meas.append("OBSERVABLE_INCLUDE", 
                               [stim.target_rec(- len(data_indices) + data_index) for data_index in logical_X_qubit_indices],
                               (i))
        ## Final projective measurement layer for the fault circuit
        circuit_final_meas_f = stim.Circuit()
#         circuit_final_meas_f.append("DEPOLARIZE1", data_indices, (1*self.pz)) # for debug
        circuit_final_meas_f.append("DEPOLARIZE1",  data_indices, (self.error_params['p_m'])) # Add the measurement error
        circuit_final_meas_f.append("MX", data_indices)
        circuit_final_meas_f.append("SHIFT_COORDS", [], (1))
        # Obtain the syndroms
        for i in range(len(X_ancilla_indices)):
            supported_data_indices = list(np.where(hx[X_ancilla_indices[i] - n - n_Z_ancilla,:] == 1)[0])
            rec_indices = []
            for data_index in supported_data_indices:
                rec_indices.append(- len(data_indices) + data_index)
            rec_indices.append(- len(X_ancilla_indices) + i - len(data_indices))
            circuit_final_meas_f.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices], (0))
        # Obtain the logical measurements result
        for i in range(len(lx)):
            logical_X_qubit_indices = list(np.where(lx[i,:] == 1)[0])
            circuit_final_meas_f.append("OBSERVABLE_INCLUDE", 
                               [stim.target_rec(- len(data_indices) + data_index) for data_index in logical_X_qubit_indices],
                               (i))
            
        circuit = circuit_init + self.num_rounds*circuit_stab_meas_rep + circuit_final_meas
        fault_circuit = circuit_init_f + circuit_stab_meas_rep + circuit_final_meas_f
        
        # Add noise
        noisy_circuit = circuit
        noisy_fault_circuit = fault_circuit
        ## Add CX gate errors
        noisy_circuit = AddCXError(noisy_circuit, 'DEPOLARIZE2(%f)' % self.error_params["p_CX"])
        noisy_fault_circuit = AddCXError(noisy_fault_circuit, 'DEPOLARIZE2(%f)' % self.error_params["p_CX"])

        self.circuit = noisy_circuit
        self.fault_circuit = noisy_fault_circuit
        self.detector_sampler = self.circuit.compile_detector_sampler()


    def _generate_circuit_graph(self):
#         fault_circuit = self._generate_fault_circuit()
        fault_circuit = self.fault_circuit
        num_logicals = self.num_logicals
        num_checks = self.num_checks
        num_rep = self.num_rep
        num_rounds = self.num_rounds
        H_list, L_list, channel_prob_list = GenFaultHyperGraph(str(fault_circuit.detector_error_model(flatten_loops=True)), num_rounds=num_rounds,
                                                               num_rep=num_rep, num_logicals=num_logicals)
        # construct the two decoders based on the fault hypergraph
        h1 = H_list[0]
        L1 = L_list[0]
        channel_ps1 = channel_prob_list[0] 

        h2 = H_list[-1]
        L2 = L_list[-1]
        channel_ps2 = channel_prob_list[-1]
        
        self.circuit_graph = {'h1':h1, 'L1':L1, 'channel_ps1':channel_ps1, 'h2':h2, 'L2':L2, 'channel_ps2':channel_ps2}
        
        # get the space correction matrix for h1
        h1_space_cor = GenCorrecHyperGraph(str(fault_circuit.detector_error_model(flatten_loops=True)), num_rounds=num_rounds,
                                                               num_rep=num_rep, num_checks=num_checks, 
                                                               num_logicals=num_logicals)
        self.h1_space_cor = h1_space_cor
        
    def _decoding_samples(self, samples):
        h1, L1 = self.circuit_graph['h1'], self.circuit_graph['L1']
        h2, L2 = self.circuit_graph['h2'], self.circuit_graph['L2']
        
#         print('error params:', self.error_params)
        '''Decoding the samples obtained from the circuit_sampler'''
        syndrome_historys = [1.0*np.reshape(np.array(sample[:-len(self.eval_code.lx)]), [self.num_cycles, np.shape(self.eval_code.hx)[0]]) for sample in samples]
        logical_values = [1.0*np.array(sample[-len(self.eval_code.lx):]) for sample in samples]
        
#         print('syndrome history:', syndrome_historys, 'logical_values:', logical_values)
        if_failure_list = []

        for i in range(len(samples)):
            syndrome_history = syndrome_historys[i]
            logical_value = logical_values[i]

            detector_values_list = []
            for j in range(self.num_rounds):
                detector_values_list.append(np.reshape(syndrome_history[j*self.num_rep:(j + 1)*self.num_rep, :], [self.num_rep*self.num_checks]))
            detector_values_final_round = syndrome_history[self.num_rounds*self.num_rep:,:]
            detector_values_list.append(np.reshape(detector_values_final_round, [detector_values_final_round.shape[0]*detector_values_final_round.shape[1]]))

            # decode for the first (num_rounds) rounds 
            total_syn_cor_space = 0
            total_log_cor = 0
            for j in range(self.num_rounds):
                syn = detector_values_list[j]
                syn[:self.num_checks] = (syn[:self.num_checks] + total_syn_cor_space)%2 # update the first round of syndrome based on the accumulated space correction

                cor = self.decoder1_z.decode(syn)
                syn_cor_space = (self.h1_space_cor@cor)%2
                log_cor = L1@cor%2
                total_syn_cor_space = (total_syn_cor_space + syn_cor_space)%2
                total_log_cor = (total_log_cor + log_cor)%2

            # decode the final round
            final_syn = detector_values_list[-1]
            final_syn = (final_syn + total_syn_cor_space)%2
            final_cor = self.decoder2_z.decode(final_syn)
            final_log_cor = L2@final_cor%2
            total_log_cor = (total_log_cor + final_log_cor)%2

            residual_syn =  (final_syn + h2@final_cor)%2
            residual_logicals = (logical_value + total_log_cor)%2

            if residual_syn.any() or residual_logicals.any():
                if_failure_list.append(1)
            else:
                if_failure_list.append(0)
            # if residual_syn.any():
            #     if_failure_list.append(np.ones(self.eval_code.N))
            # else:
            #     if_failure_list.append(residual_logicals)

        return if_failure_list
    
    
    def LogicalFailureRate(self, num_samples:int):
        num_cycles = self.num_cycles
        # assert int(num_rounds)%2 == 1 # the number of cycles have to be odd in order to have invertible functons from wer to logical error rate

        eval_func = lambda samples: self._decoding_samples(samples)[0]

        batch_size = 20000
        if num_samples <= batch_size: 
            samples = self.detector_sampler.sample(shots=num_samples, append_observables=True)
            sim_if_error_list = parmap(eval_func, [[sample] for sample in samples], nprocs = mp.cpu_count())
        else:
            N = int(num_samples/batch_size)
            num_samples = N*batch_size
            samples = self.detector_sampler.sample(shots=num_samples, append_observables=True)
            sim_if_error_list = []
            for i in range(N):
                sim_if_error_list += parmap(eval_func, [[sample] for sample in samples[i*batch_size:(i + 1)*batch_size]], nprocs = mp.cpu_count())
             
        error_count = np.sum(sim_if_error_list)

        # compute logical failure rate LFR = 1 - (1-p_L)^{1/num_cycle}
        p_L = error_count/num_samples # total logical error probability
        delta_pL = np.sqrt(p_L*(1 - p_L)/num_samples)
        LFR = 1.0 - (1.0 - p_L)**(1/num_cycles)
        delta_LFR = np.sqrt((1 - p_L)**(1/num_cycles - 1)/num_cycles)*delta_pL

        return LFR, delta_LFR


# Wap up a code family
class CodeFamily_SpaceTime():
    def __init__(self, code_list:list, decoder1_class:DecoderClass, decoder2_class:DecoderClass):
        self.code_list = code_list
        self.decoder1_class = decoder1_class
        self.decoder2_class = decoder2_class
    
    def EvalWER(self, noise_model:str, eval_logical_type:str, eval_p_list:list, 
                      num_samples:int, num_cycles=1, num_rep=1, circuit_type='coloration',
                circuit_error_params=None, if_plot=True, if_adaptive=False, adaptive_params=None, pi0 = 1):
        assert noise_model in ['data', 'phenl', 'circuit'], f'noise_model should be one of [data, phenl, circuit]'
        assert eval_logical_type in ['X', 'Z', 'Total'], f'eval_type should be one of [X, Y, Total]'
        
        # Data threshold
        if noise_model == 'data':
            eval_wer_list = []
            for eval_code in self.code_list:
                for eval_p in eval_p_list:
                    p = eval_p*3/2
                    pauli_error_probs = [p/3, p/3, p/3] # depolarizing channel [px, py, pyz]
                    
                    code_and_noise_channel_params = {'h': eval_code.hz, 'p_data': eval_p}
                    decoder_x = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hx, 'p_data': eval_p}
                    decoder_z = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                
                
                    code_simulator = CodeSimulator_DataError(code=eval_code, decoder_x=decoder_x, 
                                      decoder_z=decoder_z, pauli_error_probs=pauli_error_probs,
                                                             eval_logical_type=eval_logical_type)

                    eval_wer_list.append(code_simulator.WordErrorRate(num_samples)[0])
        
        # Phenomenological threshold
        elif noise_model == 'phenl':
            eval_wer_list = []
            for eval_code in self.code_list:
                for eval_p in eval_p_list:
                    p = 3/2*eval_p
                    q = eval_p

                    p_data = p*2/3
                    p_synd = q

                    pauli_error_probs = [p/3, p/3, p/3] # depolarizing channel [px, py, pyz]
                    
                    code_and_noise_channel_params = {'h': eval_code.hz, 'p_data': p_data, 'p_syndrome':p_synd, 'num_rep':num_rep}
                    dec1_x = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hx, 'p_data': p_data, 'p_syndrome':p_synd, 'num_rep':num_rep}
                    dec1_z = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hz, 'p_data': p_data}
                    dec2_x = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hx, 'p_data': p_data}
                    dec2_z = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_simulator = CodeSimulator_SpaceTime(code=eval_code, decoder1_x= dec1_x, decoder1_z= dec1_z,
                                                              decoder2_x=dec2_x, decoder2_z=dec2_z,
                                                              pauli_error_probs=pauli_error_probs, q=q, eval_logical_type=eval_logical_type, num_rep=num_rep)

                    eval_wer_list.append(code_simulator.WordErrorRate(num_cycles=num_cycles, num_samples=num_samples)[0])

        
        # Circuit threshold
        elif noise_model == 'circuit':
            eval_wer_list = []
            eval_p_adapt_list = []
            for eval_code in self.code_list:
                if if_adaptive:
                    WEREst = adaptive_params['WEREst']
                    min_wer = adaptive_params['min_wer']
                    eval_p_list_adpt = [eval_p for eval_p in eval_p_list if WEREst(eval_code.N, eval_p) >= min_wer]
                else:
                    eval_p_list_adpt = eval_p_list

                eval_wer_per_code_list = []
                for eval_p in eval_p_list_adpt:
                    p = eval_p
                    error_params = {"p_i": circuit_error_params["p_i"]*p, "p_state_p": circuit_error_params["p_state_p"]*p, "p_m": circuit_error_params["p_m"]*p, "p_CX":circuit_error_params["p_CX"]*p, 
                    "p_idling_gate": circuit_error_params["p_idling_gate"]*p}

                    # print('error parameters:', error_params)

                    # generate the circuit fault graph using stim
                    circuit_simulator = CodeSimulator_Circuit_SpaceTime(code=eval_code, decoder1_z=None, decoder1_x=None, 
                                          decoder2_z=None, decoder2_x=None, 
                                            pi0=pi0*p, num_cycles=num_cycles, num_rep=num_rep, 
                                          error_params=error_params, eval_logical_type=eval_logical_type, 
                                          circuit_type = circuit_type, rand_scheduling_seed = 1)


                    circuit_simulator._generate_circuit()
                    circuit_simulator._generate_circuit_graph()
                    circuit_fault_graphs = circuit_simulator.circuit_graph

                    code_and_noise_channel_params = {'code_h':eval_code.hx, 'h':circuit_fault_graphs['h1'], 'channel_probs': circuit_fault_graphs['channel_ps1']}
                    dec1_z = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
    
                    code_and_noise_channel_params = {'code_h':eval_code.hx, 'h':circuit_fault_graphs['h2'], 'channel_probs': circuit_fault_graphs['channel_ps2']}
                    dec2_z = self.decoder2_class.GetDecoder(code_and_noise_channel_params)

                    circuit_simulator.decoder1_z = dec1_z 
                    circuit_simulator.decoder2_z = dec2_z

                    eval_wer_per_code_list.append(circuit_simulator.WordErrorRate(num_samples=num_samples)[0])
                    # else:
                    #     target_failures, batch_size, max_batches = adaptive_params['target_failures'], adaptive_params['batch_size'], adaptive_params['max_batches']
                    #     eval_wer_list.append(circuit_simulator.WordErrorRate_TargetFailure(target_failures, batch_size, max_batches)[0])
            
                eval_p_adapt_list.append(np.array(eval_p_list_adpt))
                eval_wer_list.append(np.array(eval_wer_per_code_list))

        else:
            print('no valid error model selected')
                
        return eval_wer_list, eval_p_adapt_list
        
        
        
    def EvalThreshold(self, noise_model:str, eval_logical_type:str, eval_method:str, est_threshold:float,
                      num_samples:int, num_cycles=1, data_synd_noise_ratio=1, circuit_type='coloration', circuit_error_params=None, if_plot=False):
        """Evalutate the threshold of the code family"""
        assert noise_model in ['data', 'phenl', 'circuit'], f'noise_model should be one of [data, phenl, circuit]'
        assert eval_logical_type in ['X', 'Z', 'Total'], f'eval_type should be one of [X, Y, Total]'
        assert eval_method in ['extrapolation'], f'eval_method should be one of [extrapolation]'
        
        if eval_method == 'extrapolation':
            eval_p_list = 10**(np.linspace(np.log10(est_threshold*0.4), np.log10(est_threshold*0.8), 6))
            eval_wer_array = self.EvalWER(noise_model, eval_logical_type, eval_p_list, 
                      num_samples, num_cycles, data_synd_noise_ratio, circuit_type, circuit_error_params, if_plot=False)
            
            return ThresholdEst_extrapolation(eval_p_list, eval_wer_array, if_plot)
        
        
    def EvalSustainableThreshold(self, noise_model:str, eval_logical_type:str, eval_method:str, est_threshold:float, 
                      num_samples_per_cycle:int, num_cycles_list:list, data_synd_noise_ratio=1, circuit_type='coloration', circuit_error_params=None, if_plot=False):
        
        sweep_threshold_list = [self.EvalThreshold(noise_model=noise_model, eval_logical_type=eval_logical_type, eval_method=eval_method, est_threshold=est_threshold, 
                                                   num_samples=int(num_samples_per_cycle/sweep_num_of_cycle), 
                                                   num_cycles=sweep_num_of_cycle, data_synd_noise_ratio=data_synd_noise_ratio, circuit_type=circuit_type, 
                                                   circuit_error_params=circuit_error_params, if_plot=if_plot) for sweep_num_of_cycle in num_cycles_list]
        
        # Fit the sustaimable threshold
        def FitSusThreshold(N, p_sus, p_0, gamma):
            pth = p_sus*(1 - (1 - p_0/p_sus)*np.exp(-gamma*N))
            return pth
        
        initial_guess = (0.01, 0.05, 0.05)
        popt, pcov = curve_fit(FitSusThreshold, np.array(num_cycles_list), np.array(sweep_threshold_list), p0=initial_guess)
        p_sus = popt[0]
        
        plt.figure()
        plt.plot(num_cycles_list, sweep_threshold_list, 'D')
        plt.plot(num_cycles_list, FitSusThreshold(np.array(num_cycles_list), popt[0], popt[1], popt[2]), '-')
        
        return p_sus
        
        
    def EvalEffectiveDistances(self, noise_model:str, eval_logical_type:str, eval_method:str, est_threshold:float,
                      num_samples:int, num_cycles=1, data_synd_noise_ratio=1, circuit_type='coloration', if_plot=False):
        """Evalutate the effective distances of the code family"""
        assert noise_model in ['data', 'phenl', 'circuit'], f'noise_model should be one of [data, phenl, circuit]'
        assert eval_logical_type in ['X', 'Z', 'Total'], f'eval_type should be one of [X, Y, Total]'
        assert eval_method in ['extrapolation'], f'eval_method should be one of [extrapolation]'
        
        
        eval_p_list = 10**(np.linspace(np.log10(est_threshold/6), np.log10(est_threshold/4), 5))
        eval_wer_array = self.EvalWER(noise_model, eval_logical_type, eval_p_list, 
                    num_samples, num_cycles, data_synd_noise_ratio, circuit_type, if_plot=False)
            
        return DistanceEst(eval_p_list, eval_wer_array, if_plot)
