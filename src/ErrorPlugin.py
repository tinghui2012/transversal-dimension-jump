import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
import sinter
from typing import *
import re

## CX gate errors
def AddCXError(circuit:stim.Circuit, error_instruction:str) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique cx instructions
    cx_instructions = re.findall('CX.*\n', circuit_str)
    unique_cx_instructions = list(set(cx_instructions))
    unique_cx_instructions
    
    ## Add gate errors after each cx instruction
    for cx_ins in unique_cx_instructions:
        circuit_str = circuit_str.replace(cx_ins, cx_ins + 
                                      cx_ins.replace('CX', error_instruction))
    
    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit


## CZ gate errors
def AddCZError(circuit:stim.Circuit, error_instruction:str) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique cx instructions
    cz_instructions = re.findall('CZ.*\n', circuit_str)
    unique_cz_instructions = list(set(cz_instructions))
    
    ## Add gate errors after each cx instruction
    for cz_ins in unique_cz_instructions:
        circuit_str = circuit_str.replace(cz_ins, cz_ins + 
                                      cz_ins.replace('CZ', error_instruction))
    
    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit

## Add single-qubit error before each round
def AddSingleQubitErrorBeforeRound(circuit:stim.Circuit, error_instruction:str, target_qubit_indices:list = []) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique reset and (measure & reset) instructions
    reset_instructions = re.findall('\nR .*\n', circuit_str) + re.findall(' R .*\n', circuit_str)
    measure_reset_instructions = re.findall('\nMR .*\n', circuit_str) + re.findall(' MR .*\n', circuit_str)
    unique_reset_instructions = list(set(reset_instructions + measure_reset_instructions))
#     unique_measure_reset_instructions = list(set(reset_instructions))
    
    ## Add single-qubit errors after each reset instruction
    for reset_ins in unique_reset_instructions:
        if target_qubit_indices:
            circuit_str = circuit_str.replace(reset_ins, reset_ins + 
                                              error_instruction + ' ' + ''.join([str(i) + ' ' for i in target_qubit_indices]) + '\n')
    
#     for measure_reset_str in unique_measure_reset_instructions:
#         if target_qubit_indices:
#             circuit_str = circuit_str.replace(reset_ins, reset_ins + 
#                                               error_instruction + ' ' + ''.join([str(i) + ' ' for i in target_qubit_indices]) + '\n')
    
    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit

## Add measurement error
def AddSingleQubitErrorBeforeRound(circuit:stim.Circuit, error_instruction:str, target_qubit_indices:list = []) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique reset and (measure & reset) instructions
    reset_instructions = re.findall('\nR .*\n', circuit_str) + re.findall(' R .*\n', circuit_str)
    measure_reset_instructions = re.findall('\nMR .*\n', circuit_str) + re.findall(' MR .*\n', circuit_str)
    unique_reset_instructions = list(set(reset_instructions + measure_reset_instructions))
#     unique_measure_reset_instructions = list(set(reset_instructions))
    
    ## Add single-qubit errors after each reset instruction
    for reset_ins in unique_reset_instructions:
        if target_qubit_indices:
            circuit_str = circuit_str.replace(reset_ins, reset_ins + 
                                              error_instruction + ' ' + ''.join([str(i) + ' ' for i in target_qubit_indices]) + '\n')
    
#     for measure_reset_str in unique_measure_reset_instructions:
#         if target_qubit_indices:
#             circuit_str = circuit_str.replace(reset_ins, reset_ins + 
#                                               error_instruction + ' ' + ''.join([str(i) + ' ' for i in target_qubit_indices]) + '\n')
    
    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit

## Add measurement error
def AddMeasurementError(circuit:stim.Circuit, meas_p:float) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique reset and (measure & reset) instructions
    measure_instructions = re.findall('\nM .*\n', circuit_str) + re.findall(' M .*\n', circuit_str)
    measure_reset_instructions = re.findall('\nMR .*\n', circuit_str) + re.findall(' MR .*\n', circuit_str)
    unique_measure_instructions = list(set(measure_instructions + measure_reset_instructions))
#     unique_measure_reset_instructions = list(set(reset_instructions))
    
    ## Add single-qubit errors after each reset instruction
    for measure_ins in unique_measure_instructions:
        if 'MR' in measure_ins:
            circuit_str = circuit_str.replace(measure_ins, 
                                         measure_ins.replace('MR', 'X_ERROR(%f)' % meas_p) + measure_ins)
        else:
            circuit_str = circuit_str.replace(measure_ins, 
                                         measure_ins.replace('M', 'X_ERROR(%f)' % meas_p) + measure_ins)

    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit

## Add Idling error (only add idling on the data qubits during ancilla measurements and reset)
def AddIdlingError(circuit:stim.Circuit, error_instruction:str, target_qubit_indices:list = []) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique reset and (measure & reset) instructions
    measure_instructions = re.findall('\nM .*\n', circuit_str) + re.findall(' M .*\n', circuit_str)
    measure_reset_instructions = re.findall('\nMR .*\n', circuit_str) + re.findall(' MR .*\n', circuit_str)
    unique_measure_instructions = list(set(measure_instructions + measure_reset_instructions))
    
    ## Add single-qubit errors after each reset instruction
    for measure_ins in unique_measure_instructions:
        circuit_str = circuit_str.replace(measure_ins, measure_ins + 
                                         error_instruction + ' ' + ''.join([str(i) + ' ' for i in target_qubit_indices]) + '\n')
    
    modified_circuit = stim.Circuit(circuit_str)


    ## Find all the unique measure & reset instructions
    measure_reset_instructions = re.findall('\nMR .*\n', circuit_str) + re.findall(' MR .*\n', circuit_str)
    unique_reset_instructions = list(set(measure_reset_instructions))
    
    ## Add X errors after each reset instruction as reset errors
    for reset_ins in unique_reset_instructions:
        circuit_str = circuit_str.replace(reset_ins, reset_ins + 
                                         error_instruction + ' ' + ''.join([str(i) + ' ' for i in target_qubit_indices]) + '\n')

    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit

## Add Idling error (only add idling on the data qubits during ancilla measurements for now)
def AddResetError(circuit:stim.Circuit, reset_p:float) -> stim.Circuit:
    circuit_str = str(circuit)
    
    ## Find all the unique reset and (measure & reset) instructions
    reset_instructions = re.findall('\nR .*\n', circuit_str) + re.findall(' R .*\n', circuit_str)
    measure_reset_instructions = re.findall('\nMR .*\n', circuit_str) + re.findall(' MR .*\n', circuit_str)
    unique_reset_instructions = list(set(reset_instructions + measure_reset_instructions))
    
    ## Add X errors after each reset instruction as reset errors
    for reset_ins in unique_reset_instructions:
        if 'MR' in reset_ins:
            circuit_str = circuit_str.replace(reset_ins, reset_ins + 
                                         reset_ins.replace('MR', 'X_ERROR(%f)' % reset_p))
        else:
            circuit_str = circuit_str.replace(reset_ins, reset_ins + 
                                         reset_ins.replace('R', 'X_ERROR(%f)' % reset_p) )

    modified_circuit = stim.Circuit(circuit_str)
    return modified_circuit