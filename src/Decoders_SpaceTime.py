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

from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

# BPOSD decoder
from bposd import bposd_decoder

class BPOSD_Decoder():
    def __init__(self, h:np.ndarray, channel_probs:np.ndarray, max_iter:int, bp_method:str, 
                ms_scaling_factor:float, osd_method:str, osd_order:int):
        self.decoder = bposd_decoder(
                h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order, )
        self.h = h
    
    def decode(self, synd:np.ndarray):
        self.decoder.decode(synd)
        return self.decoder.osdw_decoding




# First-min BP decoders
from ldpc import bp_decoder

class FirstMinBPDecoder():
    def __init__(self, h:np.ndarray, channel_probs:np.ndarray, max_iter:int, bp_method:str, 
                ms_scaling_factor:float):
        self.decoder = bp_decoder(parity_check_matrix=h,
                                 channel_probs=channel_probs,
                                  max_iter=1,
                                  bp_method=bp_method,
                                  ms_scaling_factor=ms_scaling_factor,)
        self.h = h
        self.max_iter = max_iter
    
    def decode(self, synd:np.ndarray):
        correction = np.zeros(np.shape(self.h)[1])
        current_synd = synd
        iter_counter = 0
        new_correction = self.decoder.decode(current_synd)
        new_synd = (self.h@(new_correction) % 2 + current_synd)%2
        while (np.sum(new_synd) <= np.sum(current_synd)) and (iter_counter < self.max_iter):
            current_synd = new_synd
            correction = (correction + new_correction) % 2
            iter_counter += 1
            
            new_correction = self.decoder.decode(current_synd)
            new_synd = (self.h@(new_correction) % 2 + current_synd)%2
            
        return correction


class BPDecoder():
    def __init__(self, h:np.ndarray, channel_probs:np.ndarray, max_iter:int, bp_method:str, 
                ms_scaling_factor:float):
        self.decoder = bp_decoder(parity_check_matrix=h,
                                 channel_probs=channel_probs,
                                  max_iter=max_iter,
                                  bp_method=bp_method,
                                  ms_scaling_factor=ms_scaling_factor,)
        self.h = h
        self.max_iter = max_iter
    
    def decode(self, synd:np.ndarray):
        correction = self.decoder.decode(synd)         
        return correction


# Build the decoder classes
class DecoderClass(ABC): 
    @abstractmethod
    def GetDecoder(self):
        pass
    
    
class BPOSD_Decoder_Class(DecoderClass): 
    def __init__(self, max_iter_ratio:int, bp_method:str, 
                ms_scaling_factor:float, osd_method:str, osd_order:int):
        '''Initialize the decoder class with the default params'''
        self.decoder_default_params = {'max_iter_ratio':max_iter_ratio, 'bp_method':bp_method, 'ms_scaling_factor':ms_scaling_factor,
                              'osd_method':osd_method, 'osd_order':osd_order}

    def GetDecoder(self, code_and_noise_channel_params):
        """Get the decoder with parameters related to the code and the noise channel"""
        assert 'h' in code_and_noise_channel_params.keys(), f'missing the check matrix h'
        assert 'p_data' in code_and_noise_channel_params.keys(), f'missing the data error prob: p_data'
        
        # Set the parameters related to the code and the noise channel 
        h = code_and_noise_channel_params['h']
#         num_checks, num_qubits = h.shape
        if 'p_syndrome' in code_and_noise_channel_params.keys():
            num_checks, num_qubits = h.shape[0], h.shape[1] - h.shape[0]
            channel_probs = np.hstack([code_and_noise_channel_params['p_data']*np.ones(num_qubits), code_and_noise_channel_params['p_syndrome']*np.ones(num_checks)])
        else:
            num_checks, num_qubits = h.shape
            channel_probs = code_and_noise_channel_params['p_data']*np.ones(num_qubits)
            
        # Set the default parameters
        max_iter = num_qubits/self.decoder_default_params['max_iter_ratio']
        bp_method=self.decoder_default_params['bp_method']
        ms_scaling_factor=self.decoder_default_params['ms_scaling_factor']
        osd_method=self.decoder_default_params['osd_method']
        osd_order=self.decoder_default_params['osd_order']

        decoder = BPOSD_Decoder(
                h=h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order)
        
        return decoder
        
        
class BP_Decoder_Class(DecoderClass): 
    def __init__(self, max_iter_ratio:int, bp_method:str, 
                ms_scaling_factor:float):
        '''Initialize the decoder class with the default params'''
        self.decoder_default_params = {'max_iter_ratio':max_iter_ratio, 'bp_method':bp_method, 'ms_scaling_factor':ms_scaling_factor}

    def GetDecoder(self, code_and_noise_channel_params):
        """Get the decoder with parameters related to the code and the noise channel"""
        assert 'h' in code_and_noise_channel_params.keys(), f'missing the check matrix h'
        assert 'p_data' in code_and_noise_channel_params.keys(), f'missing the data error prob: p_data'
        
        # Set the parameters related to the code and the noise channel 
        h = code_and_noise_channel_params['h']
        if 'p_syndrome' in code_and_noise_channel_params.keys():
            num_checks, num_qubits = h.shape[0], h.shape[1] - h.shape[0]
            channel_probs = np.hstack([code_and_noise_channel_params['p_data']*np.ones(num_qubits), code_and_noise_channel_params['p_syndrome']*np.ones(num_checks)])
        else:
            num_checks, num_qubits = h.shape
            channel_probs = code_and_noise_channel_params['p_data']*np.ones(num_qubits)

        # Set the default parameters
        max_iter = num_qubits/self.decoder_default_params['max_iter_ratio']
        bp_method=self.decoder_default_params['bp_method']
        ms_scaling_factor=self.decoder_default_params['ms_scaling_factor']

        decoder = BPDecoder(h=h,
                                 channel_probs=channel_probs,
                                  max_iter=max_iter,
                                  bp_method=bp_method,
                                  ms_scaling_factor=ms_scaling_factor)
        
        return decoder
        




# space-time decoders (with repeated measurement)
def GetSpaceTimeCheckMat(h, t0):
    num_checks, number_qubits = h.shape
    h_DS = np.hstack([h, np.identity(num_checks)])

    ST_h = np.zeros([t0*num_checks, t0*(number_qubits + num_checks)])
    for i in range(t0):
        for j in range(t0):
            if j == i:
                h_ij = h_DS
            elif (i >= 1) and (j == i - 1):
                h_ij = np.hstack([0*h, np.identity(num_checks)])
            else:
                h_ij = np.hstack([0*h, 0*np.identity(num_checks)])

            ST_h[i*num_checks:(i + 1)*num_checks, j*(number_qubits + num_checks):(j + 1)*(number_qubits + num_checks)] = h_ij 
    return ST_h


# space-time syndrome decoders
    
    
class ST_BP_Decoder_syndrome():
    def __init__(self, h:np.ndarray, p_data:float, p_synd:float, max_iter:int, bp_method:str, ms_scaling_factor:int, num_rep:int):
        self.num_checks, self.num_qubits = h.shape
        self.h = h
        self.num_rep = num_rep
        self.ST_h = GetSpaceTimeCheckMat(h, num_rep) # obtain the space time check matrix
        
        self.space_decoder = bp_decoder(
                self.ST_h,
                channel_probs=np.hstack([p_data*np.ones(self.num_qubits), 
                                                         p_synd*np.ones(self.num_checks)]*num_rep),
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor)
    
    def decode(self, detector_history:np.ndarray):
        syndrome_input = np.reshape(detector_history, detector_history.shape[0]*detector_history.shape[1])
        error_history = self.space_decoder.decode(syndrome_input)
        data_errors = [error_history[i*(self.num_checks + self.num_qubits):(i*(self.num_checks + self.num_qubits) + self.num_qubits)] for i in range(self.num_rep)]
#         print('data errors:', data_errors)
        correction = 0
        for data_error in data_errors:
            correction += data_error
        return correction%2



class ST_BP_Decoder_Class(DecoderClass): 
    def __init__(self, max_iter_ratio:int, bp_method:str, 
                ms_scaling_factor:float):
        '''Initialize the decoder class with the default params'''
        self.decoder_default_params = {'max_iter_ratio':max_iter_ratio, 'bp_method':bp_method, 'ms_scaling_factor':ms_scaling_factor}

    def GetDecoder(self, code_and_noise_channel_params):
        """Get the decoder with parameters related to the code and the noise channel"""
        assert 'h' in code_and_noise_channel_params.keys(), f'missing the check matrix h'
        assert 'p_data' in code_and_noise_channel_params.keys(), f'missing the data error prob: p_data'
        assert 'num_rep' in code_and_noise_channel_params.keys(), f'missing the data error prob: p_data'
        
        # Set the parameters related to the code and the noise channel 
        h = code_and_noise_channel_params['h']
        p_data = code_and_noise_channel_params['p_data']
        num_checks, num_qubits = h.shape
        if 'p_syndrome' in code_and_noise_channel_params.keys():
            p_synd = code_and_noise_channel_params['p_data']
        else:
            p_synd = 0

        # Set the default parameters
        max_iter = num_qubits/self.decoder_default_params['max_iter_ratio']
        bp_method=self.decoder_default_params['bp_method']
        ms_scaling_factor=self.decoder_default_params['ms_scaling_factor']
        num_rep = code_and_noise_channel_params['num_rep']

        decoder = ST_BP_Decoder_syndrome(h=h, p_data=p_data, p_synd=p_synd, max_iter=max_iter, bp_method=bp_method,
                                      ms_scaling_factor=ms_scaling_factor, num_rep=num_rep)
        
        return decoder



class ST_BP_Decoder_Circuit():
    def __init__(self, h:np.ndarray, channel_probs, max_iter:int, bp_method:str, ms_scaling_factor:int):
        self.num_checks, self.num_qubits = h.shape
        self.h = h
        
        self.space_decoder = bp_decoder(h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor)
    
    def decode(self, synd:np.ndarray):
        correction  = self.space_decoder.decode(synd)
        return correction%2


class ST_BPOSD_Decoder_Circuit():
    def __init__(self, h:np.ndarray, channel_probs:np.ndarray, max_iter:int, bp_method:str, 
                ms_scaling_factor:float, osd_method:str, osd_order:int):
        self.decoder = bposd_decoder(
                h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order, )
        self.h = h
    
    def decode(self, synd:np.ndarray):
        self.decoder.decode(synd)
        return self.decoder.osdw_decoding

        

class ST_BP_Decoder_Circuit_Class(DecoderClass): 
    def __init__(self, max_iter_ratio:int, bp_method:str, 
                ms_scaling_factor:float):
        '''Initialize the decoder class with the default params'''
        self.decoder_default_params = {'max_iter_ratio':max_iter_ratio, 'bp_method':bp_method, 'ms_scaling_factor':ms_scaling_factor}

    def GetDecoder(self, code_and_noise_channel_params):
        """Get the decoder with parameters related to the code and the noise channel"""
        assert 'h' in code_and_noise_channel_params.keys(), f'missing the check matrix h'
        assert 'code_h' in code_and_noise_channel_params.keys(), f'missing the code'
        assert 'channel_probs' in code_and_noise_channel_params.keys(), f'missing the channel_probs'
        
        # Set the parameters related to the code and the noise channel 
        h = code_and_noise_channel_params['h']
        num_checks, num_qubits = code_and_noise_channel_params['code_h'].shape
        channel_probs = code_and_noise_channel_params['channel_probs']

        # Set the default parameters
        max_iter = int(num_qubits/self.decoder_default_params['max_iter_ratio'])
        bp_method=self.decoder_default_params['bp_method']
        ms_scaling_factor=self.decoder_default_params['ms_scaling_factor']

        decoder = ST_BP_Decoder_Circuit(h=h, channel_probs=channel_probs, max_iter=max_iter, bp_method=bp_method,
                                      ms_scaling_factor=ms_scaling_factor)
        
        return decoder

class ST_FirstMinBP_Decoder_Circuit_Class(DecoderClass): 
    def __init__(self, max_iter_ratio:int, bp_method:str, 
                ms_scaling_factor:float):
        '''Initialize the decoder class with the default params'''
        self.decoder_default_params = {'max_iter_ratio':max_iter_ratio, 'bp_method':bp_method, 'ms_scaling_factor':ms_scaling_factor}

    def GetDecoder(self, code_and_noise_channel_params):
        """Get the decoder with parameters related to the code and the noise channel"""
        assert 'h' in code_and_noise_channel_params.keys(), f'missing the check matrix h'
        assert 'code_h' in code_and_noise_channel_params.keys(), f'missing the code'
        assert 'channel_probs' in code_and_noise_channel_params.keys(), f'missing the channel_probs'
        
        # Set the parameters related to the code and the noise channel 
        h = code_and_noise_channel_params['h']
        num_checks, num_qubits = code_and_noise_channel_params['code_h'].shape
        channel_probs = code_and_noise_channel_params['channel_probs']

        # Set the default parameters
        max_iter = int(num_qubits/self.decoder_default_params['max_iter_ratio'])
        bp_method=self.decoder_default_params['bp_method']
        ms_scaling_factor=self.decoder_default_params['ms_scaling_factor']

        decoder = FirstMinBPDecoder(h=h, channel_probs=channel_probs, max_iter=max_iter, bp_method=bp_method,
                                      ms_scaling_factor=ms_scaling_factor)
        
        return decoder



class ST_BPOSD_Decoder_Circuit_Class(DecoderClass): 
    def __init__(self, max_iter_ratio:int, bp_method:str, 
                ms_scaling_factor:float, osd_method:str, osd_order:int):
        '''Initialize the decoder class with the default params'''
        self.decoder_default_params = {'max_iter_ratio':max_iter_ratio, 'bp_method':bp_method, 'ms_scaling_factor':ms_scaling_factor,
                              'osd_method':osd_method, 'osd_order':osd_order}

    def GetDecoder(self, code_and_noise_channel_params):
        """Get the decoder with parameters related to the code and the noise channel"""
        assert 'h' in code_and_noise_channel_params.keys(), f'missing the check matrix h'
        assert 'code_h' in code_and_noise_channel_params.keys(), f'missing the code'
        assert 'channel_probs' in code_and_noise_channel_params.keys(), f'missing the channel_probs'
        
        # Set the parameters related to the code and the noise channel 
        h = code_and_noise_channel_params['h']
        num_checks, num_qubits = code_and_noise_channel_params['code_h'].shape
        channel_probs = code_and_noise_channel_params['channel_probs']
#        
        # Set the default parameters
        max_iter = num_qubits/self.decoder_default_params['max_iter_ratio']
        bp_method=self.decoder_default_params['bp_method']
        ms_scaling_factor=self.decoder_default_params['ms_scaling_factor']
        osd_method=self.decoder_default_params['osd_method']
        osd_order=self.decoder_default_params['osd_order']

        decoder = BPOSD_Decoder(
                h=h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order)
        
        return decoder
