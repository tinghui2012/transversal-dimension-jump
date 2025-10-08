import re
import numpy as np
import os, sys
module_path = os.path.abspath(os.path.join('./src/'))
if module_path not in sys.path:
        sys.path.append(module_path)
from Decoders_SpaceTime import BPOSD_Decoder


# function for generating the circuit decoding graph in the circuit level
def GenDecodingGraphs(detector_error_model:str, num_logicals:int):
    items = detector_error_model.split('\n')
    errors = [item for item in items if 'error' in item]
    detectors = [item.split()[1] for item in items if 'detector' in item and 'shift' not in item]

    combined_detectors = detectors
    combined_errors = []
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
        error_dict = {'p':error_p, 'detectors':detectors, 'logicals':flipped_logicals}
        combined_errors.append(error_dict)
    
    # construct the joint check matrix
    H_joint = np.zeros([len(combined_detectors), len(combined_errors)])
    for i in range(len(combined_detectors)):
        for j in range(len(combined_errors)):
            if combined_detectors[i] in combined_errors[j]['detectors']:
                H_joint[i,j] = 1
    # construct the joint logical correction matrix
    logicals = ['L'+str(i) for i in range(num_logicals)]
    L_joint = np.zeros([len(logicals), len(combined_errors)])
    for i in range(len(logicals)):
        for j in range(len(combined_errors)):
            if logicals[i] in combined_errors[j]['logicals']:
                L_joint[i,j] = 1
    
    channel_prob_joint = [error['p'] for error in combined_errors]
        
    return H_joint, L_joint, channel_prob_joint



class BPOSD_Decoding():
    def __init__(self, decoder_params={'max_iter':100, 'bp_method':'min_sum', 'ms_scaling_factor':0.9, 'osd_method':"osd_e", 'osd_order':6}):
        self.decoder = None
        self.L = None
        self.decoder_params = decoder_params
    
    def from_detector_error_model(self, dem=None, num_logicals=None):
        # generate the decoding graphs
        H, L, channel_prob = GenDecodingGraphs(str(dem), num_logicals=num_logicals)
        self.L = L
        
        max_iter = self.decoder_params['max_iter']
        bp_method = self.decoder_params['bp_method']
        ms_scaling_factor = self.decoder_params['ms_scaling_factor']
        osd_method = self.decoder_params['osd_method']
        osd_order = self.decoder_params['osd_order']
        self.decoder = BPOSD_Decoder(h=H, channel_probs=channel_prob,
                                    max_iter=max_iter,
                                    bp_method=bp_method,
                                    ms_scaling_factor=ms_scaling_factor,
                                    osd_method=osd_method,
                                    osd_order=osd_order)
        
    def decode_batch(self, detector_vals):
        detector_historys = [1.0*detector_val for detector_val in detector_vals]
        logical_cors = []
        for i in range(len(detector_historys)):
            detector_values = detector_historys[i]   
            cor = self.decoder.decode(detector_values)
            log_cor = self.L@cor%2
            logical_cors.append(log_cor)

        return np.vstack(logical_cors)