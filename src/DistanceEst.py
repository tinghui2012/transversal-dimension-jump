import numpy as np
import bposd
from ldpc.codes import ring_code
from bposd.hgp import hgp
from bposd import bposd_decoder
import multiprocessing as mp
from bposd.css import css_code

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



def DistanceEst_BPOSD(H, L, num_trials=1):    
    # H is the Z or X check matrix, L is the X or Z logical operator matrix
    num_qubits = np.shape(H)[1]
    num_checks = np.shape(H)[0]
    num_logicals = np.shape(L)[0]
    
    # setup the decoder parameters
    # decoder_params = {'channel_probs':0.1*np.ones(num_qubits), 'max_iter':int(num_qubits/20),
    #                  'bp_method':'min_sum', 'ms_scaling_factor':0.9, 'osd_method':'osd_e',
    #                  'osd_order':6}
    
    decoder_params = {'channel_probs':0.1*np.ones(num_qubits), 'max_iter':5, 'bp_method':'min_sum',
                  'ms_scaling_factor':0.4, 'osd_method':'osd_cs', 'osd_order': 9}
    
    def SingleEst():
        # generate random logical operators to anticommute with
        logical = np.zeros(num_logicals)
        while np.sum(logical) == 0:
            logical = np.random.choice([0, 1], size=(num_logicals))@L%2

        combined_check = np.vstack([H, logical])
        combined_syndrome = np.zeros(num_checks + 1)
        combined_syndrome[-1] = 1

        # set up the decoder
        decoder = bposd_decoder(combined_check,
                    channel_probs=decoder_params['channel_probs'],
                    max_iter=decoder_params['max_iter'],
                    bp_method=decoder_params['bp_method'],
                    ms_scaling_factor=decoder_params['ms_scaling_factor'],
                    osd_method=decoder_params['osd_method'],
                    osd_order=decoder_params['osd_order'], )

        corr = decoder.decode(combined_syndrome)
        return np.sum(corr)
    
    # perform SingleEst in parallel
    eval_func = lambda _: SingleEst()      
    distances = parmap(eval_func, [0]*num_trials, nprocs = mp.cpu_count())
    
    return np.min(distances)