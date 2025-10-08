import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
import pickle
import multiprocessing as mp

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

def WERToLFR(wers, k, num_cycles):
    LFR_per_qubit = (1.0 - np.sign(1-2*wers)*(np.abs(1-2*wers))**(num_cycles))/2 #
    LPs = 1.0 - (1-LFR_per_qubit)**(k)
    logical_failure_rate = 1.0 - (1-LPs)**(1/num_cycles) # logical error rate per cycle
    return logical_failure_rate

def Delta_LFR(LFRs_list, k_list, num_samples_list, num_cycles_list):
    delta_LFRs_list = []
    for i in range(len(LFRs_list)):
        k = k_list[i]
        LFRs = LFRs_list[i]
        num_samples = num_samples_list[i]
        num_cycles = num_cycles_list[i]

        delta_LFRs = []

        for LFR, N, n in zip(LFRs, num_samples, num_cycles):
            pL = 1 - (1 - LFR)**n
            delta_LFR = ((1 - pL)**(1/n - 1))/n*np.sqrt(pL*(1 - pL)/N)
            delta_LFRs.append(delta_LFR)
        delta_LFRs_list.append(delta_LFRs)
    return delta_LFRs_list

def WEREst(scaling_params, n, p):
    A, p_c, alpha, beta = scaling_params['A'], scaling_params['p_c'], scaling_params['alpha'], scaling_params['beta']
    return A*(p/p_c)**(alpha*n**beta/2)


def SaveWERs(file_path, eval_p_list, wer_array):
    save_data = np.vstack([eval_p_list, wer_array])
    np.savetxt(file_path, save_data)
    
def LoadWERs(file_path):
    load_data = np.loadtxt(file_path)
    eval_p_list, wer_array = load_data[0,:], load_data[1:,:] 
    return eval_p_list, wer_array

def PlotWERs(eval_p_list, wer_array, save_file_path=None):
    plt.figure(figsize=(6,4))
    plt.plot()
    for eval_error in wer_array:
        plt.plot(eval_p_list, eval_error, 'D--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$p$')
    plt.ylabel('WER')
    
    if save_file_path != None:
        plt.savefig(save_file_path, bbox_inches='tight')  


def EmpericalFit(xdata_tuple, pc, A, alpha, beta):
    p, n = xdata_tuple
    # pl = A*(p/pc)**((alpha*n**beta + 1)/2)
    pl = A*(p/pc)**(alpha*n**beta/2)
    return np.log(pl)

# def CriticalExponentFit(xdata_tuple, pc, A, B, C, alpha, beta):
#     p, n = xdata_tuple
#     x = (p/pc)**(alpha*n**beta/2)
#     # pl = A + B*x + C*x**2
#     pl = A + B*x + C*x**2
#     return np.log(pl)

def CriticalExponentFit(xdata_tuple, pc, A, B, C, alpha, beta):
    p, n = xdata_tuple
    x = (p - pc)*(alpha*n**beta/2)
    # pl = A + B*x + C*x**2
    pl = A + B*x + C*x**2
    # return np.log(pl)
    return pl


def FitWER(sweep_n_list, sweep_p_adpt_list, sweep_wer_list, if_plot=False):
    n_list = [np.array([n]*len(sweep_pl)) for n, sweep_pl in zip(sweep_n_list, sweep_wer_list)]
    fit_X = np.vstack([np.hstack(sweep_p_adpt_list), np.hstack(n_list)])
    fit_Z = np.log(np.hstack(sweep_wer_list))
    
    initial_guess = (0.04, 0.1, 0.2, 0.5)
    popt, pcov = curve_fit(EmpericalFit, fit_X, fit_Z, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    
    p_c, A, alpha, beta = popt
    delta_p_c, delta_A, delta_alpha, delta_beta = perr
    scaling_params = {'p_c': p_c, 'A':A, 'alpha':alpha, 'beta':beta}
    delta_scaling_params = {'p_c': delta_p_c, 'A':delta_A, 'alpha':delta_alpha, 'beta':delta_beta}
    
    if if_plot:
        fitted_pl_list = []
        for sweep_n, sweep_ps in zip(sweep_n_list, sweep_p_adpt_list):
            fitted_pl_list.append([np.exp(EmpericalFit((sweep_p, sweep_n), p_c, A, alpha, beta)) for sweep_p in sweep_ps])
        
        plt.figure()
        for i in range(len(sweep_n_list)):
            plt.plot(sweep_p_adpt_list[i], fitted_pl_list[i], '-', c = 'C%i'%i)
            plt.plot(sweep_p_adpt_list[i], sweep_wer_list[i], 'D', c = 'C%i'%i)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('p')
        plt.ylabel('WER')
    return scaling_params, delta_scaling_params

def FitThreshold(sweep_n_list, sweep_p_adpt_list, sweep_wer_list, if_plot=False):
    n_list = [np.array([n]*len(sweep_pl)) for n, sweep_pl in zip(sweep_n_list, sweep_wer_list)]
    fit_X = np.vstack([np.hstack(sweep_p_adpt_list), np.hstack(n_list)])
    # fit_Z = np.log(np.hstack(sweep_wer_list))
    fit_Z = np.hstack(sweep_wer_list)
    
    initial_guess = (0.006, 0.4, 4, 10, 0.07, 0.9)
    popt, pcov = curve_fit(CriticalExponentFit, fit_X, fit_Z, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    
    p_c, A, B, C, alpha, beta = popt
    delta_p_c, delta_A, delta_B, delta_C, delta_alpha, delta_beta = perr
    scaling_params = {'p_c': p_c, 'A':A, 'B':B, 'C':C, 'alpha':alpha, 'beta':beta}
    delta_scaling_params = {'p_c': delta_p_c, 'A':delta_A, 'B':delta_B, 'C':delta_C, 'alpha':delta_alpha, 'beta':delta_beta}
    
    if if_plot:
        fitted_pl_list = []
        for sweep_n, sweep_ps in zip(sweep_n_list, sweep_p_adpt_list):
            # fitted_pl_list.append([np.exp(CriticalExponentFit((sweep_p, sweep_n), p_c, A, B, C, alpha, beta)) for sweep_p in sweep_ps])
            fitted_pl_list.append([CriticalExponentFit((sweep_p, sweep_n), p_c, A, B, C, alpha, beta) for sweep_p in sweep_ps])
        
        plt.figure()
        for i in range(len(sweep_n_list)):
            plt.plot(sweep_p_adpt_list[i], fitted_pl_list[i], '-', c = 'C%i'%i)
            plt.plot(sweep_p_adpt_list[i], sweep_wer_list[i], 'D', c = 'C%i'%i)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('p')
        plt.ylabel('WER')
    return scaling_params, delta_scaling_params

# def ThresholdEstCodeFamily(sweep_n_list, sweep_p_list, sweep_pl_total_list, if_plot=False):
#     num_p = len(sweep_p_list)
#     num_code = len(sweep_pl_total_list)
    
#     fit_n_list = copy.deepcopy(sweep_n_list)
#     sweep_p_list = list(sweep_p_list)*num_code
#     sweep_n1_list = []
#     for sweep_n in sweep_n_list:
#         sweep_n1_list += [sweep_n]*num_p
#     sweep_n_list = sweep_n1_list
#     sweep_pl_list = list(np.reshape(np.array(sweep_pl_total_list) + 1e-10, [num_p*num_code, ]))
    
#     fit_X = np.vstack([np.reshape(np.array(sweep_p_list), [1, num_p*num_code]), 
#                        np.reshape(np.array(sweep_n_list), [1, num_p*num_code])])
#     fit_Z = np.reshape(np.array(sweep_pl_total_list), [num_p*num_code, ])
#     initial_guess = (0.04, 0.1, 0.2, 0.5)
#     popt, pcov = curve_fit(EmpericalFit, fit_X, fit_Z, p0=initial_guess)
#     perr = np.sqrt(np.diag(pcov))
    
#     # plot
#     p_c, A, alpha, beta = popt
#     fit_p_list = list(set(sweep_p_list))
#     fit_pl_list = np.reshape(np.array(sweep_pl_list), [len(fit_n_list), len(fit_p_list)])
#     if if_plot:
#         fitted_pl_list = []
#         for sweep_n in fit_n_list:
#             fitted_pl_list.append([EmpericalFit((sweep_p, sweep_n), p_c, A, alpha, beta) for sweep_p in fit_p_list])
        
#         plt.figure()
#         for i in range(len(fit_n_list)):
#             plt.plot(fit_p_list, fitted_pl_list[i], '-', c = 'C%i'%i)
#             plt.plot(sweep_p_list[:num_p], sweep_pl_list[i*num_p:(i + 1)*num_p], 'D', c = 'C%i'%i)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('p')
#         plt.ylabel('WER')
    
#     print('p_c:', popt[0])
    
#     return p_c, A, alpha, beta

def FitK(n, alpha_k, beta_k):
    return alpha_k*n**(beta_k)

def RateEstCodeFamily(sweep_n_list, sweep_k_list):
    initial_guess = (0.1, 1)
    popt, pcov = curve_fit(FitK, np.array(sweep_n_list), np.array(sweep_k_list), p0=initial_guess)
    return popt


def p_idling(code_type:str, n, pg):
    tau_t = 50e-6
    a_p = 0.02*1e-6/(1e-6)**2
    d = 5e-6
    
    if code_type == 'hgp':
        L = np.sqrt(2*n)
    elif code_type == 'lp':
        L = 2*n/8
    
    t1 = 2*tau_t*np.log(L)
    t2 = (3 + 2*np.sqrt(2))*np.sqrt(6*L*d/a_p)
    t = t1 + t2
    
#     coh_time = 2
    coh_time = 10
    pi0 = t/coh_time
    
    # assume that the idling error also changes with the gate error
    pg0 = 0.5e-2
    pi = pi0*pg/pg0
    
    return pi