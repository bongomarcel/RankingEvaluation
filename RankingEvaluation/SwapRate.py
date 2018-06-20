'''
Created on 12 Mar 2018

@author: marcel
'''

from __future__ import division
from _collections import defaultdict
import numpy as np
import time
import random as rand
import warnings

cur_time = lambda : int(round(time.time()))
# hard-coded parameters
numberOfBins = 21

def swapRate(metricProperties, system_id_rank, queryId_ids, fuzziness, bootstrapSize):
    ''' Computes the following:
        (1) the Achieved Significance Level (ASL).
        (2) the estimated difference required (estDiff) based on the swap method WITH replacement.
        (3) the estimated difference required based on the paired bootstrap test.
        
        References:
        Implementation: Dunaiski, Geldenhuys and Visser. How to evaluate rankings of academic entities using test data. 2018. ()
        Swap Rate (without replacement): Voorhees and Buckley. The effect of topic set size on retrieval experiment error. 2002. (https://doi.org/10.1145/564376.564432)
        ASL and Bootstrap Test (with replacement): Sakai. Evaluating evaluation metrics based on the bootstrap. 2006. (https://doi.org/10.1145/1148170.1148261)
        
    '''
    methodName = metricProperties[0]
    canonicalMethodName = metricProperties[1]
    methodFunction = metricProperties[2]
    cutoff = metricProperties[3]
    totalRanks = metricProperties[4]
    binNames = [i for i in range(numberOfBins)]
    system_query_result = defaultdict(lambda : defaultdict())
    queryIds = queryId_ids.keys()
    systemNames = system_id_rank.keys()
    for system in systemNames:
        for queryId, ids in queryId_ids.iteritems():
            sortedRanks = sorted([system_id_rank[system][id_] for id_ in ids])
            with warnings.catch_warnings(record=True) as f:
                try:
                    result, count = methodFunction(sortedRanks, cutoff, totalRanks)
#                 except TypeError:
#                     print('Warning. Type error when computing: method=%s, system=%s' %(methodName, system))
#                     result, count = np.NaN, np.NaN
                except ValueError:
                    print('Warning. Value error when computing: method=%s, system=%s' %(methodName, system))
                    result, count = np.NaN, np.NaN
            if len(f) > 0:
                print('Warning. Invalid result for: method=%s, result=%s, count=%s, totalRanks=%s, k=%s, requiresTotalRanks=%s' %(methodName, result, count, totalRanks, str(cutoff), requiresTotalRanks))
                return (methodName, canonicalMethodName, cutoff, fuzziness, bootstrapSize, 0, np.nan, np.nan, np.nan, np.nan, np.nan)
            system_query_result[system][queryId] = result
    system_results = defaultdict(lambda : list())
    for system in systemNames:
        for queryId in queryIds:
            system_results[system].append(system_query_result[system][queryId])
    bootstrapId_samples = dict()
    bootstrapId_samples2 = dict()
    for i in range(bootstrapSize):
        bootstrapId_samples[i] = [rand.choice(queryIds) for _ in queryIds]
    for i in range(bootstrapSize):
        bootstrapId_samples2[i] = [rand.choice(queryIds) for _ in queryIds]
    countAllPair = 0
    bin_swapRate = defaultdict(lambda : 0)
    aslCount = 0
    min_value = 10000000000
    max_value = 0
    for b in range(len(bootstrapId_samples)):
        samples1 = bootstrapId_samples[b]
        samples2 = bootstrapId_samples2[b]
        for i in range(len(systemNames)):
            for j in range(i+1, len(systemNames)):
                for query1, query2 in zip(samples1, samples2):
                    result1_i = system_query_result[systemNames[i]][query1]
                    result1_j = system_query_result[systemNames[j]][query1]
                    result2_i = system_query_result[systemNames[i]][query2]
                    result2_j = system_query_result[systemNames[j]][query2]
                    if max_value < max(abs(result1_i - result1_j), abs(result2_i - result2_j)):
                        max_value = max(abs(result1_i - result1_j), abs(result2_i - result2_j))
                    if min_value > min(abs(result1_i - result1_j), abs(result2_i - result2_j)):
                        min_value = min(abs(result1_i - result1_j), abs(result2_i - result2_j))
    if min_value == max_value:
        print('WARNING: min_value == max_value: ', min_value, methodName)
    bins = [min_value + i * ((max_value - min_value)/21/5) for i in range(21)]
    bin_counter = defaultdict(lambda : 0)
    bin_swap_counter = defaultdict(lambda : 0)
    for b in range(len(bootstrapId_samples)):
        samples1 = bootstrapId_samples[b]
        samples2 = bootstrapId_samples2[b]
        for i in range(len(systemNames)):
            for j in range(i+1, len(systemNames)):
                for query1, query2 in zip(samples1, samples2):
                    result1_i = system_query_result[systemNames[i]][query1]
                    result1_j = system_query_result[systemNames[j]][query1]
                    result2_i = system_query_result[systemNames[i]][query2]
                    result2_j = system_query_result[systemNames[j]][query2]
                    D_b = result1_i - result1_j
                    D_b2 = result2_i - result2_j
                    prev_index = None
                    added = False
                    for binIndex in range(len(bins)):
                        if abs(D_b) < bins[binIndex]:
                            bin_counter[binNames[prev_index]] +=1
                            if D_b * D_b2 <= 0:
                                bin_swap_counter[binNames[prev_index]] +=1
                                prev_index = None
                                added = True
                                break
                        prev_index = binIndex
                    if not added:
                        bin_counter[binNames[prev_index]] += 1                                
                        if D_b * D_b2 <= 0:
                            bin_swap_counter[binNames[prev_index]] += 1
                                
        for binName in binNames:
            if bin_counter[binName] > 0:
                bin_swapRate[binName] = bin_swap_counter[binName] / bin_counter[binName]
            else:
                bin_swapRate[binName] = -1
    DIFF = list()
    for system_i in range(len(systemNames)):
        for system_j in range(system_i + 1, len(systemNames)):
            z_vector = list()
            with warnings.catch_warnings(record=True) as w:
                for x_i, y_i in zip(system_results[systemNames[system_i]], system_results[systemNames[system_j]]):
                    z_vector.append(x_i - y_i)
                z_norm = np.average(z_vector)
                if np.std(z_vector) == 0:
                    t_z = 0
                else:
                    t_z = z_norm / (np.std(z_vector) / np.sqrt(len(z_vector)))#, np.std(z_vector, ddof=1)
                w_vector = dict()
                for i, z_i in enumerate(z_vector):
                    queryId = queryIds[i]
                    w_vector[queryId] = (z_i - z_norm)
                asl_count = 0
                t_w_XY = list()
                for bootId, bootSample, in bootstrapId_samples.iteritems():
                    w_vectorB = list()
                    with warnings.catch_warnings(record=True) as v:
                        for boot in bootSample:
                            w_vectorB.append(w_vector[boot])
                        if np.std(w_vectorB) == 0:
                            t_w_b = 0
                        else:
                            t_w_b = (sum(w_vectorB)/len(w_vectorB)) / (np.std(w_vectorB) / np.sqrt(len(w_vectorB)))#, np.std(z_vector, ddof=1)
                        if len(v) > 0:
                            print('warning caught:', methodName, t_w_b, w_vectorB)
                        else:
                            t_w_XY.append((abs(t_w_b), abs(sum(w_vectorB)/len(w_vectorB))))
                            if abs(t_w_b) > 0 and abs(t_w_b) >= abs(t_z):
                                asl_count += 1
                if len(w) > 0:
                    print('warning caught:', methodName, systemNames[system_i], systemNames[system_j], t_z, w)
                asl = asl_count / len(bootstrapId_samples)
                countAllPair += 1
                sorted_t_w_XY = sorted(t_w_XY, key=lambda x:x[0], reverse=True)
                DIFF.append(sorted_t_w_XY[int(fuzziness*len(bootstrapId_samples))-1][1]) 
                if asl < fuzziness:
                    aslCount += 1
    estDiff = max(DIFF)
    binCutoff = -1
    binIndex = 0
    for bin_ in binNames:
        if bin_swapRate[bin_] >= 0 and bin_swapRate[bin_] <= fuzziness:
            binCutoff = bin_
            break
        binIndex += 1
    aslRate = 100 * aslCount / countAllPair
    maxBinValue = bins[-1]
    minBinValue = bins[0]
    if binCutoff == -1:
        binValue = -1
        swapDiffeMin = np.NINF
        swapDiffeMax = np.infty
    else:
        binValue = bins[binCutoff]
        swapDiffeMin = minBinValue + binIndex * (maxBinValue - minBinValue)/len(bins)
        swapDiffeMax = minBinValue + (binIndex+1) * (maxBinValue - minBinValue)/len(bins)
    return (methodName, canonicalMethodName, cutoff, fuzziness, bootstrapSize, aslRate, estDiff, binValue, swapDiffeMin, swapDiffeMax)