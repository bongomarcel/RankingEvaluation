
from __future__ import division
from _collections import defaultdict
from scipy import stats
import numpy as np
import warnings
from EvaluationMeasures import isNegativeMethod

def errorTieRates(methodProperties, system_id_rank, queryId_ids, fuzziness):
    '''Computes the following:
        (1) error rate.
        (2) tie rate.
        (3) rel rate: the average percentage of entities used.
        (4) bootstrap significance without replacement.
        
        References:
        Implementation: Dunaiski, Geldenhuys and Visser. How to evaluate rankings of academic entities using test data. 2018. ()
        Error and Tie Rate: Buckley and Voorhees. Evaluating evaluation measure stability. 2000. (https://doi.org/10.1145/345508.345543)
    '''
    
    methodName = methodProperties[0]
    canonicalMethodName = methodProperties[1]
    methodFunction = methodProperties[2]
    cutoff = methodProperties[3]
    totalRanks = methodProperties[4]
    system_query_result = defaultdict(lambda : defaultdict())
    systemNames = system_id_rank.keys()
    query_performanceSum = defaultdict(lambda : 0.0)
    query_relSum = defaultdict(lambda : 0.0)
    query_performanceCount = defaultdict(lambda : 0.0)
    for system in systemNames:
        for queryId, ids in queryId_ids.iteritems():
            sortedRanks = sorted([system_id_rank[system][id_] for id_ in ids])            
            try:
                result, count = methodFunction(sortedRanks, cutoff, totalRanks)
#             except TypeError:
#                 print('Warning. Type error when computing: method=%s, system=%s' %(methodName, system))
#                 result, count = np.NaN, np.NaN
            except ValueError:
                print('Warning. Value error when computing: method=%s, system=%s' %(methodName, system))
                result, count = np.NaN, np.NaN
            system_query_result[system][queryId] = result
            query_performanceSum[queryId] += result
            query_relSum[queryId] += 100 * (count / len(sortedRanks))
            query_performanceCount[queryId] += 1
    
    avgRelRate = 0.0
    system_results = defaultdict(lambda : list())
    for query in queryId_ids:
        query_performanceSum[query] /= (query_performanceCount[query] / fuzziness)
        avgRelRate += query_relSum[query]
    avgRelRate /= (len(queryId_ids) * len(systemNames))
    for system in systemNames:
        for queryId in queryId_ids:
            system_results[system].append(system_query_result[system][queryId])
            
    
    queryIds = queryId_ids.keys()
    systemMatrix_GrEqLe = defaultdict(lambda : defaultdict(lambda : [0,0,0]))
    
    for i in range(len(systemNames)):
        for j in range(i+1, len(systemNames)):#range(len(systemNames)):#
            for query in queryIds:
                result_i = system_query_result[systemNames[i]][query]
                result_j = system_query_result[systemNames[j]][query]
                meanPerformance = query_performanceSum[query]
                if isNegativeMethod(canonicalMethodName):
                    if result_i < result_j - meanPerformance:
                        systemMatrix_GrEqLe[systemNames[i]][systemNames[j]][0] += 1
                    elif result_i > result_j + meanPerformance:
                        systemMatrix_GrEqLe[systemNames[i]][systemNames[j]][2] += 1
                    else:
                        systemMatrix_GrEqLe[systemNames[i]][systemNames[j]][1] += 1
                else:
                    if result_i > result_j + meanPerformance:
                        systemMatrix_GrEqLe[systemNames[i]][systemNames[j]][0] += 1
                    elif result_i < result_j - meanPerformance:
                        systemMatrix_GrEqLe[systemNames[i]][systemNames[j]][2] += 1
                    else:
                        systemMatrix_GrEqLe[systemNames[i]][systemNames[j]][1] += 1
                 
    countSigsPair = 0
    countAllPair = 0
    significance_level = fuzziness
    nominatorSumAbs = 0
    denominator = 0
    tieCountAbs = 0
    for system_i in range(len(systemNames)):
        for system_j in range(system_i + 1, len(systemNames)):
            with warnings.catch_warnings(record=True) as w:
                statistic = stats.ttest_rel(system_results[systemNames[system_i]], system_results[systemNames[system_j]])
                if len(w) > 0:
                    print('ttest', methodName, systemNames[system_i], systemNames[system_j], statistic, w)
                elif statistic[1] <= significance_level:
                    countSigsPair += 1
#             with warnings.catch_warnings(record=True) as w:
#                 statistic = stats.wilcoxon(system_results[systemNames[system_i]], system_results[systemNames[system_j]])
            countAllPair += 1
            absCounts = systemMatrix_GrEqLe[systemNames[system_i]][systemNames[system_j]]
            nominatorSumAbs += min([absCounts[0], absCounts[2]])
            tieCountAbs += absCounts[1]
            denominator += sum(absCounts)
      
    errorRate = 100 * nominatorSumAbs / denominator
    tieRate = 100 * tieCountAbs / denominator
    percSignSign = 100 * countSigsPair / countAllPair
        
    return (methodName, canonicalMethodName, cutoff, fuzziness, avgRelRate, errorRate, tieRate, percSignSign)