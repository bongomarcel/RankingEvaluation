'''
Created on 16 Jun 2018

@author: Marcel Dunaiski
'''
from __future__ import division
from _collections import defaultdict

from EvaluationMeasures import isNegativeMethod
from multiprocessing import Pool
from HelperTools import StreamReader, Experiment, Results
from SwapRate import swapRate
from ErrorTieRate import errorTieRates
from Pipeline import Pipeline
import EvaluationMeasures as ms
import numpy as np
from scipy import stats
import time, sys, argparse
import random as rand
import warnings

cur_time = lambda : int(round(time.time()))

################################################################
# Add evaluation methods here that should be evaluated.
# Custom evaluation methods can be added to EvaluationMeasures.py
# The following conventios is required:
#    {name : (function, cutoff, requiresTotalRanks)}    
# totalRanks should be True for methods that require total ranks number (which is set with -n).
################################################################
def getEvaluationMethods():
    return {
        'Average' : (ms.Average, None, False),
        'Average@10' : (ms.AP_at_n, 10, False),
        'Average@20' : (ms.Average_at_k, 20, False),
        'Average@Avg' : (ms.Average_at_avg, None, False),
        'Median' : (ms.Median, None, False),
        'AP' : (ms.AP, None, False),
        'nDCG@Avg' : (ms.NDCG_at_avg, None, False), 
        'AP_inter_per' : (ms.AP_interpolated_buckets, None, True),
        'P@2000_per' : (ms.P_at_k_buckets, 2000 , True)
    }

################################################################
# Example of how an evaluation method with multiple cutoff can 
# be specified programmatically.
################################################################
def getMehodOverRange():
    methods = dict()
    for i in range(10,1001,10):
        methods['P@' + '%5d' %(i)] = (ms.P_at_k, i, False)
        methods['Recall@' + '%5d' %(i)] = (ms.Recall_at_k, i, False)
    return methods

def computeSwapRate(experiment):
    fuzziness = experiment.evaluationProperties[3]
    bootstrapSize = experiment.evaluationProperties[4]
    experiment.correlation = computeCorrelation(experiment.system_id_rank)
    experiment.swapResults = swapRate(experiment.metricProperties, experiment.system_id_rank, experiment.queryId_ids, fuzziness, bootstrapSize)
    return experiment

def computeErrorTieRate(experiment):
    fuzziness = experiment.evaluationProperties[3]
    experiment.correlation = computeCorrelation(experiment.system_id_rank)
    experiment.errTieResults = errorTieRates(experiment.metricProperties, experiment.system_id_rank, experiment.queryId_ids, fuzziness)
    return experiment

def computeCorrelation(system_id_rank):
    correlations = list()
    systemList = system_id_rank.keys()
    ids = None
    for system_i in range(1, len(systemList)):
        if ids is None:
            ids = [id_ for id_ in system_id_rank[systemList[system_i]]]
        ranks_i = [system_id_rank[systemList[system_i]][id_] for id_ in ids]
        for system_j in range(len(systemList)):
            if system_i < system_j:
                ranks_j = [system_id_rank[systemList[system_j]][id_] for id_ in ids]
                correlations.append(stats.spearmanr(list(ranks_i), list(ranks_j))[0])
    return (np.average(correlations), np.std(correlations))

def computeActualDecisionRates(system_id_rank, totalRanks, metricProperties, estValue):
    #metricProperties = (methodNameId, norm_name, methodFunction, cutoff, requires_totalRanks)
    methodId = metricProperties[0]
    methodFunction = metricProperties[2]
    cutoff = metricProperties[3]
    requiresTotalRanks = metricProperties[4]
    system_result = dict()
    for system in system_id_rank:
        sortedRanks = sorted(system_id_rank[system].values())
        try:
            with warnings.catch_warnings(record=True) as err:
                result, count = methodFunction(sortedRanks, cutoff, totalRanks)
            if len(err) > 0:
                print('Warning: Invalid result: method=%s, result=%s, count=%s, total_ranks=%s, k=%s, requiresTotalRanks=%s' %(methodId, result, count, totalRanks, str(cutoff), requiresTotalRanks))
                result = np.NaN
#         except TypeError:
#             print('Warning. Type error when computing: method=%s, system=%s' %(methodId, system))
#             result, count = np.NaN, np.NaN
        except ValueError:
            print('Warning. Value error when computing: method=%s, system=%s' %(methodId, system))
            result, count = np.NaN, np.NaN
        system_result[system] = result
    
    smallestDiff = 10000000000
    decisionCount = 0
    systemList = system_id_rank.keys()
    for i in range(len(systemList)):
        system_i = systemList[i]
        for j in range(i + 1, len(systemList)):
            system_j = systemList[j]
            diff = abs(system_result[system_i] - system_result[system_j])
            if diff > estValue and estValue > 0:
                decisionCount += 1
            if diff > 0 and diff < smallestDiff:
                smallestDiff = diff
    decisionRate = 100 * decisionCount / (len(systemList) * (len(systemList) - 1)/2)
    return (methodId, cutoff, decisionRate, system_result)
    
    
def getRanks(inputFile, useColumnHeadersAsSystemIds=True, returnColumnIDs=False):
    system_id_rank = dict()
    column_name = dict()
    list_of_columnNames = list()
    row = 0
    ids = set()
    for parts in StreamReader(inputFile):
        columnIndex = 0
        if row == 0:
            for name in parts[1:]:
                columnIndex += 1
                column_name[columnIndex] = name
                list_of_columnNames.append(name)
        else:
            id_ = None
            for element in parts:
                if columnIndex == 0:
                    id_ = element
                    if id_ in ids:
                        print("WARNING: Non unique ID found in row %d: '%s'" %(row + 1, id_))
                    ids.add(id_)
                else:
                    columnId = columnIndex
                    if useColumnHeadersAsSystemIds:
                        columnId = column_name[columnIndex]
                    if not system_id_rank.has_key(columnId):
                        system_id_rank[columnId] = dict()
                    rank = float(element)
                    if rank <= 0:
                        print("WARNING: non-positive rank '%f' found for system: '%s'" %(rank, column_name[columnIndex]))
                    system_id_rank[columnId][id_] = rank
                columnIndex += 1
        row += 1
    if returnColumnIDs:
        return (list_of_columnNames, system_id_rank)
    return system_id_rank


def getExperimentsFromParameterRanges(methodNames_parameters, splitCounts, systemCounts, entityCounts, significanceValues, cutoffLevels, system_id_rank, totalRanks, bootstrapSize=1000):
    experiments = list()
    for splitCount in splitCounts:
        for systemCount in systemCounts:
            if entityCounts is None:
                entityCounts = [len(system_id_rank[system_id_rank.keys()[0]].keys())]
            for entityCount in entityCounts:
                idsList = system_id_rank[system_id_rank.keys()[0]].keys()
                rand.shuffle(idsList)
                idsList = idsList[:entityCount]
                #THIS IS WITHOUT REPLACEMENT
                q, r = divmod(len(idsList), splitCount)
                indices = [q*i + min(i, r) for i in xrange(splitCount+1)]
                idsForQueries = [idsList[indices[i]:indices[i+1]] for i in xrange(splitCount)]
                queryId_ids = dict()
                for i, ids in enumerate(idsForQueries):
                    queryId_ids['%d' %(i)] = ids
                systemIdsList = system_id_rank.keys()
                sampledSystemNames = rand.sample(systemIdsList, systemCount)
                sampled_system_id_rank = defaultdict()
                for sampledSystemName in sampledSystemNames:
                    if not sampled_system_id_rank.has_key(sampledSystemName):
                        sampled_system_id_rank[sampledSystemName] = dict()
                    for id_ in system_id_rank[sampledSystemName]:
                        sampled_system_id_rank[sampledSystemName][id_] = system_id_rank[sampledSystemName][id_]                         
                for fuzziness in significanceValues:
                    for methodId in methodNames_parameters.keys():
                        methodFunction, hasCutoff, requires_totalRanks = methodNames_parameters[methodId] #(function, cutoff, requiresTotal)
                        if requires_totalRanks:
                            curTotalRanks = totalRanks
                        else:
                            curTotalRanks = None
                        if hasCutoff is True: #requires specifying cutoff
                            for cutoff in cutoffLevels:
                                uniqueMethodId = methodId + '_' + str(cutoff)
                                measureProperties = (uniqueMethodId, methodId, methodFunction, cutoff, curTotalRanks)
                                evaluationProperties = (splitCount, systemCount, entityCount, fuzziness, bootstrapSize)
                                ex = Experiment(measureProperties, sampled_system_id_rank, queryId_ids, evaluationProperties)
                                experiments.append(ex)
                        else: #cutoff is either a value or None
                            uniqueMethodId = methodId
                            measureProperties = (uniqueMethodId, methodId, methodFunction, hasCutoff, curTotalRanks)
                            evaluationProperties = (splitCount, systemCount, entityCount, fuzziness, bootstrapSize)
                            ex = Experiment(measureProperties, sampled_system_id_rank, queryId_ids, evaluationProperties)
                            experiments.append(ex)
    return experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Evaluation Tool.', usage='%(prog)s [options] [file containing ranks]')
    parser.add_argument('-s', '--seed', default=None, type=int, help='seed (default: None)') #seed
    parser.add_argument('-t', '--threads', default=1, type=int, help='number of threads to use. (default: 1)') # nr threads
    parser.add_argument('-i', '--iter', default=50, type=int, help='number of iterations over which results are averaged. (default: 50)') # iterations
    parser.add_argument('-b', '--boot', default=1000, type=int, help='number of bootstraps to use. (default: 1000)') # bootstrap size
    
    parser.add_argument('--cutoffs', nargs='*', default=None, type=int, help='list of cut-off ranks to use. (default: None)') # cut-off level    
    parser.add_argument('--splits', nargs='*', default=[10], type=int, help='list of splits') #splits
    parser.add_argument('--systems', nargs='*', default=None, type=int, help='list of use only a subset of systems. (default: All systems are used)')# system counts
    parser.add_argument('--entities', nargs='*', default=None, type=int, help='use only a subset of entities. (default: All entities are used)')# entity count
    parser.add_argument('--sigs', nargs='*', default=[0.005, 0.01, 0.05, 0.1, 0.15], type=float, help='specify significance thresholds (default: [0.005, 0.01, 0.05, 0.1, 0.15])')# significance values
    
    parser.add_argument('-w', '--swap', action='store_true', help='use swap method to compute significance values. (default: bootstrap method)')# swap method
    parser.add_argument('-e', '--err', action='store_true', help='compute error and tie rates. (not default)')
    parser.add_argument('-p', '--printing', default='all', choices=['all', 'sign', 'eval', 'actual'], help='print all results, significance results only, or method evaluation results only. (default: all)')
    parser.add_argument('-n', '--totalranks', default=None, type=int, help='total ranks required to compute permille/percentile ranks (default: None)') # total ranks
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbosely') # total ranks
    
    parser.add_argument('-f', '--file', nargs='?', type=argparse.FileType('r'), default=sys.stdin) # input file ranks
    parser.add_argument('-o', '--out', nargs='?', type=argparse.FileType('w'), default=sys.stdout) # output file
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    seed_ = args.seed
    totalRanks = args.totalranks
    
    nr_threads = args.threads
    iterations = args.iter
    bootstrapSize = args.boot
    
    useSwap = args.swap
    printOption = args.printing
    useEstDiff = useSwap == False    # use bootstrap method and not swap method
    useErrTie = args.err
    #computeEval = args.printing == 'all' or args.printing == 'eval'
    computeActualDecisionRate = args.printing == 'all' or args.printing == 'actual'
    
    list_of_cutoffs = args.cutoffs
    list_of_splits = args.splits
    list_of_entityCounts = args.entities
    list_of_signValues = args.sigs
    list_of_systemCounts = args.systems
        
    inputFile = args.file
    outputFile = args.out
    verbose = args.verbose
    
     
    systemList, system_id_rank = getRanks(inputFile, True, True)
    if len(systemList) < 2:
        print("Error: only %d systems found in input file '%s'." %(len(systemList), inputFile.name))
        sys.exit()
    
    if list_of_systemCounts is None:    
        list_of_systemCounts = [len(system_id_rank.keys())]
    if list_of_systemCounts is None:
        list_of_systemCounts = [len(systemList)]
    
    print('Computation overview:')
    print('Seed:\t%s' %(seed_))
    print('Nr. of threads:\t%d' %(nr_threads))
    print('Nr. of iterations:\t%d' %(iterations))
    print('Bootstrap size:\t%d' %(bootstrapSize))
    print('Significance method used:\t%s' %('Est. Diff. Method' if useEstDiff else 'Swap Method (bootstrapped)'))
    print('Compute actual decision rate:\t%s' %(computeActualDecisionRate))
    print('Print option: %s' %(printOption))
    print('List of cut-off levels:\t%s' %(list_of_cutoffs))
    print('List of splits:\t%s' %(list_of_splits))
    print('List of system counts:\t%s' %(list_of_systemCounts))
    print('List of entity counts:\t%s' %('using all' if list_of_entityCounts is None else list_of_entityCounts))
    print('List of significance values:\t%s' %(list_of_signValues))
    print('Verbose:\t%s' %(verbose))
    print('File input:\t%s' %(inputFile.name))
    print('File output:\t%s' %(outputFile.name))
    methodNames_parameters = getEvaluationMethods()
    minQuerySize = len(system_id_rank[systemList[0]]) / max(list_of_splits)
    print('Minimum query size: %.2f' %(minQuerySize))
    if minQuerySize < 5:
        print('WARNING: results will be very unreliable with small query sizes.')
    print('\n')
    experiment_results = Results()
    experiment_results.systemList = systemList
    pool = Pool(nr_threads)
        
    big_bang_time = cur_time()
    if useSwap or useEstDiff:
        print('Computing ASL rate and swap method (bootstrapped approach)') #computes: ASL rate, swap method (with replacement), bootstrapped approach.
        begin_time = cur_time()
        rand.seed(seed_)
        for iter_ in range(iterations):
            print('Computing iteration %d of %d, time=%f.' %(iter_ + 1, iterations, (cur_time() - begin_time) / 60))
            data = getExperimentsFromParameterRanges(methodNames_parameters, list_of_splits, list_of_systemCounts, list_of_entityCounts, list_of_signValues, list_of_cutoffs, system_id_rank, totalRanks, bootstrapSize)
            pipe = Pipeline([computeSwapRate], nr_threads=nr_threads, update_interval=100, verbose=verbose)
            pipe.execute(data, f_result_handler=experiment_results.handleExperimentResult, chunksize=10, pool=pool)
        print('Done computing ASL rate and swap method. Time: %f\n' %((cur_time() - begin_time) / 60))
        
    if useErrTie:
        print('Computing Error and Tie rate') #computes: error and tie rate, relRate, signRate (without replacement)
        begin_time = cur_time()
        rand.seed(seed_)
        for iter_ in range(iterations):
            print('Computing iteration %d of %d, time=%f.' %(iter_ + 1, iterations, (cur_time() - begin_time) / 60))
            data = getExperimentsFromParameterRanges(methodNames_parameters, list_of_splits, list_of_systemCounts, list_of_entityCounts, list_of_signValues, list_of_cutoffs, system_id_rank, totalRanks, bootstrapSize)
            pipe = Pipeline([computeErrorTieRate], nr_threads=nr_threads, update_interval=100, verbose=verbose)
            pipe.execute(data, f_result_handler=experiment_results.handleExperimentResult, chunksize=10, pool=pool)
        print('Done computing Error and Tie rate. Time: %f\n' %((cur_time() - begin_time) / 60 ))
    
    if computeActualDecisionRate:
        print('Computing actual decision rates and system performances.')
        data = list()
        evalResultsData = None
        if useEstDiff:
            evalResultsData = experiment_results.splitCount_system_entity_fuzziness_cutoff_method_estDiff
        elif useSwap:
            evalResultsData = experiment_results.splitCount_system_entity_fuzziness_cutoff_method_estValueMin
        for splitCount, systemCount_other in evalResultsData.iteritems():
            for systemCount, entityCount_other in systemCount_other.iteritems():
                for entityCount, fuzziness_other in entityCount_other.iteritems():
                    for fuzziness, cutoff_measure_estValue in fuzziness_other.iteritems():
                        for norm_name in methodNames_parameters.keys():
                            properties = methodNames_parameters[norm_name]
                            requires_totalRanks = False
                            if len(properties) == 2:
                                methodFunction, hasCutoff = properties
                            else:
                                methodFunction, hasCutoff, requires_totalRanks = properties
                            if hasCutoff is True: #requires specifying cutoff
                                for cutoff in cutoff_measure_estValue:
                                    methodNameId = norm_name + '_' + str(cutoff)
                                    metricProperties = (methodNameId, norm_name, methodFunction, cutoff, requires_totalRanks)
                                    estValue = np.average(cutoff_measure_estValue[cutoff][methodNameId])
                                    data.append((system_id_rank, totalRanks, metricProperties, estValue))
                                    
                            else: #cutoff is either a value or None
                                methodNameId = norm_name
                                metricProperties = (methodNameId, norm_name, methodFunction, hasCutoff, requires_totalRanks)
                                estValue = np.average(cutoff_measure_estValue[hasCutoff][methodNameId])
                                data.append((system_id_rank, totalRanks, metricProperties, estValue))
        pipe = Pipeline([computeActualDecisionRates], nr_threads=nr_threads, update_interval=100, verbose=verbose)
        results = pipe.execute(data, f_result_handler=None, chunksize=10, pool=pool)
        for r in results:
            method, cutoff, decisionRate, system_result = r
            experiment_results.method_cutoff_decisionRate[method][cutoff] = decisionRate
            experiment_results.method_cutoff_system_result[method][cutoff] = system_result
            for system, result in system_result.iteritems():
                experiment_results.system_method_actualResult[system][method] = result
        print('Done computing actual decision rates and system performances. Time: %f\n' %((cur_time() - begin_time) / 60))
        
    sys = len(system_id_rank.keys())
    ent = len(system_id_rank[system_id_rank.keys()[0]].keys())
    #GET BEST PARAMETERS
    c = list_of_cutoffs[0] if list_of_cutoffs is not None else None
    sp = list_of_splits[0]
    
    with warnings.catch_warnings(record=True) as w:
        outputFile.write('Results:\n')
        if printOption == 'all' or printOption == 'eval':
            experiment_results.writeEvalToFile(outputFile)
            
        if printOption == 'all' or printOption == 'actual':
            experiment_results.writeActualToFile(outputFile)
        
        if printOption == 'all' or printOption == 'sign':
            method_f_estMinDiff = defaultdict(lambda : defaultdict(lambda : 0))
            method_f_estDiff = defaultdict(lambda : defaultdict(lambda : 0))
            for f in list_of_signValues:
                for method, system_result in experiment_results.method_cutoff_system_result.iteritems():
                    method_f_estMinDiff[method][f] = np.average(experiment_results.splitCount_system_entity_fuzziness_cutoff_method_estValueMin[sp][sys][ent][f][c][method])
                    method_f_estDiff[method][f] = np.average(experiment_results.splitCount_system_entity_fuzziness_cutoff_method_estDiff[sp][sys][ent][f][c][method])
                            
            signStrings = ['***', '**', '*', '.', ',', '_', '_1', '_2', '_3', '_4', '_5']
            fValue_string = {v : s for v, s in zip(list_of_signValues, signStrings)}
            for methodName in methodNames_parameters.keys():
                outputFile.write(methodName + "\tSignificance: 0 %s ' ' 1\n" %(''.join([" '" + s + "' " + str(v) for v, s in zip(list_of_signValues, signStrings)])))
                outputFile.write('%s\t%s\n' %('Systems', '\t'.join(systemList)))
                isNegative = isNegativeMethod(methodName)
                for system_i in systemList:
                    str_values = list()
                    for system_j in systemList:
                        if system_i == system_j:
                            str_values.append('--')
                        else:
                            valueFound = False
                            for fValue in list_of_signValues:
                                score_i = experiment_results.system_method_actualResult[system_i][methodName]
                                score_j = experiment_results.system_method_actualResult[system_j][methodName]
                                if useSwap:
                                    estDiff = method_f_estDiff[methodName][fValue]
                                else:
                                    estDiff = method_f_estMinDiff[methodName][fValue]
                                if estDiff > 0:
                                    if isNegative and score_j - score_i > estDiff:
                                        str_values.append(fValue_string[fValue])
                                        valueFound = True
                                        break
                                    elif not isNegative and score_i - score_j > estDiff:
                                        str_values.append(fValue_string[fValue])
                                        valueFound = True
                                        break
                            if not valueFound:
                                str_values.append(' ')
                    outputFile.write('%s\t%s\n' %(system_i, '\t'.join(str_values)))
                outputFile.write('\n')
    if len(w) > 0:
        print('Warning. Some invalid result observed in the results.')
    print('DONE! Total time: %s\n' %((cur_time() - big_bang_time) / 60))
    outputFile.close()
    
    