'''
Created on 04 Jul 2016

@author: marcel
'''
from __future__ import division
from _collections import defaultdict
import numpy as np

class StreamReader(object):
    
    def __init__(self, f, sep='\t'):
        self.file = f
        self.sep_ = sep
        
    def __iter__(self):
        for line in self.file:
            line = line.decode('utf-8')
            if self.sep_ is None:
                yield line.rstrip('\r\n')
            else:
                parts = line.rstrip('\r\n').split(self.sep_)
                yield parts

class Experiment(object):
    
    evaluationProperties = ()     # (computeSwap, computeErrTie, splitCount, systemCount, entityCount, fuzziness, bootstrapSize)
    metricProperties = ()         # (metricNameId, canonicalMethodName, methodFunction, cutoff, totalranks)
    system_id_rank = []           # {system : {entityId : rank}}
    queryId_ids = []              # {queryId : [entityId]}
    errTieResults = None          # (metricNameId, canonicalMethodName, cutoff, f, relRate, errRate, tieRate, signRate)
    swapResults = None            # (metricNameId, canonicalMethodName, cutoff, f, bootSize, asl, estDiff, binValue, swapDiffMin, swapDiffMax)
    correlation = 0
    
    def __init__(self, metricProperties, system_id_rank, queryId_ids, evaluationProperties):
        self.metricProperties = metricProperties
        self.evaluationProperties = evaluationProperties
        self.system_id_rank = system_id_rank
        self.queryId_ids = queryId_ids
        
    def __str__(self):
        str_ = 'evaluationProperties %s\n' %(str(self.evaluationProperties))
        str_ += 'metricProperties %s\n' %(str(self.metricProperties))
        str_ += 'system_id_rank size=%s\n' %(len(self.system_id_rank))
        str_ += 'queryId_ids size=%s\n' %(len(self.queryId_ids))
        str_ += 'errTieResults %s\n' %(str(self.errTieResults))
        str_ += 'queryId_ids %s\n' %(str(self.swapResults))
        str_ += 'correlation %s\n' %(str(self.correlation))
        return str_

class Results(object):
    splitCount_system_entity_fuzziness_cutoff_method_relRates = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_errorRates = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_tieRates = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_signRates = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    
    splitCount_system_entity_fuzziness_cutoff_method_asl = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_binValue = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_estDiff = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_estValueMin = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    splitCount_system_entity_fuzziness_cutoff_method_estValueMax = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: list()))))))
    
    method_cutoff_decisionRate = defaultdict(lambda : defaultdict(lambda: list()))
    method_cutoff_system_result = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda: None)))
    
    splitCount_system_entity_correlation = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda: 0)))
    methodName_normName = dict()
    system_method_actualResult = defaultdict(lambda: defaultdict(lambda: np.NaN))
    systemList = None
    
    def writeEvalToFile(self, f_out):
        f_out.write('method\tnorm\tcorrelation\tstddev corr\tsystemCount\tsplitCount\tentityCount\tfuzziness\tcutoff\terrRate\tstddev errRate\ttieRate\tstddev tieRate\ttRate\trelRate\tasl\testDiff\tswapDiffMin\tswapDiffMax\tdecisions\n')
        for splitCount, system_other in self.splitCount_system_entity_fuzziness_cutoff_method_asl.iteritems():
            for system, entity_other in system_other.iteritems():
                for entityCount, fuzziness_other in entity_other.iteritems():
                    correlation, correlation_std = self.splitCount_system_entity_correlation[splitCount][system][entityCount]
                    for fuzziness, cutoff_other in fuzziness_other.iteritems():
                        for cutoff, method_other in cutoff_other.iteritems():
                            for method, asl in method_other.iteritems():
                                relRates = self.splitCount_system_entity_fuzziness_cutoff_method_relRates[splitCount][system][entityCount][fuzziness][cutoff][method]
                                errRates = self.splitCount_system_entity_fuzziness_cutoff_method_errorRates[splitCount][system][entityCount][fuzziness][cutoff][method]
                                tieRates = self.splitCount_system_entity_fuzziness_cutoff_method_tieRates[splitCount][system][entityCount][fuzziness][cutoff][method]
                                signRates = self.splitCount_system_entity_fuzziness_cutoff_method_signRates[splitCount][system][entityCount][fuzziness][cutoff][method]
                                estDiff = self.splitCount_system_entity_fuzziness_cutoff_method_estDiff[splitCount][system][entityCount][fuzziness][cutoff][method]
                                #binValue = self.splitCount_system_entity_fuzziness_cutoff_method_binValue[splitCount][system][entityCount][fuzziness][cutoff][method]
                                estDiffMin = self.splitCount_system_entity_fuzziness_cutoff_method_estValueMin[splitCount][system][entityCount][fuzziness][cutoff][method]
                                estDiffMax = self.splitCount_system_entity_fuzziness_cutoff_method_estValueMax[splitCount][system][entityCount][fuzziness][cutoff][method]
                                decisions = self.method_cutoff_decisionRate[method][cutoff]
                                f_out.write('%s\t%s\t%s\t%.5f\t%.5f\t%d\t%d\t%.3f\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' %(method, self.methodName_normName[method], correlation, correlation_std, system, splitCount, entityCount, fuzziness, cutoff, np.average(errRates), np.std(errRates), np.average(tieRates), np.std(tieRates), np.average(signRates), np.average(relRates), np.average(asl), np.average(estDiff), np.average(estDiffMin), np.average(estDiffMax), np.average(decisions)))
        f_out.write('\n')
        
    def writeActualToFile(self, f_out):
        methodList = self.system_method_actualResult[self.systemList[0]].keys()
        f_out.write('Measure\t%s\n' %('\t'.join(self.systemList)))
        for method in methodList:
            f_out.write('%s\t%s\n' %(method, '\t'.join([str(self.system_method_actualResult[system][method]) for system in self.systemList])))
        f_out.write('\n')
        
        
        
    def handleExperimentResult(self, experiment):
        #evaluationProperties = (splitCount, systemCount, entityCount, fuzziness, bootstrapSize)
        metricProperties = experiment.metricProperties
        methodName = metricProperties[0]
        canonicalMethodName = metricProperties[1]
        self.methodName_normName[methodName] = canonicalMethodName
        splitCount = experiment.evaluationProperties[0]
        systemCount = experiment.evaluationProperties[1]
        entityCount = experiment.evaluationProperties[2]
        if experiment.swapResults is not None:
            result = experiment.swapResults
            cutoff = result[2]
            f = result[3]
            #boot = result[4]
            asl =  result[5]
            estDiff = result[6]
            binValue = result[7]
            estDiffMin = result[8]
            estDiffMax = result[9]
            self.splitCount_system_entity_correlation[splitCount][systemCount][entityCount] = experiment.correlation
            self.splitCount_system_entity_fuzziness_cutoff_method_asl[splitCount][systemCount][entityCount][f][cutoff][methodName].append(asl)
            self.splitCount_system_entity_fuzziness_cutoff_method_estDiff[splitCount][systemCount][entityCount][f][cutoff][methodName].append(estDiff)
            self.splitCount_system_entity_fuzziness_cutoff_method_binValue[splitCount][systemCount][entityCount][f][cutoff][methodName].append(binValue)
            self.splitCount_system_entity_fuzziness_cutoff_method_estValueMin[splitCount][systemCount][entityCount][f][cutoff][methodName].append(estDiffMin)
            self.splitCount_system_entity_fuzziness_cutoff_method_estValueMax[splitCount][systemCount][entityCount][f][cutoff][methodName].append(estDiffMax)
        if experiment.errTieResults is not None:
            result = experiment.errTieResults
            cutoff = result[2]
            f = result[3]
            relRate = result[4]
            errRate = result[5]
            tieRate = result[6]
            signRate = result[7]
            self.splitCount_system_entity_correlation[splitCount][systemCount][entityCount] = experiment.correlation
            self.splitCount_system_entity_fuzziness_cutoff_method_relRates[splitCount][systemCount][entityCount][f][cutoff][methodName].append(relRate)
            self.splitCount_system_entity_fuzziness_cutoff_method_errorRates[splitCount][systemCount][entityCount][f][cutoff][methodName].append(errRate)
            self.splitCount_system_entity_fuzziness_cutoff_method_tieRates[splitCount][systemCount][entityCount][f][cutoff][methodName].append(tieRate)
            self.splitCount_system_entity_fuzziness_cutoff_method_signRates[splitCount][systemCount][entityCount][f][cutoff][methodName].append(signRate)
        
        