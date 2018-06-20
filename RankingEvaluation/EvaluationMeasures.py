'''
Created on 04 Mar 2018

@author: Marcel Dunaiski
'''
from __future__ import division
from _collections import defaultdict
import math
import numpy as np

bucket_count = 1000
intervals = list(np.arange(100, 10001, 100))

def isNegativeMethod(canonicalName):
    negativeMethods = ['average', 'median', 'min', 'max', 'average@k', 'median@k']
    if canonicalName.lower() in negativeMethods:
        return True
    if canonicalName.lower().startswith('average@') or canonicalName.lower().startswith('median@'):
        return True
    return False

def Average(sortedRank, cutoff=None, totalRanks=None):
    return (sum(sortedRank) / len(sortedRank), len(sortedRank))

def Average_at_avg(sortedRanks, cutoff=None, totalRanks=None):
    return Average_at_k(sortedRanks, np.floor(np.average(sortedRanks)))

def Average_at_testsize(sortedRanks, cutoff=None, totalRanks=None):
    return Average_at_k(sortedRanks, len(sortedRanks))

def Average_at_k(sortedRanks, k, totalRanks=None):
    count = 0
    sum_ = 0
    for rank in sortedRanks:
        if rank <= k:
            count += 1
            sum_ += rank
        else:
            break
    if count == 0:
        return (np.NaN, 0)
    return (sum_ / count, count)

def Average_at_recall(sortedRanks, recallLevel, totalRanks=None):
    recall_count = int(math.ceil(len(sortedRanks) * recallLevel))
    if recall_count < 1:
        return (0, 0)
    return Average_at_k(sortedRanks, sortedRanks[recall_count - 1])

def Median(sortedRanks, cutoff=None, totalRanks=None):
    return (np.median(np.array(sortedRanks)), len(sortedRanks))

def Median_at_k(sortedRanks, k, totalRanks=None):
    tmp_list = list()
    for rank in sortedRanks:
        if rank <= k:
            tmp_list.append(rank)
        else:
            break
    if len(tmp_list) < 1:
        return (np.NaN, 0)
    return (np.median(tmp_list), len(tmp_list))

def Median_at_recall(sortedRanks, recallLevel, totalRanks=None):
    recall_count = int(math.ceil(len(sortedRanks) * recallLevel))
    if recall_count < 1:
        return (np.NaN, 0)
    return Median_at_k(sortedRanks, sortedRanks[recall_count - 1])

def Median_at_avg(sortedRanks, cutoff=None, totalRanks=None):
    return Median_at_k(sortedRanks, np.floor(np.average(sortedRanks)))

def Median_at_testsize(sortedRanks, cutoff=None, totalRanks=None):
    return Median_at_k(sortedRanks, len(sortedRanks))

def Min(sortedRanks, cutoff=None, totalRanks=None):
    return (sortedRanks[0], 1)

def Max(sortedRanks, cutoff=None, totalRanks=None):
    return (sortedRanks[-1], 1)

def NDCG_10(sortedRanks, cutoff=None, totalRanks=None):
    return NDCG_at_n(sortedRanks, 10)

def NDCG_at_testsize(sortedRanks, cutoff=None, totalRanks=None):
    return NDCG_at_n(sortedRanks, len(sortedRanks))

def ROC_AUC_std_intervals(sortedRanks, cutoff=None, totalRanks=None):
    return ROC_AUC_intervals(sortedRanks, cutoff, totalRanks)

def ROC_at_10000_auc(sortedRanks, cutoff=None, totalRanks=None):
    return ROC_AUC(sortedRanks, 10000)

def ROC_at_avg_auc(sortedRanks, cutoff=None, totalRanks=None):
    return ROC_at_k_auc(sortedRanks, np.floor(np.average(sortedRanks)))

def ROC_at_k_auc(sortedRanks, k, totalRanks=None):
    return ROC_AUC(sortedRanks, k)

def P_at_recall(sortedRanks, recallLevel, totalRanks=None):
    recall_count = int(math.ceil(len(sortedRanks) * recallLevel))
    if recall_count < 1:
        return (0, 0)
    value = recall_count / sortedRanks[recall_count - 1]
    return (value if value < 1 else 1, recall_count)

def P_at_50recall(sortedRanks, cutoff=None, totalRanks=None):
    return P_at_recall(sortedRanks, 0.5)

def P_at_avg(sortedRanks, cutoff=None, totalRanks=None):
    return P_at_k(sortedRanks, np.floor(np.average(sortedRanks)))

def P_at_testsize(sortedRanks, cutoff=None, totalRanks=None):
    return P_at_k(sortedRanks, len(sortedRanks))

def P_at_k(sortedRanks, k, totalRanks=None):
    count = 0
    for rank in sortedRanks:
        if rank >= k:
            count += 1
            value = count / k #rank
            return (value if value < 1 else 1, count)
        count += 1
    value = count / k #sortedRanks[-1]
    return (value if value < 1 else 1, count)

def P_at_avg_buckets(sortedRanks, cutoff, total_ranks):
    return P_at_k_buckets(sortedRanks, np.floor(np.average(sortedRanks)) * (total_ranks / bucket_count), total_ranks)

def P_at_50recall_buckets(sortedRanks, cutoff, total_ranks):
    if len(sortedRanks) % 2 == 0:
        n = (sortedRanks[int(len(sortedRanks)/2) - 1]+sortedRanks[int(len(sortedRanks)/2)]) / 2
    else:
        n = sortedRanks[int(len(sortedRanks)/2)]
    n = n * (total_ranks / bucket_count)
    return P_at_k_buckets(sortedRanks, n, total_ranks)

def AP_at_10(sortedRanks, cutoff=None, totalRanks=None):
    return AP_at_n(sortedRanks, 10)

def AP_at_avg(sortedRanks, cutoff=None, totalRanks=None):
    return AP_at_n(sortedRanks, np.floor(np.average(sortedRanks)))

def AP_at_avg_buckets(sortedRanks, cutoff, total_ranks):
    return AP_at_n_buckets(sortedRanks, np.floor(np.average(sortedRanks)) * (total_ranks / bucket_count), total_ranks)

def AP_at_recall(sortedRanks, recallLevel, totalRanks=None):
    recall_count = int(math.ceil(len(sortedRanks) * recallLevel))
    if recall_count < 1:
        return (0, 0)
    return AP_at_n(sortedRanks, sortedRanks[recall_count - 1])

def AP_at_50recall(sortedRanks, cutoff=None, totalRanks=None):
    if len(sortedRanks) % 2 == 0:
        n = (sortedRanks[int(len(sortedRanks)/2) - 1]+sortedRanks[int(len(sortedRanks)/2)]) / 2
    else:
        n = sortedRanks[int(len(sortedRanks)/2)]
    return AP_at_n(sortedRanks, int(n))

def AP_at_50recall_buckets(sortedRanks, cutoff, total_ranks):
    if len(sortedRanks) % 2 == 0:
        n = (sortedRanks[int(len(sortedRanks)/2) - 1]+sortedRanks[int(len(sortedRanks)/2)]) / 2
    else:
        n = sortedRanks[int(len(sortedRanks)/2)]
    n = n * (total_ranks / bucket_count)
    return AP_at_n_buckets(sortedRanks, n, total_ranks)

def AP_at_testsize(sortedRanks, cutoff=None, totalRanks=None):
    return AP_at_n(sortedRanks, len(sortedRanks))

def Recall_at_10(sortedRanks, cutoff=None, totalRanks=None):
    return Recall_at_k(sortedRanks, 10)

def Recall_at_testsize(sortedRanks, cutoff=None, totalRanks=None):
    return Recall_at_k(sortedRanks, len(sortedRanks))

def Recall_at_avg(sortedRanks, cutoff=None, totalRanks=None):
    return Recall_at_k(sortedRanks, np.floor(np.average(sortedRanks)))

def Recall_at_avg_buckets(sortedRanks, cutoff, total_ranks):
    return Recall_at_k_buckets(sortedRanks, np.floor(np.average(sortedRanks)) * (total_ranks / bucket_count), total_ranks)

def MAP_std_intervals(sortedRanks, cutoff=None, totalRanks=None):
    return MAP_intervals(sortedRanks, cutoff, totalRanks)

def NDCG_at_avg(sortedRanks, cutoff=None, totalRanks=None):
    return NDCG_at_n(sortedRanks, np.floor(np.average(sortedRanks)))

def NDCG_at_avg_buckets(sortedRanks, cutoff, total_ranks):
    return NDCG_at_n_buckets(sortedRanks, np.floor(np.average(sortedRanks)) * (total_ranks / bucket_count), total_ranks)

def NDCG_at_recall(sortedRanks, recallLevel, totalRanks=None):
    recall_count = int(math.ceil(len(sortedRanks) * recallLevel))
    if recall_count < 1:
        return (0, 0)
    return NDCG_at_n(sortedRanks, sortedRanks[recall_count - 1])

def NDCG_at_50recall(sortedRanks, cutoff=None, totalRanks=None):
    if len(sortedRanks) % 2 == 0:
        n = (sortedRanks[int(len(sortedRanks)/2) - 1]+sortedRanks[int(len(sortedRanks)/2)]) / 2
    else:
        n = sortedRanks[int(len(sortedRanks)/2)]
    return NDCG_at_n(sortedRanks, n)

def NDCG_at_50recall_buckets(sortedRanks, cutoff, total_ranks):
    if len(sortedRanks) % 2 == 0:
        n = (sortedRanks[int(len(sortedRanks)/2) - 1]+sortedRanks[int(len(sortedRanks)/2)]) / 2
    else:
        n = sortedRanks[int(len(sortedRanks)/2)]
    n = n * (total_ranks / bucket_count)
    return NDCG_at_n_buckets(sortedRanks, n, total_ranks)

def Recall_at_k_buckets(sortedRanks, k, total_ranks):
    index_count = defaultdict(lambda : 0)
    count = 0
    for rank in sortedRanks:
        count += 1
        for i in range(1, bucket_count+1):
            if rank <= i:
                index_count[i] += 1
    prev_count = 0
    value = 0
    cur_bucket_max = 0
    for i in range(1, bucket_count+1):
        cur_bucket_max = i * (total_ranks / bucket_count)
        if cur_bucket_max <= k:
            prev_count = index_count[i]
        else:
            count = index_count[i]
            above = cur_bucket_max - k
            below = k - (i - 1) * (total_ranks / bucket_count)
            ratio = below / (above + below)
            value = (prev_count + (count - prev_count) * ratio)
            break
    v = value / len(sortedRanks)
    return (v if v < 1 else 1, value)

def Recall_at_k(sortedRanks, k, totalRanks=None):
    count = 0
    for rank in sortedRanks:
        if rank > k or count == k:
            return (count / len(sortedRanks), count)
        count += 1
    return (count / len(sortedRanks), count)

def P_at_recall_buckets(sortedRanks, recallLevel, total_ranks):
    recall_count = int(math.ceil(len(sortedRanks) * recallLevel))
    return Recall_at_k_buckets(sortedRanks, recall_count, total_ranks)


def P_at_k_buckets(sortedRanks, k, total_ranks):
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(1, bucket_count + 1):
            if rank <= i:
                index_count[i] += 1
    prev_count = 0
    value = 0
    counter = 0
    for i in range(1, bucket_count+1):
        cur_bucket_max = i * (total_ranks / bucket_count)
        count = index_count[i]
        if cur_bucket_max <= k:
            prev_count = count
        else:
            above = cur_bucket_max - k
            below = k - (i - 1) * (total_ranks / bucket_count)
            ratio = below / (above + below)
            value = (prev_count + (count - prev_count) * ratio) / k
            counter = (prev_count + (count - prev_count) * ratio)
            break
    return (value if value < 1 else 1, counter)


def AP_buckets(sortedRanks, cutoff, total_ranks):
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(1, bucket_count+1):
            if rank <= i:
                index_count[i] += 1
    sum_ = 0
    for i in range(1, bucket_count+1):
        counter = index_count[i]
        value = counter / (i * (total_ranks / bucket_count))
        sum_ += value if value < 1 else 1
    return (sum_ / bucket_count, len(sortedRanks))
        
def AP(sortedRanks, cutoff=None, totalRanks=None):
    count = 0
    sum_ = 0
    
    for rank in sortedRanks:
        count += 1
        value = count / rank
        sum_ += value if value < 1 else 1
    return (sum_ / count, count)

def AP_interpolated_buckets(sortedRanks, cutoff, total_ranks):
    values = list()
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(1, bucket_count+1):
            if rank <= i:
                index_count[i] += 1
    for i in range(1, bucket_count+1):
        counter = index_count[i]
        value = counter / (i * (total_ranks / bucket_count))
        values.append(value)
    values.reverse()
    sum_ = 0
    max_ = -1
    for value in values:
        if value > max_:
            sum_ += value
            max_ = value
        else:
            sum_ += max_
    return (sum_ / len(values), len(values))

def AP_interpolated(sortedRanks, cutoff=None, totalRanks=None):
    values = list()
    count = 1
    for rank in sortedRanks:
        value = count / rank
        values.append(value if value < 1 else 1)
        count += 1
    values.reverse()
    sum_ = 0
    max_ = -1
    for value in values:
        if value > max_:
            sum_ += value
            max_ = value
        else:
            sum_ += max_
    return (sum_ / len(values), len(values))

def AP_at_n_buckets(sortedRanks, n, total_ranks):
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(bucket_count+1):
            if rank <= i:
                index_count[i] += 1
    sum_ = 0
    prev_count = 0
    denominator = 0
    counter = 0
    for i in range(1, bucket_count+1):
        cur_bucket_max = i * (total_ranks / bucket_count)
        if cur_bucket_max <= n:
            denominator += 1
            count = index_count[i]
            value = count / (i * (total_ranks / bucket_count))
            sum_ += value if value < 1 else 1
            prev_count = count
        else:
            count = index_count[i]
            above = cur_bucket_max - n
            below = n - (i - 1) * (total_ranks / bucket_count)
            ratio = below / (above + below)
            denominator += ratio
            value = (prev_count + (count - prev_count) * ratio) / n
            sum_ += value if value < 1 else 1
            counter = (prev_count + (count - prev_count) * ratio)
            break
    
    if denominator < 1:
        denominator = 1
    if denominator == 0:
        return (0, count)
    return (sum_ / denominator, counter)

def AP_at_n(sortedRanks, n, totalRanks=None):
    count = 0
    sum_ = 0
    for rank in sortedRanks:
        if rank > n or count >= n:
            break
        count += 1
        value = count / rank
        sum_ += value if value < 1 else 1
        
    denominator = min([len(sortedRanks), n])
    if denominator == 0:
        return (0, count)
    return (sum_ / denominator, count)

def R_precision_buckets(sortedRanks, total_ranks):
    return P_at_k_buckets(sortedRanks, len(sortedRanks), total_ranks)

def R_precision(sortedRanks, cutoff=None, totalRanks=None):
    testSize = len(sortedRanks)
    count = 0
    for rank in sortedRanks:
        if rank <= testSize:
            count += 1
        else:
            break
    value = count/testSize
    return (value if value < 1 else 1, count)

def NDCG_buckets(sortedRanks, cutoff, total_ranks):
    index_optimal_dcg = defaultdict(lambda : 0)
    index_relCount = defaultdict(lambda : 0)
    sum_ = 0
    prev_count = 0
    for i in range(1, bucket_count+1):
        count = 0
        for rank in sortedRanks:
            if rank <= i:
                count += 1
        index_relCount[i] = count - prev_count
        prev_count = count
        opt_dcg = min([len(sortedRanks), i * (total_ranks / bucket_count)])
        if opt_dcg == 0:
            break
        value = opt_dcg - min([sum_, len(sortedRanks)])
        index_optimal_dcg[i] = value / math.log(i + 1, 2)
        sum_ += value
    optimal_dcg = 0
    dcg = 0
    count = 0
    for i in range(1, bucket_count+1):
        optimal_dcg += index_optimal_dcg[i]
        dcg += index_relCount[i] / math.log(i + 1, 2)
        count += index_relCount[i]
    
    return (dcg / optimal_dcg, count)

def NDCG(sortedRanks, cutoff=None, totalRanks=None):
    optimal_dcg = 0
    dcg = 0
    count = 1
    for rank in sortedRanks:
        optimal_dcg += 1 / (math.log(count + 1, 2))
        dcg += 1 / (math.log(rank + 1, 2))
        count += 1
    return (dcg / optimal_dcg, count - 1)

def NDCG_at_n_buckets(sortedRanks, n, total_ranks):
    index_optimal_dcg = defaultdict(lambda : 0)
    index_relCount = defaultdict(lambda : 0)
    sum_ = 0
    prev_count = 0
    for i in range(1, 1001):
        count = 0
        for rank in sortedRanks:
            if rank <= i:
                count += 1
        index_relCount[i] = count - prev_count
        prev_count = count
        opt_dcg = min([len(sortedRanks), i * (total_ranks / bucket_count)])
        if opt_dcg == 0:
            break
        value = opt_dcg - min([sum_, len(sortedRanks)])
        index_optimal_dcg[i] = value / math.log(i + 1, 2)
        sum_ += value
    optimal_dcg = 0
    dcg = 0
    total_count = 0
    for i in range(1, bucket_count+1):
        cur_bucket_max = i * (total_ranks / bucket_count)
        if cur_bucket_max <= n:
            optimal_dcg += index_optimal_dcg[i]
            dcg += index_relCount[i] / math.log(i + 1, 2)
            total_count += index_relCount[i]
        else:
            above = cur_bucket_max - n
            below = n - (i - 1) * (total_ranks / bucket_count)
            optimal_dcg += index_optimal_dcg[i] * (below / (above + below))
            dcg += index_relCount[i] / math.log(i + 1, 2) * (below / (above + below))
            total_count += index_relCount[i]
            break    
    return (dcg / optimal_dcg, total_count)

def NDCG_at_n(sortedRanks, n, totalRanks=None):
    optimal_dcg = 0
    for rank in range(1, int(n) + 1):
        optimal_dcg += 1 / math.log(rank + 1, 2)
        
    dcg = 0
    count = 0
    for rank in sortedRanks:
        if rank <= n:
            dcg += 1 / math.log(rank + 1, 2)
            count += 1
        else:
            break
    return (dcg / optimal_dcg, count)

def PR_AUC(sortedRanks, cutoff=None, totalRanks=None):
    counter = 1
    prec_recalls = list()
    for rank in sortedRanks:
        recall = counter / len(sortedRanks)
        value = counter / rank
        precision = value if value < 1 else 1
        prec_recalls.append((precision, recall))
        counter += 1
        
    pr_auc = 0
    prev_prec = None
    prev_recall = None
    for prec, recall in prec_recalls:
        if prev_prec is not None:
            pr_auc += ((prev_prec + prec) / 2) * (recall - prev_recall)
        prev_prec = prec
        prev_recall = recall
    return (pr_auc, counter - 1)

def PR_AUC_buckets(sortedRanks, cutoff, total_ranks):
    prec_recalls = list()
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(1, bucket_count + 1):
            if rank <= i:
                index_count[i] += 1
    for i in range(1, bucket_count + 1):
        counter = index_count[i]
        recall = counter / len(sortedRanks)
        value = counter / (i * (total_ranks / bucket_count))
        precision = value if value < 1 else 1
        prec_recalls.append((precision, recall))
        
    pr_auc = 0
    prev_prec = None
    prev_recall = None
    for prec, recall in prec_recalls:
        if prev_prec is not None:
            pr_auc += ((prev_prec + prec) / 2) * (recall - prev_recall)
        prev_prec = prec
        prev_recall = recall
    return (pr_auc, counter - 1)

def PR_AUC_interpolated_buckets(sortedRanks, cutoff, total_ranks):
    prec_recalls = list()
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(1, bucket_count + 1):
            if rank <= i:
                index_count[i] += 1
    for i in range(1, bucket_count+1):
        counter = index_count[i]
        recall = counter / len(sortedRanks)
        value = counter / (i * (total_ranks / bucket_count))
        precision = value if value < 1 else 1
        prec_recalls.append((precision, recall))
    
    prec_recalls.reverse()
    interPrec_recalls = list()
    max_prec = -1
    for prec, recall in prec_recalls:
        max_prec = max(max_prec, prec)
        interPrec_recalls.append((max_prec, recall))
    interPrec_recalls.reverse()
        
    pr_auc = 0
    prev_prec = None
    prev_recall = None
    for prec, recall in interPrec_recalls:
        if prev_prec is not None:
            pr_auc += ((prev_prec + prec) / 2) * (recall - prev_recall)
        prev_prec = prec
        prev_recall = recall
    return (pr_auc, counter - 1)

def PR_AUC_interpolated(sortedRanks, cutoff=None, totalRanks=None):
    counter = 1
    prec_recalls = list()
    for rank in sortedRanks:
        recall = counter / len(sortedRanks)
        value = counter / rank
        precision = value if value < 1 else 1
        prec_recalls.append((precision, recall))
        counter += 1
    
    prec_recalls.reverse()
    interPrec_recalls = list()
    max_prec = -1
    for prec, recall in prec_recalls:
        max_prec = max(max_prec, prec)
        interPrec_recalls.append((max_prec, recall))
    interPrec_recalls.reverse()
        
    pr_auc = 0
    prev_prec = None
    prev_recall = None
    for prec, recall in interPrec_recalls:
        if prev_prec is not None:
            pr_auc += ((prev_prec + prec) / 2) * (recall - prev_recall)
        prev_prec = prec
        prev_recall = recall
    return (pr_auc, counter - 1)

def ROC_AUC(sortedRanks, k, totalRanks=None):
    roc_auc = 0
    count = 1
    prev_fpr = None
    prev_tpr = None
    tot_rel = len(sortedRanks)
    for rank in sortedRanks:
        if rank > k:
            break
        tpr = count / tot_rel
        if count > rank:
            fpr = 0
        else:
            fpr = (rank - count) / ((rank - count) + k - count)
        if prev_fpr is not None:
            roc_auc += ((prev_tpr + tpr) / 2) * (fpr - prev_fpr)
        prev_tpr = tpr
        prev_fpr = fpr
        count += 1
    return (roc_auc, count - 1)

def ROC_AUC_buckets(sortedRanks, cutoff, total_ranks):
    index_count = defaultdict(lambda : 0)
    for rank in sortedRanks:
        for i in range(1, bucket_count + 1):
            if rank <= i:
                index_count[i] += 1
    
    roc_auc = 0
    count = 1
    prev_fpr = None
    prev_tpr = None
    tot_rel = len(sortedRanks)
    for i in range(1, bucket_count + 1):
        cur_bucket_max = i * (total_ranks / bucket_count)
        count = index_count[i]
        tpr = count / tot_rel
        fp = cur_bucket_max - min(count, cur_bucket_max)
        tn = total_ranks - cur_bucket_max - (tot_rel - count)
        fpr = 0
        if fp + tn > 0:
            fpr = fp / (fp + tn)
        if prev_fpr is not None:
            roc_auc += ((prev_tpr + tpr) / 2) * (fpr - prev_fpr)
        prev_tpr = tpr
        prev_fpr = fpr
        count += 1
    return (roc_auc, len(sortedRanks))
            

def ROC_AUC_intervals(sortedRanks, cutoff, totalRanks):
    interval_tp = defaultdict(lambda : 0)
    counter = 0
    for rank in sortedRanks:
        used = False
        for i in intervals:
            if rank <= i:
                interval_tp[i] += 1
                used = True
        if used:
            counter += 1
     
    tot_authors = intervals[-1]
    tot_awards_authors = len(sortedRanks)
     
    interval_tpr = defaultdict(lambda : 0)
    interval_fpr = defaultdict(lambda : 0)
    for i in intervals:
        interval_tpr[i] =  interval_tp[i] / tot_awards_authors
        interval_fpr[i] = (i - interval_tp[i]) / tot_authors
     
    roc_auc = 0
    prev_i = None
    for i in intervals:
        if prev_i is not None:
            roc_auc += ((interval_tpr[prev_i] + interval_tpr[i])/2) * (interval_fpr[i] - interval_fpr[prev_i])
        prev_i = i
    return (roc_auc, counter)

def MAP_intervals_buckets(sortedRanks, cutoff, total_ranks):
    ap_sum = 0
    for interval in intervals:
        ap, count = AP_at_n_buckets(sortedRanks, interval, total_ranks)
        ap_sum += ap
    
    return (ap_sum / len(intervals), count)

def MAP_intervals(sortedRanks, cutoff=None, totalRanks=None):
    interval_apSum = defaultdict(lambda : 0)
    counter = 1
    count = 0
    for rank in sortedRanks:
        used = False
        for i in intervals:
            if rank <= i:
                value = counter / rank
                interval_apSum[i] += value if value < 1 else 1 
                used = True
        counter += 1
        if used:
            count += 1
    map_value = 0
    for i in intervals:
        map_value += interval_apSum[i] / i
    return (map_value / len(intervals), count)    

def MAP_geom_intervals_buckets(sortedRanks, cutoff, total_ranks, intervals):
    interval_apSum = defaultdict(lambda : 0)
    for interval in intervals:
        ap, count = AP_at_n_buckets(sortedRanks, interval, total_ranks)
        interval_apSum[interval] = ap
    
    map_value = 1
    for i in intervals:
        map_value *= (interval_apSum[i] / i)
    return (map_value ** (1/2), count)

def MAP_geom_intervals(sortedRanks, cutoff, totalRanks=None):
    interval_apSum = defaultdict(lambda : 0)
    counter = 1
    count = 0
    for rank in sortedRanks:
        used = False
        for i in intervals:
            if rank <= i:
                value = counter / rank
                interval_apSum[i] += value if value < 1 else 1
                used = True
        counter += 1
        if used:
            count += 1
    map_value = 1
    for i in intervals:
        map_value *= (interval_apSum[i] / i)
    return (map_value ** (1/2), count)    
