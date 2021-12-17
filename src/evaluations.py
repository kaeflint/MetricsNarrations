from data_utils import normalize_whitespace,normalize_text
import numpy as np
def extractMetricMentions(narration,replace_identical=True,ref_mentions=None):
    metrics_list = ['recall', 'sensitivity', 'f1-score',
                    'precision', 'f2-score', 'f1','f1score','f2score','g-mean','gmean',
                    'f2', 'accuracy',
                    'auc',
                    'specificity', ]
    nn = normalize_text(narration).split()
    # replace identical metrics such as recall and sensitivity based on the mention in the table
    
    # Take care of when recall and sensitivity is mentioned together
    
    found_metrics = [s.lower()#.replace('-score', '').replace('score', '')
                     for s in set(metrics_list).intersection(nn)]
    if replace_identical:
        if 'recall' in set(found_metrics) and 'sensitivity' in set(found_metrics):
            if 'sensitivity' in ref_mentions:
                found_metrics = ' '.join(found_metrics).replace('recall','').split()
            elif 'sensitivity' in ref_mentions:
                found_metrics = ' '.join(found_metrics).replace('sensitivity','').split()
    return found_metrics
def getOverlap(ref,sys,return_len=True):
    # Get the items that are in common
    overlap = set(ref).intersection(sys)
    if return_len:
        return len(overlap)
    else:
        return overlap
def metricMentionScore(reference_metrics,sys_output):
    ref_counts =  np.sum([len(e) for e in reference_metrics])
    sys_mentions = [extractMetricMentions(r,ref_mentions=t) for r,t in zip(sys_output,reference_metrics)]
    g_sysCount = np.sum([len(e) for e in sys_mentions])
    overlap=np.sum([getOverlap(r,b) for r,b in zip(reference_metrics,sys_mentions)])
    recall=overlap/ref_counts
    precision=overlap/g_sysCount
    f1= 2*(recall*precision)/(recall+precision)
    return dict(f1_score=round(f1,5),recall=round(recall,5),precision=round(precision,5))#,sys_mentions