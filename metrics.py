import numpy as np
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve, accuracy_score


def compute_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_aupr(labels, preds):
    p, r, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    aupr = auc(r, p)
    return aupr


def compute_mcc(labels, preds):
    labels = labels.astype(np.float64)
    preds = preds.astype(np.float64)
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def acc_score(labels, preds):
    acc = accuracy_score(labels.flatten(), preds.flatten())
    return acc

def compute_performance_max(labels, preds):
    predictions_max = None
    p_max = 0
    r_max = 0
    sp_max = 0
    mcc_max = 0
    acc_max = 0
    t_max = 0
    for t in range(1, 1000):
        threshold = t / 1000.0
        predictions = (preds > threshold).astype(np.int32)
        
        mcc = compute_mcc(labels, predictions)
        acc = acc_score(labels, predictions)
        
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = len(labels)-tp-fn-fp
        
        if tp + fp > 0:
            p = tp / (1.0 * (tp + fp))
        else:
            p = 0.0
        if tp + fn > 0:
            r = tp / (1.0 * (tp + fn))
        else:
            r = 0.0
            
        if tn + fp > 0:
            sp = tn / (1.0 * (tn + fp))
        else:
            sp = 0.0
            
        if mcc > mcc_max:
            p_max = p
            r_max = r
            sp_max = sp
            mcc_max = mcc
            acc_max = acc
            t_max = threshold
            predictions_max = predictions

    return p_max, r_max, sp_max, mcc_max, acc_max, t_max, predictions_max

def compute_performance(labels, preds, threshold):
    predictions = (preds > threshold).astype(np.int32)
    tp = np.sum(labels * predictions)
    fp = np.sum(predictions) - tp
    fn = np.sum(labels) - tp
    tn = len(labels)-tp-fn-fp

    if tp + fp > 0:
        p = tp / (1.0 * (tp + fp))
    else:
        p = 0.0
    if tp + fn > 0:
        r = tp / (1.0 * (tp + fn))
    else:
        r = 0.0
        
    if tn + fp > 0:
        sp = tn / (1.0 * (tn + fp))
    else:
        sp = 0.0
        
    mcc = compute_mcc(labels, predictions)
    acc = acc_score(labels, predictions)

    return p, r, sp, mcc, acc, predictions
