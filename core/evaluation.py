# https://github.com/stamakro/GCN-for-Structure-and-Function/blob/fb148d5579adbb805c1d054d24216db285198540/scripts/evaluation.py

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support


def smin(Ytrue, Ypred, termIC, nrThresholds):
	'''
    get the minimum normalized semantic distance

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrThresholds: the number of thresholds to check.

    OUTPUT:
        the minimum nsd that was achieved at the evaluated thresholds

	'''

	thresholds = np.linspace(0.0, 1.0, nrThresholds)
	ss = np.zeros(thresholds.shape)

	for i, t in enumerate(thresholds):
		ss[i] = normalizedSemanticDistance(Ytrue, (Ypred >=t).astype(int), termIC, avg=True, returnRuMi=False)

	return np.min(ss)


''' helper functions follow '''
def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi = False):
    '''
    evaluate a set of protein predictions using normalized semantic distance
    value of 0 means perfect predictions, larger values denote worse predictions,

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, predicted binary label ndarray (not compressed). Must have hard predictions (0 or 1, not posterior probabilities)
        termIC: output of ic function above

    OUTPUT:
        depending on returnRuMi and avg. To get the average sd over all proteins in a batch/dataset
        use avg = True and returnRuMi = False
        To get result per protein, use avg = False

    '''

    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)

    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru ** 2 + mi ** 2)

    if not returnRuMi:
        return sd

    return [ru, mi, sd]

def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num =  np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom =  np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom

    if avg:
        nru = np.mean(nru)

    return nru

def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num =  np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom =  np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom

    if avg:
        nmi = np.mean(nmi)

    return nmi


def fmax(Ytrue, Ypred, nrThresholds):
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        thr = np.round(t, 2)
        pr[i], rc[i], ff[i], _ = precision_recall_fscore_support(Ytrue, (Ypred >=t).astype(int), average='samples')

    return np.max(ff)


def bootstrap(Ytrue, Ypred, ic, nrBootstraps=1000, nrThresholds=51, seed=1002003445):
    '''
    perform bootstrapping (https://en.wikipedia.org/wiki/Bootstrapping)
    to estimate variance over the test set. The following metrics are used:
    protein-centric average precision, protein centric normalized semantic distance, term-centric roc auc

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrBootstraps: the number of bootstraps to perform
        nrThresholds: the number of thresholds to check for calculating smin.

    OUTPUT:
        a dictionary with the metric names as keys (auc, roc, sd) and the bootstrap results as values (nd arrays)
    '''

    np.random.seed(seed)
    seedonia = np.random.randint(low=0, high=4294967295, size=nrBootstraps)

    bootstraps_psd = np.zeros((nrBootstraps,), float)
    bootstraps_pauc = np.zeros((nrBootstraps,), float)
    bootstraps_troc = np.zeros((nrBootstraps,), float)
    bootstraps_pfmax = np.zeros((nrBootstraps,), float)

    for m in range(nrBootstraps):
    	[newYtrue, newYpred] = resample(Ytrue, Ypred, random_state=seedonia[m])

    	bootstraps_pauc[m] = average_precision_score(newYtrue, newYpred, average='samples')
    	bootstraps_psd[m] = smin(newYtrue, newYpred, ic, nrThresholds)

    	tokeep = np.where(np.sum(newYtrue, 0) > 0)[0]
    	newYtrue = newYtrue[:, tokeep]
    	newYpred = newYpred[:, tokeep]

    	tokeep = np.where(np.sum(newYtrue, 0) < newYtrue.shape[0])[0]
    	newYtrue = newYtrue[:, tokeep]
    	newYpred = newYpred[:, tokeep]

    	bootstraps_troc[m] = roc_auc_score(newYtrue, newYpred, average='macro')
    	bootstraps_pfmax[m] = fmax(newYtrue, newYpred, nrThresholds)

    return {'auc': bootstraps_pauc, 'sd': bootstraps_psd, 'roc': bootstraps_troc, 'fmax': bootstraps_pfmax}