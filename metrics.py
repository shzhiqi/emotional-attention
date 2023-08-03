__version__ = 'beta 1.0'
__author__ = 'zhiqi'
import os
import math
import glob
import numpy as np
import scipy.misc
import scipy.ndimage
import random
import pyemd
from tqdm import tqdm
from collections import OrderedDict

def calc_all_maps(images):
    assert(len(images) != 0)
    all_map = None
    for ind, image in enumerate(images):
        im = scipy.misc.imread(image)
        if ind == 0:
            all_map = np.zeros(im.shape, dtype=np.float)
        all_map += im
    return all_map

def nss(pred_sal, fix_map):
    fix_map = fix_map.astype(np.bool)
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    pred_sal = (pred_sal - np.mean(pred_sal)) / np.std(pred_sal)
    return np.mean(pred_sal[fix_map])

def auc_judd(pred_sal, fix_map, jitter=True):
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = pred_sal.flatten().astype(np.float)
    if jitter:
        jitter = np.random.rand(pred_sal.shape[0]) / 1e7
        pred_sal += jitter
    pred_sal = (pred_sal - pred_sal.min())/(pred_sal.max() - pred_sal.min())
    all_thres = np.sort(pred_sal[fix_map])[::-1]

    tp = np.concatenate([[0], np.linspace(0.0, 1.0, all_thres.shape[0]), [1]])
    fp = np.zeros((all_thres.shape[0]))
    sorted_sal = np.sort(pred_sal)
    for ind, thres in enumerate(all_thres):
        above_thres = sorted_sal.shape[0] - sorted_sal.searchsorted(thres, side='left')
        fp[ind] = (above_thres-ind) * 1. / (pred_sal.shape[0] - all_thres.shape[0])
    fp = np.concatenate([[0], fp, [1]])
    return np.trapz(tp, fp)

def auc_borji(pred_sal, fix_map, n_split=100, step_size=.1):
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = pred_sal.flatten().astype(np.float)
    pred_sal = (pred_sal - pred_sal.min())/(pred_sal.max() - pred_sal.min())
    sal_fix = pred_sal[fix_map]
    sorted_sal_fix = np.sort(sal_fix)
    
    r = np.random.randint(0, pred_sal.shape[0], (sal_fix.shape[0], n_split))
    rand_fix = pred_sal[r]
    auc = np.zeros((n_split))
    for i in xrange(n_split):
        cur_fix = rand_fix[:,i]
        sorted_cur_fix = np.sort(cur_fix)
        max_val = np.maximum(cur_fix.max(), sal_fix.max())
        tmp_all_thres = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros((tmp_all_thres.shape[0]))
        fp = np.zeros((tmp_all_thres.shape[0]))
        for ind, thres in enumerate(tmp_all_thres):
            tp[ind] = (sorted_sal_fix.shape[0] - sorted_sal_fix.searchsorted(thres, side='left'))*1./sal_fix.shape[0]
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(thres, side='left'))*1./sal_fix.shape[0]
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)

def cc(pred_sal, gt_sal):
    if pred_sal.shape != gt_sal.shape:
        pred_sal = scipy.misc.imresize(pred_sal, gt_sal.shape)
    pred_sal = (pred_sal - pred_sal.mean())/(pred_sal.std())
    gt_sal = (gt_sal - gt_sal.mean())/(gt_sal.std())
    return np.corrcoef(pred_sal.flat, gt_sal.flat)[0, 1]

def sim(pred_sal, gt_sal):
    if pred_sal.shape != gt_sal.shape:
        pred_sal = scipy.misc.imresize(pred_sal, gt_sal.shape)
    pred_sal = pred_sal.astype(np.float)
    gt_sal = gt_sal.astype(np.float)
    pred_sal = (pred_sal - pred_sal.min())/(pred_sal.max()-pred_sal.min())
    pred_sal = pred_sal / pred_sal.sum()
    gt_sal = (gt_sal - gt_sal.min())/(gt_sal.max()-gt_sal.min())
    gt_sal = gt_sal / gt_sal.sum()
    diff = np.minimum(pred_sal, gt_sal)
    return np.sum(diff)

def kl(pred_sal, fix_map):
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    eps = np.finfo(float).eps
    pred_sal = pred_sal.astype(np.float)
    fix_map = fix_map.astype(np.float)
    pred_sal = pred_sal / pred_sal.sum()
    fix_map = fix_map / fix_map.sum()
    return np.sum(fix_map * np.log(eps + fix_map / (pred_sal + eps)))
    
def ig(pred_sal, fix_map, base_sal):
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    if base_sal.shape != fix_map.shape:
        base_sal = scipy.misc.imresize(base_sal, fix_map.shape)   
    eps = np.finfo(float).eps
    fix_map = fix_map.astype(np.bool)
    pred_sal = pred_sal.astype(np.float32).flatten()
    base_sal = base_sal.astype(np.float32).flatten()
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    base_sal = (base_sal - base_sal.min()) / (base_sal.max() - base_sal.min())
    pred_sal = pred_sal / pred_sal.sum()
    base_sal = base_sal / base_sal.sum()
    locs = fix_map.flatten()
    return np.mean(np.log2(eps+pred_sal[locs])-np.log2(eps+base_sal[locs])) 

def auc_shuffled(pred_sal, fix_map, base_map, n_split=100, step_size=.1):
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    assert(base_map.shape == fix_map.shape)
    pred_sal = pred_sal.flatten().astype(np.float)
    base_map = base_map.flatten().astype(np.float)
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    sal_fix = pred_sal[fix_map]
    sorted_sal_fix = np.sort(sal_fix)
    ind = np.where(base_map>0)[0]
    n_fix = sal_fix.shape[0]
    n_fix_oth = np.minimum(n_fix, ind.shape[0])
    
    rand_fix = np.zeros((n_fix_oth, n_split))
    for i in xrange(n_split):
        rand_ind = random.sample(ind, n_fix_oth)
        rand_fix[:,i] = pred_sal[rand_ind]
    auc = np.zeros((n_split))
    for i in xrange(n_split):
        cur_fix = rand_fix[:, i]
        sorted_cur_fix = np.sort(cur_fix)
        max_val = np.maximum(cur_fix.max(), sal_fix.max())
        tmp_all_thres = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros((tmp_all_thres.shape[0]))
        fp = np.zeros((tmp_all_thres.shape[0]))
        for ind, thres in enumerate(tmp_all_thres):
            tp[ind] = (sorted_sal_fix.shape[0] - sorted_sal_fix.searchsorted(thres, side='left')) * 1. / n_fix
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(thres, side='left')) * 1. / n_fix_oth
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)

def emd(pred_sal, fix_map, downsize=32):
    fix_map = scipy.misc.imresize(fix_map, np.array(fix_map.shape) / downsize)
    pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    fix_map = fix_map.astype(np.float)
    fix_map = (fix_map - fix_map.min())/(fix_map.max() - fix_map.min())
    pred_sal = pred_sal.astype(np.float)
    pred_sal = (pred_sal - pred_sal.min())/(pred_sal.max() - pred_sal.min())

    fix_map = fix_map / fix_map.sum()
    pred_sal = pred_sal / pred_sal.sum()
    
    r, c = fix_map.shape
    N = r*c
    dist = np.zeros((N, N), dtype=np.float)
    
    j = 0
    for c1 in range(c):
        for r1 in range(r):
            j = j + 1; i = 0
            for c2 in range(c):
                for r2 in range(r):
                    i = i + 1
                    dist[i-1,j-1]= math.sqrt((r1-r2)*(r1-r2)+(c1-c2)*(c1-c2))
    p = pred_sal.flatten()
    q = fix_map.flatten()
    return pyemd.emd(p, q, dist, extra_mass_penalty=0.)

def get_image_ext(folder):
    images = os.listdir(folder)
    if len(images) == 0:
        print folder, " doesn't include any images"
        assert(len(images) > 0)
    res = dict()
    for image in images:
        ext = os.path.splitext(image)[-1]
        if ext in res:
            res[ext] += 1
        else:
            res[ext] = 1
    ext = ''; cnt = 0
    for text in res:
        if res[text] > cnt:
            cnt = res[text]
            ext = text
    return ext

def check_image_exist(image_infos):
    for image_info in image_infos:
        for i in range(3):
            image_path = image_info[i]
            assert(os.path.exists(image_info[0]), "can't find image:" + image_path)


def calc_all(pred_sal_folder, gt_sal_folder, binary_folder, base='binary'):
    pred_sal_ext = get_image_ext(pred_sal_folder)
    gt_sal_ext = get_image_ext(gt_sal_folder)
    binary_ext = get_image_ext(binary_folder)

    # pred_sal_images = sorted(glob.glob(os.path.join(pred_sal_folder, "*"+pred_sal_ext)))
    # gt_sal_images = sorted(glob.glob(os.path.join(gt_sal_folder, "*"+gt_sal_ext)))
    
    assert(base in ['binary', 'pred_sal', 'gt_sal'])

    if base == 'binary':
        selected_image_names = os.path.join(binary_folder, "*"+binary_ext)
    elif base == 'pred_sal':
        selected_image_names = os.path.join(pred_sal_folder, "*"+pred_sal_ext)
    elif base == 'gt_sal':
        selected_image_names = os.path.join(gt_sal_folder, "*"+gt_sal_ext)

    selected_images = sorted(glob.glob(selected_image_names))  

    assert(len(selected_images) > 0)

    selected_names = [os.path.splitext(os.path.basename(x))[0] for x in selected_images]

    pred_sals = [os.path.join(pred_sal_folder, x+pred_sal_ext) for x in selected_names]
    gt_sals = [os.path.join(gt_sal_folder, x+gt_sal_ext) for x in selected_names]
    binarys = [os.path.join(binary_folder, x+binary_ext) for x in selected_names]

    image_infos = zip(pred_sals, gt_sals, binarys)
    

    check_image_exist(image_infos)

    metrics = ['NSS', 'AUC_Judd', 'AUC_Borji', 'sAUC', 'CC', 'SIM', 'KL', 'IG', 'EMD']
    res = OrderedDict()
    for metric in metrics:
        res[metric] = list()

    all_map = calc_all_maps(binarys)

    for ind, image_info in enumerate(tqdm(image_infos)):
        pred_sal = scipy.misc.imread(image_info[0])
        gt_sal = scipy.misc.imread(image_info[1])
        fix_map = scipy.misc.imread(image_info[2])

        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
        gt_sal = scipy.misc.imresize(gt_sal, fix_map.shape)
        
        res['NSS'].append(nss(pred_sal, fix_map))
        res['AUC_Judd'].append(auc_judd(pred_sal, fix_map))
        res['AUC_Borji'].append(auc_borji(pred_sal, fix_map))
        res['sAUC'].append(auc_shuffled(pred_sal, fix_map, all_map-fix_map))
        res['CC'].append(cc(pred_sal, gt_sal))
        res['SIM'].append(sim(pred_sal, gt_sal))
        res['KL'].append(kl(pred_sal, fix_map))
        res['IG'].append(ig(pred_sal, fix_map, all_map-fix_map))
        res['EMD'].append(emd(pred_sal, fix_map))
    et = time.time()

    for metric in metrics:
        print metric, np.mean(np.array(res[metric]))
    