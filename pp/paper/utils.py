import math
import numpy as np
import uproot


def get_data(observables, filename, treename):
    return uproot.open(filename)[treename].arrays(observables, library='pd')


def calc_hist(vals, bins=10, weights=None, density=False):
    # NOTE: fig15_multiplicity uses a different normalization (per-bin width,
    # then divides by weights.sum()). Keep that variant local to that notebook.
    if weights is None:
        weights = np.ones(vals.shape)
    hist, edges = np.histogram(vals, bins=bins, weights=weights)
    digits = np.digitize(vals, edges)
    errs = np.asarray([np.linalg.norm(weights[digits == i])
                       for i in range(1, len(edges))])
    if density:
        norm = weights.sum() * (edges[1] - edges[0])
        hist /= norm
        errs /= norm
    return hist, errs, edges


def get_rc_and_err(df, ob):
    # Returns (rc, rc_err).
    # NOTE: fig3_fig5 and fig7_closure need a 3-return variant that also
    # returns p_final and prints it — keep that version local to those notebooks.
    if ob == 'b':
        nos = sum(df[df.b == -1]['w'])
        nss = sum(df[df.b == +1]['w'])
    elif ob == 'b_ch':
        nos = sum(df[df.b_ch == -1]['w'])
        nss = sum(df[df.b_ch == +1]['w'])
    elif ob == 'bs':
        nos = sum(df[df.bs == -1]['ws'])
        nss = sum(df[df.bs == +1]['ws'])
    else:
        raise ValueError(f"Unknown observable '{ob}'. Expected 'b', 'b_ch', or 'bs'.")
    rc = (nss - nos) / (nss + nos)
    w_array = np.unique(df['w'])
    err2_array = []
    for i in range(len(w_array)):
        mask = df.w == w_array[i]
        if ob == 'b':
            p = sum(df[mask & (df.b == +1)]['w']) / sum(df[mask]['w'])
        elif ob == 'b_ch':
            p = sum(df[mask & (df.b_ch == +1)]['w']) / sum(df[mask]['w'])
        elif ob == 'bs':
            p = sum(df[mask & (df.bs == +1)]['w']) / sum(df[mask]['ws'])
        variance = p * (1 - p) / df[mask].shape[0]
        w = sum(df[mask]['w']) / sum(df['w'])
        err2_array.append(w ** 2 * variance)
    rc_err = 2 * np.sqrt(sum(err2_array))
    return rc, rc_err


def get_rc_and_err_unweighted(df, ob):
    # Unweighted version: uses raw counts instead of weighted sums.
    if ob == 'bs':
        nos = df[df.bs == -1].shape[0]
        nss = df[df.bs == +1].shape[0]
    else:
        raise ValueError(f"Unknown observable '{ob}'. Expected 'bs'.")
    n = nss + nos
    p = nss / n
    rc = (nss - nos) / n
    rc_err = 2 * np.sqrt(p * (1 - p) / n)
    return rc, rc_err


def get_rc_and_err_new(df, ob):
    # Like get_rc_and_err but uses per-row reweighting factors stored in df['r'].
    if ob == 'b':
        nos = sum(np.array(df[df.b == -1]['w']) * np.array(df[df.b == -1]['r']))
        nss = sum(np.array(df[df.b == +1]['w']) * np.array(df[df.b == +1]['r']))
    elif ob == 'b_ch':
        nos = sum(np.array(df[df.b_ch == -1]['w']) * np.array(df[df.b_ch == -1]['r']))
        nss = sum(np.array(df[df.b_ch == +1]['w']) * np.array(df[df.b_ch == +1]['r']))
    elif ob == 'bs':
        nos = sum(np.array(df[df.bs == -1]['ws']) * np.array(df[df.bs == -1]['r']))
        nss = sum(np.array(df[df.bs == +1]['ws']) * np.array(df[df.bs == +1]['r']))
    else:
        raise ValueError(f"Unknown observable '{ob}'. Expected 'b', 'b_ch', or 'bs'.")
    rc = (nss - nos) / (nss + nos)
    w_array = np.unique(df['w'])
    err2_array = []
    for i in range(len(w_array)):
        mask = df.w == w_array[i]
        if ob == 'b':
            p = (sum(np.array(df[mask & (df.b == +1)]['w']) * np.array(df[mask & (df.b == +1)]['r']))
                 / sum(np.array(df[mask]['w']) * np.array(df[mask]['r'])))
        elif ob == 'b_ch':
            p = (sum(np.array(df[mask & (df.b_ch == +1)]['w']) * np.array(df[mask & (df.b_ch == +1)]['r']))
                 / sum(np.array(df[mask]['w'] * df[mask]['r'])))
        elif ob == 'bs':
            p = (sum(np.array(df[mask & (df.bs == +1)]['w']) * np.array(df[mask & (df.bs == +1)]['r']))
                 / sum(np.array(df[mask]['ws'] * df[mask]['r'])))
        variance = p * (1 - p) / df[mask].shape[0]
        w = sum(df[mask]['w']) / sum(df['w'])
        err2_array.append(w ** 2 * variance)
    rc_err = 2 * np.sqrt(sum(err2_array))
    return rc, rc_err


def subtract_fake(data, fake_rate, fake_dist):
    return (data - fake_rate * fake_dist) / (1 - fake_rate)


def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)
