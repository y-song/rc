import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
from charge_correlator.plot.utils import (get_data, calc_hist, get_rc_and_err, get_rc_and_err_unweighted, get_rc_and_err_new)

EMB_FILE   = '/nfs/roberts/home/ys668/output/run12/rc_reweighted_to_herwig_2d.csv'
DATA_FILE  = '/nfs/roberts/pi/pi_lbh35/youqi/data0613_chargedrc.root'

COLUMNS    = ['pt', 'b', 'w', 'y1', 'y2', 'phi1', 'phi2',
              'pts', 'bs', 'ws', 'y1s', 'y2s', 'phi1s', 'phi2s', 'nefs']

PT_EDGES   = [15, 20, 25, 30, 40, 60]
PT_CENTERS = [17.5, 22.5, 27.5, 35, 50]
PT_XERR    = [2.5, 2.5, 2.5, 5, 10]
B_BINS     = np.array([-1.2, 0.0, 1.2])

plt.rcParams['font.size'] = 12


def bin_df_in_pts(df, col='pts'):
    return [df[(df[col] > PT_EDGES[i]) & (df[col] < PT_EDGES[i+1])] for i in range(5)]

def delta_phi(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)

def get_rc_and_err_arr(df_arr, ob):
    rc_arr, err_arr = [], []
    for df in df_arr:
        rc, err = get_rc_and_err(df, ob)
        rc_arr.append(rc)
        err_arr.append(err)
    return rc_arr, err_arr

def get_rc_and_err_unweighted_arr(df_arr, ob):
    rc_arr, err_arr = [], []
    for df in df_arr:
        rc, err = get_rc_and_err_unweighted(df, ob)
        rc_arr.append(rc)
        err_arr.append(err)
    return rc_arr, err_arr

def get_rc_and_err_new_arr(df_arr, ob):
    rc_arr, err_arr = [], []
    for df in df_arr:
        rc, err = get_rc_and_err_new(df, ob)
        rc_arr.append(rc)
        err_arr.append(err)
    return rc_arr, err_arr

def get_matched_count_and_err(data, fake_rate, fake_dist):
    corrected = data - sum(data) * fake_rate * fake_dist
    err = np.sqrt(corrected)
    return corrected, err

# 3-return variant
def _get_rc_p(df, ob):
    if ob == 'b':
        nos, nss = sum(df[df.b == -1]['w']),  sum(df[df.b == +1]['w'])
    elif ob == 'b_ch':
        nos, nss = sum(df[df.b_ch == -1]['w']), sum(df[df.b_ch == +1]['w'])
    elif ob == 'bs':
        nos, nss = sum(df[df.bs == -1]['ws']), sum(df[df.bs == +1]['ws'])
    else:
        raise ValueError(f"Unknown observable '{ob}'")
    rc = (nss - nos) / (nss + nos)
    p_final = nss / (nss + nos)
    w_array = np.unique(df['w'])
    err2_array = []
    for w_i in w_array:
        mask = df.w == w_i
        if ob == 'b':
            p = sum(df[mask & (df.b == +1)]['w'])  / sum(df[mask]['w'])
        elif ob == 'b_ch':
            p = sum(df[mask & (df.b_ch == +1)]['w']) / sum(df[mask]['w'])
        elif ob == 'bs':
            p = sum(df[mask & (df.bs == +1)]['w'])  / sum(df[mask]['ws'])
        variance = p * (1 - p) / df[mask].shape[0]
        w = sum(df[mask]['w']) / sum(df['w'])
        err2_array.append(w**2 * variance)
    rc_err = 2 * np.sqrt(sum(err2_array))
    return rc, rc_err, p_final


# Process embedding

df_allreco = pd.read_csv(EMB_FILE, low_memory=False).query('pts>15 and bs!=-9 and w<1').copy()
df_allreco['distance1'] = np.sqrt(
    delta_phi(df_allreco['phi1'].values, df_allreco['phi1s'].values)**2
    + (df_allreco['y1'].values - df_allreco['y1s'].values)**2)
df_allreco['distance2'] = np.sqrt(
    delta_phi(df_allreco['phi2'].values, df_allreco['phi2s'].values)**2
    + (df_allreco['y2'].values - df_allreco['y2s'].values)**2)
df_match   = df_allreco.query('pt!=-9 and b!=-9 and distance1<0.02 and distance2<0.02')
df_allfake = df_allreco.query('pt==-9 or b==-9 or distance1>0.02 or distance2>0.02')

df_allreco_pts = bin_df_in_pts(df_allreco)
df_match_pts   = bin_df_in_pts(df_match)
df_allfake_pts = bin_df_in_pts(df_allfake)

rc_allreco, err_allreco = get_rc_and_err_arr(df_allreco_pts, 'bs')
rc_match,   err_match   = get_rc_and_err_arr(df_match_pts,   'bs')
rc_allfake, err_allfake = get_rc_and_err_arr(df_allfake_pts, 'bs')

# fig, ax = plt.subplots(figsize=(4, 4))
# ax.errorbar(PT_CENTERS, rc_allfake, xerr=PT_XERR, yerr=err_allfake, capsize=3, color='b', ls='none', label='mistagged')
# ax.errorbar(PT_CENTERS, rc_allreco, xerr=PT_XERR, yerr=err_allreco, capsize=3, color='k', ls='none', label='matched + mistagged')
# ax.errorbar(PT_CENTERS, rc_match,   xerr=PT_XERR, yerr=err_match,   capsize=3, color='r', ls='none', label='matched')
# ax.set_xlabel(r'$p_{\mathrm{T,jet}}^{\mathrm{det}}$ [GeV$/c$]', fontsize=15)
# ax.set_ylabel('det.-level charged $r_c$')
# ax.set_ylim(-0.6, -0.1)
# ax.legend(loc='lower center')
# ax.text(30, -0.33, 'STAR Simulation\n$p$+$p$ $\\sqrt{s}=200$ GeV\nanti-$k_{\\mathrm{T}}$ jets, $R=0.4$')
# plt.tight_layout()
# plt.show()

b_fake_array = []
for df_f in df_allfake_pts:
    y, _, _ = calc_hist(df_f['bs'], bins=B_BINS, weights=df_f['ws'], density=True)
    b_fake_array.append(y / sum(y))

b_match_array, b_match_err_array = [], []
for df_m in df_match_pts:
    y, _, _ = calc_hist(df_m['bs'], bins=B_BINS, weights=df_m['ws'], density=True)
    b_match_array.append(y / sum(y))
    p, q = y[0] / sum(y), y[1] / sum(y)
    eff, eff_err, _ = calc_hist(df_m['bs'], bins=[B_BINS[0], B_BINS[-1]], weights=df_m['ws'], density=True)
    sqrt_n_over_n = eff_err[0] / eff[0]
    b_match_err_array.append(np.array([np.sqrt(p * q) * sqrt_n_over_n] * 2))

matched_mask = ((df_allreco.pt != -9) & (df_allreco.b != -9) &
                (df_allreco.distance1 < 0.02) & (df_allreco.distance2 < 0.02))
match_fake, _ = np.histogram(df_allreco['pts'],               bins=PT_EDGES, weights=df_allreco['ws'])
match, _      = np.histogram(df_allreco[matched_mask]['pts'], bins=PT_EDGES, weights=df_allreco[matched_mask]['ws'])
fake_rate_array = 1 - match / match_fake

# Process data

tree_data = uproot.open(DATA_FILE)['ResultTree'].arrays(['pt', 'b'], library='np')
df_data = pd.DataFrame({
    'pts': np.concatenate(tree_data['pt']),
    'bs':  np.concatenate(tree_data['b']),
}).query('pts>15 and bs!=-9 and bs!=0')

df_data_pts = bin_df_in_pts(df_data)
rc_data, err_data = get_rc_and_err_unweighted_arr(df_data_pts, 'bs')

b_data_array = []
for df_d in df_data_pts:
    y, _, _ = calc_hist(df_d['bs'], bins=B_BINS)
    b_data_array.append(y)


# 1. Mistagged subtraction
b_fakesubtracted_array, b_fakesubtractederr_array = [], []
rc1_array, rc1_err_array = [], []

for i in range(5):
    b_sub, _ = get_matched_count_and_err(b_data_array[i], fake_rate_array[i], b_fake_array[i])
    nos, nss = b_sub[0], b_sub[1]
    n, p = nos + nss, nss / (nos + nss)
    rc1_array.append((nss - nos) / n)
    rc1_err_array.append(2 * np.sqrt(p * (1 - p) / n))
    b_fakesubtracted_array.append(b_sub)
    b_fakesubtractederr_array.append(np.array([np.sqrt(p * (1 - p) / n)] * 2) * sum(b_sub))

# fig, ax = plt.subplots(figsize=(4, 4))
# ax.errorbar(PT_CENTERS, rc_data,   xerr=PT_XERR, yerr=err_data,      capsize=3, color='k', ls='none', label='raw data')
# ax.errorbar(PT_CENTERS, rc1_array, xerr=PT_XERR, yerr=rc1_err_array, capsize=3, color='r', ls='none', label='data after\nmistagged subtraction')
# ax.set_xlabel(r'$p_{\mathrm{T,jet}}^{\mathrm{det}}$ [GeV$/c$]', fontsize=15)
# ax.set_ylabel('det.-level charged $r_c$')
# ax.set_ylim(-0.6, -0.1)
# ax.legend(loc='lower center')
# ax.text(20, -0.25, 'STAR\n$p$+$p$ $\\sqrt{s}=200$ GeV\nanti-$k_{\\mathrm{T}}$ jets, $R=0.4$')
# plt.tight_layout()
# plt.show()

# 2. pT correction

reweight_array, reweight_err_array = [], []
for i in range(5):
    num     = b_fakesubtracted_array[i] / sum(b_fakesubtracted_array[i])
    denom   = b_match_array[i]          / sum(b_match_array[i])
    num_err = b_fakesubtractederr_array[i] / sum(b_fakesubtracted_array[i])
    den_err = b_match_err_array[i]         / sum(b_match_array[i])
    rw      = num / denom
    reweight_array.append(rw)
    reweight_err_array.append(rw * np.sqrt((num_err / num)**2 + (den_err / denom)**2))

# Reweight embedding
rw = np.array(reweight_array)   # shape (5, 2): rw[i,0]=os weight, rw[i,1]=ss weight
pts_vals = df_match['pts'].values
bs_vals  = df_match['bs'].values
bin_idx  = np.clip(np.digitize(pts_vals, PT_EDGES) - 1, 0, 4)
df_match = df_match.copy()
df_match['r'] = np.where(bs_vals == -1, rw[bin_idx, 0], rw[bin_idx, 1])

# Central values for rc and statistical uncertainty due to finite number of embedding jets
df_match_pt = bin_df_in_pts(df_match, col='pt')
rc_match_corrected, err_match_corrected = get_rc_and_err_new_arr(df_match_pt, 'bs')
print('central rc values:', rc_match_corrected)


# Statistical uncertainty from reweight_err_array
# rwerr_match_corrected = []
# 
# for i in range(5):
#     df_cut_pt = df_match[(df_match['pt'] > PT_EDGES[i]) & (df_match['pt'] < PT_EDGES[i+1])]
#     df_cut_pt_pts = bin_df_in_pts(df_cut_pt)
#     df_cut_pt_ss  = df_cut_pt.query('bs==1')
#     p_array, prob_ss_array = [], []
#     for j in range(5):
#         _, _, p = _get_rc_p(df_cut_pt_pts[j], 'bs')
#         prob_ss = (sum(df_cut_pt_ss[(df_cut_pt_ss.pts > PT_EDGES[j]) & (df_cut_pt_ss.pts < PT_EDGES[j+1]) &
#                                     (df_cut_pt_ss.pt  > PT_EDGES[i]) & (df_cut_pt_ss.pt  < PT_EDGES[i+1])]['w']) /
#                    sum(df_cut_pt_ss[(df_cut_pt_ss.pt > PT_EDGES[i]) & (df_cut_pt_ss.pt  < PT_EDGES[i+1])]['w']))
#         p_array.append(p)
#         prob_ss_array.append(prob_ss)
#     p_err = np.sqrt(sum((reweight_err_array[j][1] * p_array[j] * prob_ss_array[j])**2 for j in range(5)))
#     rwerr_match_corrected.append(2 * p_err)
# 
# rc_rwerr   = np.asarray(rwerr_match_corrected)
# rc_err     = np.asarray(err_match_corrected)
# rc_err_tot = np.sqrt(rc_rwerr**2 + rc_err**2)

# fig, ax = plt.subplots(figsize=(4, 4))
# ax.errorbar(PT_CENTERS, rc_match_corrected, xerr=PT_XERR, yerr=rc_err_tot,
#             capsize=3, color='k', ls='none', label='fully corrected')
# ax.errorbar(PT_CENTERS, rc1_array, xerr=PT_XERR, yerr=rc1_err_array,
#             capsize=3, color='r', ls='none', label='after mistagged correction')
# ax.set_xlabel('full jet $p_{\\mathrm{T}}$ [GeV$/c$]')
# ax.set_ylabel('charged $r_c$')
# ax.set_ylim(-0.6, -0.1)
# ax.legend(loc='lower center')
# ax.text(20, -0.25, 'STAR data\n$p$+$p$ $\\sqrt{s}=200$ GeV\nanti-$k_{\\mathrm{T}}$ jets, $R=0.4$')
# plt.tight_layout()
# plt.show()