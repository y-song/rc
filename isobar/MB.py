import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import os

def getData(observables, filename, treename):
    return (uproot.open(filename)[treename]).arrays(observables, library='pd')

def applyCut(inputDataframe, cut):
    dataframe = inputDataframe
    nbeforecut = dataframe.shape[0]
    cutDataframe = dataframe.query(cut)
    return cutDataframe

def getRcAndErr(df, ob):
    if (ob == 'b'):
        nos = sum(df[df.b==-1]['w'])
        nss = sum(df[df.b==+1]['w'])
    elif (ob == 'b_ch'):
        nos = sum(df[df.b_ch==-1]['w'])
        nss = sum(df[df.b_ch==+1]['w'])
    elif (ob == 'bs'):
        nos = sum(df[df.bs==-1]['ws'])
        nss = sum(df[df.bs==+1]['ws'])
    else:
        print('Double check the input observable name!')
    rc = (nss-nos)/(nss+nos)
    p = nss/(nss+nos)
    w_array = np.unique(df['w'])
    err2_array = []
    variance_array_test = []
    for i in range(0, len(w_array)):
        if (ob == 'b'):
            p = sum(df[(df.w == w_array[i]) & (df.b==+1)]['w']) / sum(df[df.w == w_array[i]]['w'])
        elif (ob == 'b_ch'):
            p = sum(df[(df.w == w_array[i]) & (df.b_ch==+1)]['w']) / sum(df[df.w == w_array[i]]['w'])            
        elif (ob == 'bs'):
            p = sum(df[(df.w == w_array[i]) & (df.bs==+1)]['w']) / sum(df[df.w == w_array[i]]['ws'])
        else:
            print('Double check the input observable name!')        
        variance = p*(1-p)/df[df.w == w_array[i]].shape[0]
        w = sum(df[df.w == w_array[i]]['w'])/sum(df['w'])
        err2_array.append ( w**2*variance )
    rc_err = 2*np.sqrt(sum(err2_array))
    return rc, rc_err

def calc_hist(vals, bins=10, weights=None, density=True):
    if weights is None:
        weights = np.ones(vals.shape)
    hist, bins = np.histogram(vals, bins=bins, weights=weights)
    digits = np.digitize(vals, bins)
    errs = np.asarray([np.linalg.norm(weights[digits==i]) for i in range(1, len(bins))])
    if density:
        density_int = weights.sum() * (bins[1] - bins[0])
        hist /= density_int
        errs /= density_int        
    return hist, errs, bins

observables = ['centrality','rho','jet_index','jet_pt','jet_area','jet_neutral_fraction','jet_random_track1_charge','jet_random_track2_charge','jet_leading_track_charge',\
    'jet_subleading_track_pt','jet_subleading_track_charge']

df_mb_array = []
filepath = '/gpfs/gibbs/project/havener/ys668/other_output/rc/isobar/mb_day118/'
njet = 0
files = os.listdir(filepath)
for filename in files:
    print('reading in file', filename)
    df_temp = applyCut(getData(observables, filepath+filename, "jet_tree"), \
                 'jet_pt > 15 and jet_subleading_track_pt > 0 and jet_neutral_fraction < 0.9')
    df_mb_array.append(df_temp)
    print('read in', df_temp.shape[0], 'jets')
    njet += df_temp.shape[0]
    #if (njet > 50000000):
    #    break
df_mb = pd.concat([df for df in df_mb_array])
print('total number of jets: ', df_mb.shape[0])

print('number of jets with jet_random_track1_charge == -1:', df_mb[df_mb.jet_random_track1_charge==-1].shape[0])
print('number of jets with jet_random_track1_charge == +1:', df_mb[df_mb.jet_random_track1_charge==+1].shape[0])

df_mb.eval('b_ch = jet_leading_track_charge * jet_subleading_track_charge', inplace=True) 
df_mb.eval('b = jet_random_track1_charge * jet_random_track2_charge', inplace=True)
df_mb.eval('jet_pt_sub = jet_pt - rho * jet_area', inplace=True)
df_mb['w'] = np.ones(df_mb.shape[0])

df_mb = applyCut(df_mb, 'jet_pt_sub > 15')
df_mb_central = applyCut(df_mb, 'centrality >= 12')
df_mb_peri_peri = applyCut(df_mb, '0 <= centrality <= 3')

df_mb_central_array = [ df_mb_central[(df_mb_central.jet_pt_sub>15) & (df_mb_central.jet_pt_sub<20)],\
                        df_mb_central[(df_mb_central.jet_pt_sub>20) & (df_mb_central.jet_pt_sub<25)],\
			df_mb_central[(df_mb_central.jet_pt_sub>25) & (df_mb_central.jet_pt_sub<30)],\
			df_mb_central[(df_mb_central.jet_pt_sub>30) & (df_mb_central.jet_pt_sub<40)] ]

df_mb_peri_peri_array = [ df_mb_peri_peri[(df_mb_peri_peri.jet_pt_sub>15) & (df_mb_peri_peri.jet_pt_sub<20)],\
			  df_mb_peri_peri[(df_mb_peri_peri.jet_pt_sub>20) & (df_mb_peri_peri.jet_pt_sub<25)] ]
                          #df_mb_peri_peri[(df_mb_peri_peri.jet_pt_sub>25) & (df_mb_peri_peri.jet_pt_sub<30)],\
                          #df_mb_peri_peri[(df_mb_peri_peri.jet_pt_sub>30) & (df_mb_peri_peri.jet_pt_sub<40)] ]

rc_mb_central_array = []
rc_err_mb_central_array = []
for i in range(0, len(df_mb_central_array)):
    rc_mb_central_array.append(getRcAndErr(df_mb_central_array[i], 'b_ch')[0])
    rc_err_mb_central_array.append(getRcAndErr(df_mb_central_array[i], 'b_ch')[1])

rc_mb_peri_peri_array = []
rc_err_mb_peri_peri_array = []
for i in range(0, len(df_mb_peri_peri_array)):
    rc_mb_peri_peri_array.append(getRcAndErr(df_mb_peri_peri_array[i], 'b_ch')[0])
    rc_err_mb_peri_peri_array.append(getRcAndErr(df_mb_peri_peri_array[i], 'b_ch')[1])

random_rc_mb_central_array = []
random_rc_err_mb_central_array = []
for i in range(0, len(df_mb_central_array)):
    random_rc_mb_central_array.append(getRcAndErr(df_mb_central_array[i], 'b')[0])
    random_rc_err_mb_central_array.append(getRcAndErr(df_mb_central_array[i], 'b')[1])

random_rc_mb_peri_peri_array = []
random_rc_err_mb_peri_peri_array = []
for i in range(0, len(df_mb_peri_peri_array)):
    random_rc_mb_peri_peri_array.append(getRcAndErr(df_mb_peri_peri_array[i], 'b')[0])
    random_rc_err_mb_peri_peri_array.append(getRcAndErr(df_mb_peri_peri_array[i], 'b')[1])

rc_random = np.array([-0.22518997953719813, -0.21441642135412523, -0.2129313872595826])
rc_random_err = np.array([0.002925290181545458, 0.006160481432193258, 0.01141150183243689])

plt.rcParams['font.size'] = 15
fig=plt.figure(figsize=(5,5))
plt.errorbar([17.5, 22.5, 27.5, 35], rc_mb_central_array, marker='o', ls='none', color='tab:red')
plt.errorbar([17.5, 22.5], rc_mb_peri_peri_array, marker='o', ls='none', color='blue')
#plt.errorbar([7.5], rc_mb_peripheral_array, marker='o', ls='none', color='tab:green')
#plt.errorbar([22.5], rc_random[0], marker='o', ls='none', color='k')
plt.errorbar([17.5, 22.5, 27.5, 35], rc_mb_central_array, xerr=[2.5, 2.5, 2.5, 5], yerr=rc_err_mb_central_array, capsize=3, ls='none', label='$0-20\%$', color='tab:red')
plt.errorbar([17.5, 22.5], rc_mb_peri_peri_array, xerr=[2.5, 2.5], yerr=rc_err_mb_peri_peri_array, capsize=3, ls='none', label='$60-80\%$', color='blue')
eb1 = plt.errorbar([17.5, 22.5, 27.5, 35], random_rc_mb_central_array, xerr=[2.5, 2.5, 2.5, 5], yerr=random_rc_err_mb_central_array, capsize=3, ls='none', label='$0-20\%$, random tracks in jets', color='tab:red')
eb2 = plt.errorbar([17.5, 22.5], random_rc_mb_peri_peri_array, xerr=[2.5, 2.5], yerr=random_rc_err_mb_peri_peri_array, capsize=3, ls='none', label='$60-80\%$, random tracks in jets', color='b')
eb3 = plt.errorbar([22.5, 27.5, 35], rc_random, xerr=[2.5,2.5,5], yerr=rc_random_err, capsize=3, ls='none', label='pp, random tracks in jets', color='k')
eb1[-1][0].set_linestyle('--')
eb1[-1][1].set_linestyle('--')
eb2[-1][0].set_linestyle('--')
eb2[-1][1].set_linestyle('--')
eb3[-1][0].set_linestyle('--')
eb3[-1][1].set_linestyle('--')
plt.xlabel(r'jet $p_{\mathrm{T}}-\rho A$ [GeV]', fontsize=15)
plt.axhline(y=0.0, color='grey', linestyle='--')
plt.ylabel('charged $r_c$', fontsize=15)
plt.ylim(-0.6,0.1)
#plt.xlim(15,40)1
#plt.legend(loc='lower right', fontsize=15, framealpha=1)
plt.savefig('MB.png', bbox_inches='tight')
print("0-20:", rc_mb_central_array, rc_err_mb_central_array)
print("60-80:", rc_mb_peri_peri_array, rc_err_mb_peri_peri_array)
print("0-20:", random_rc_mb_central_array, random_rc_err_mb_central_array)
print("60-80:", random_rc_mb_peri_peri_array, random_rc_err_mb_peri_peri_array)
#0-20: [-0.02204835947487408, -0.032725810457593645, -0.07659991597815433, -0.14251207729468598] [0.001046830861901199, 0.0034372248353977944, 0.011798929580021253, 0.03439768676613729]
#60-80: [-0.3158756137479542, -0.3170731707317073] [0.03838436407317078, 0.10473337835422636]
#0-20: [-0.005810564320502454, -0.003622940734158179, -0.0031496062992125984, -0.046153846153846156] [0.0005900543277656326, 0.001930375684858134, 0.006726932551510942, 0.018293412188883414]
#60-80: [-0.009900990099009901, -0.6666666666666666] [0.02217862475705126, 0.03406977052534528]
