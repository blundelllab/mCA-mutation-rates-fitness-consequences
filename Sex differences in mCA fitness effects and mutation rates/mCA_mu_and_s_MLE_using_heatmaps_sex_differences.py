
'''''
Watson code for estimating sex-specific mCA mutation rate and fitness effects (and their confidence intervals) using MLE heatmaps (for both male and female)
Version 1.0 (February 2023)

Input:
    1) mCA name

Outputs:
    1) heatmap plots
    2) confidence interval plots
    3) data output of s, mu, confidence intervals and heatmap data

Usage:
mCA_mu_and_s_MLE_using_heatmaps_sex_differences.py --mCA

'''''

# imported packages
from argparse import ArgumentParser
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as plticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator, MultipleLocator
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import scipy.special
import scipy.integrate as it
from scipy import integrate
from scipy.integrate import quad
from scipy.interpolate import interp1d
import copy
import glob, os
import re
import pandas as pd
from decimal import *
from operator import itemgetter
from collections import OrderedDict
import timeit
import csv
import pandas as pd
from matplotlib.gridspec import  GridSpec
import matplotlib.colors as colors
import time
from datetime import datetime
import zipfile
# import simpleaudio as sa

plt.style.use('cwpython.mplstyle') #use custom style file

#define the colors from colorbrewer2
orange1 = '#feedde'
orange2 = '#fdbe85'
orange3 = '#fd8d3c'
orange4 = '#e6550d'
orange5 = '#a63603'
blue1 = '#eff3ff'
blue2 = '#bdd7e7'
blue3 = '#6baed6'
blue4 = '#3182bd'
blue5 = '#08519c'
green1 = '#edf8e9'
green2 = '#bae4b3'
green3 = '#74c476'
green4 = '#31a354'
green5 = '#006d2c'
grey1 = '#f7f7f7'
grey2 = '#cccccc'
grey3 = '#969696'
grey4 = '#636363'
grey5 = '#252525'
purple1 = '#f2f0f7'
purple2 = '#cbc9e2'
purple3 = '#9e9ac8'
purple4 = '#756bb1'
purple5 = '#54278f'
red1 = '#fee5d9'
red2 = '#fcae91'
red3 = '#fb6a4a'
red4 = '#de2d26'
red5 = '#a50f15'

neutral_color='#fdbf6f'
gain_color = '#e31a1c'
loss_color = '#a6cee3'

def filter_by_number_mCAs(number_mCAs, dataframe): #e.g. filter by those that only have 1 total mCA
    mask = dataframe['total_mCAs']==number_mCAs
    return dataframe[mask].copy()

def mCA_count_number_mCAs(mCA, df, number_mCAs, gender):
    if gender == 'male':
        gender_df = df[df['SEX']=='M']
    if gender == 'female':
        gender_df = df[df['SEX']=='F']

    df_mCA = gender_df[gender_df['annotation']==mCA]
    df_mCA_total_number = df_mCA[df_mCA['total_mCAs']==number_mCAs]
    return len(df_mCA_total_number.groupby(['ID']))

def mCA_count_greater_than_equal_number_mCAs(mCA, df, greater_than_equal_number_mCAs, gender):
    if gender == 'male':
        gender_df = df[df['SEX']=='M']
    if gender == 'female':
        gender_df = df[df['SEX']=='F']

    df_mCA = gender_df[gender_df['annotation']==mCA]
    df_mCA_total_number = df_mCA[df_mCA['total_mCAs']>=greater_than_equal_number_mCAs]
    return len(df_mCA_total_number.groupby(['ID']))

def log_logit_CFs_sex(df, mCA, gender): #log and logit cell fractions
    single_df = filter_by_number_mCAs(1, df) #filter the dataframe to only include people that have total 1 mCA

    if gender == 'male':
        gender_df = single_df[single_df['SEX']=='M']
    if gender == 'female':
        gender_df = single_df[single_df['SEX']=='F']

    CFs_mCA = gender_df[gender_df['annotation']== mCA]['CELL_FRAC'].to_list()

    logCFs = []
    logitCFs = []

    for i in CFs_mCA:
        logitCFs.append(scipy.special.logit(float(i)))
        logCFs.append(np.log(float(i)))

    print('total '+gender+' '+mCA+' mCAs = '+str(len(logitCFs)))
    return logCFs, logitCFs

def cumulative_cell_fraction_densities_list(logit_cell_fractions, total_people):

    logit_cell_fractions = sorted(logit_cell_fractions, reverse = True)

    cumulative_number = np.arange(np.size(logit_cell_fractions))/total_people
    log_cumulative_number = np.log(cumulative_number)

    densities = []
    for a, b in zip(logit_cell_fractions, log_cumulative_number):
        if math.isinf(b) == False:
            densities.append((a, b))

    return densities

def Probtheory(l, params): #= predicted density (i.e. normalised by 2 x mu)
    total_density=0.0
    N = 9.40166610e+04 #N inferred from DNMT3A R882H

    s=params[0]

    age40_49_ratio = 119000/(119000+168000+213000)
    age50_59_ratio = 168000/(119000+168000+213000)
    age60_69_ratio = 213000/(119000+168000+213000)

    total_density= age40_49_ratio*(integrate.quad(lambda t: (N/(1-np.exp(l))*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))*(1-np.exp(l))))))*\
                             (1/9.99), 40, 49.99))[0]+\
              age50_59_ratio*(integrate.quad(lambda t: (N/(1-np.exp(l))*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))*(1-np.exp(l))))))*\
                             (1/9.99), 50, 59.99))[0]+\
               age60_69_ratio*(integrate.quad(lambda t: (N/(1-np.exp(l))*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))*(1-np.exp(l))))))*\
                             (1/9.99), 60, 69.99)[0])

    return total_density

def Probtheory_logit(l, params): #= predicted density (i.e. normalised by 2 x mu)
    total_density=0.0
    N = 9.40166610e+04 #N inferred from DNMT3A R882H

    s=params[0]

    age40_49_ratio = 119000/(119000+168000+213000)
    age50_59_ratio = 168000/(119000+168000+213000)
    age60_69_ratio = 213000/(119000+168000+213000)

    total_density= age40_49_ratio*(integrate.quad(lambda t: (N*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))))))*\
                             (1/9.99), 40, 49.99))[0]+\
              age50_59_ratio*(integrate.quad(lambda t: (N*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))))))*\
                             (1/9.99), 50, 59.99))[0]+\
               age60_69_ratio*(integrate.quad(lambda t: (N*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))))))*\
                             (1/9.99), 60, 69.99)[0])

    return total_density

def Probtheory_logit_cumulative(integral_limit, l, params): #= predicted cumulative density
    "Natural log of the probability of observing a variant at a specific cell fraction if able to sequence perfectly"
    total_density=0.0
    N = 9.40166610e+04 #N inferred from DNMT3A R882H
    sigma = 8.1
    mean = 56.53
    dt=0.1

    s = params[0]

    age40_49_ratio = 119000/(119000+168000+213000)
    age50_59_ratio = 168000/(119000+168000+213000)
    age60_69_ratio = 213000/(119000+168000+213000)

    cumulative = (integrate.quad(lambda l: ((integrate.quad(lambda t: (N*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))))))*(1/9.99), 40, 49.99))[0]*age40_49_ratio+\
                                            (integrate.quad(lambda t: (N*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))))))*(1/9.99), 50, 59.99))[0]*age50_59_ratio+\
                                            (integrate.quad(lambda t: (N*np.exp(-((np.exp(l))/(((np.exp(s*t)-1)/(N*s))))))*(1/9.99), 60, 69.99))[0]*age60_69_ratio), l, scipy.special.logit(integral_limit)))

    return np.log(cumulative[0])

def ProbDataGivenModel_logit_cumulative(params, data, integral_limit):
    "This returns the likelihood of the variant being at that cell fraction, given the model"

    x = []
    for datapoint in data:
        logit_cell_fraction = datapoint[0]
        if scipy.special.expit(logit_cell_fraction) <1.0:
            x.append(logit_cell_fraction)

    predicted_cumulative = [Probtheory_logit_cumulative(integral_limit, l, params) for l in x]

    x_y_dict = {}
    for a, b in zip(x, predicted_cumulative):
        if math.isnan(b) == False:
            x_y_dict[a]=b #a dictionary of the x and y datapoints of the cumulative theory (but only of the sampled logit spaced cell fractions)

    total_square_distance = 0
    for datapoint in data:
        logit_cell_fraction = datapoint[0]
        mutation_rate = params[1]
        if mutation_rate >0:
            if np.exp(logit_cell_fraction) <1.0:
                if logit_cell_fraction in x_y_dict.keys(): #i.e. only calculate the square distance for a reduced sample of the logit cell fractions
                    predicted_log_density = x_y_dict[logit_cell_fraction]
                    actual_density = np.exp(datapoint[1])
                    actual_density_mu = actual_density/mutation_rate
                    actual_log_density_mu = np.log(actual_density_mu)
                    if actual_log_density_mu >0: #the first actual density will always be inf so ignore this (as will end up summing all the total square distance to inf)
                        square_distance = ((actual_log_density_mu - predicted_log_density)**2)
                        total_square_distance = total_square_distance + square_distance
        if mutation_rate <0: #if it tries to fit a negative mutation rate... make the total square distance very large
            total_square_distance = 1000000

    return total_square_distance

def heatmap_logit_cumulative(cumulative_densities_list, s_range, mu_range, grid_size, mCA, labelname, integral_limit, gender):
    # Plotting the maximum likelihood estimates on a colormesh plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex = True, figsize=(16, 7))
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    gs.update(wspace=0.4)

    axisfont=15
    titlefont=18
    axislabelfont=18
    m_size=8
    scale = 1

    s_list = np.linspace(s_range[0], s_range[1], grid_size) #list of possible values for s
    mu_list = np.logspace(np.log10(mu_range[0]), np.log10(mu_range[1]), grid_size) #list of possible values for mu

    #MLE
    logProbs = np.array([[ProbDataGivenModel_logit_cumulative([s, mu], cumulative_densities_list, integral_limit) for s in s_list] for mu in mu_list])

    ## S vs mu
    # Plot the density map using nearest-neighbor interpolation
    x1 = s_list
    y1 = mu_list
    x1, y1 = np.meshgrid(x1, y1)
    logProbs_s = (-logProbs)
    max_x, max_y = np.unravel_index(np.argmax(logProbs_s), logProbs_s.shape)
    z_max = logProbs_s[max_x, max_y]
    z1 = np.exp(logProbs_s-(z_max))

    cmap = plt.cm.coolwarm #define colors

    ax1.pcolormesh(x1,y1,z1, cmap = cmap)

    #set labels
    ax1.set_xlabel('s (%)', fontsize = axislabelfont)
    ax1.set_ylabel('\u03BC', fontsize = axislabelfont)

    # calculate best values for s and mu (max points in 3D space (x,y,z))
    xmax, ymax = np.unravel_index(np.argmax(z1), z1.shape)
    mu_max = y1[xmax, ymax]
    s_max = x1[xmax, ymax]
    z_max = z1[xmax, ymax]

    ax1.scatter(s_max, mu_max, marker = '+', s = 1000, color = grey1, lw = 5)

    ax1.xaxis.set_tick_params(width=2, color = grey3, length = 6, labelsize = 16)
    ax1.yaxis.set_tick_params(width=scale, color = grey3, length = 6, labelsize = 16)
    ax1.set_yscale('log')

    print('\u03BC (ax1) = ', mu_max)
    print('s (ax1) =', s_max)
    print('z max (ax1) = ', z_max)

    ## MU vs S
    logProbs = np.array([[ProbDataGivenModel_logit_cumulative([s, mu], cumulative_densities_list, integral_limit) for mu in mu_list] for s in s_list])

    # Plot the density map using nearest-neighbor interpolation
    x1_mu = mu_list
    y1_mu = s_list
    x1_mu, y1_mu = np.meshgrid(x1_mu, y1_mu)
    logProbs_mu = (-logProbs)
    max_x_mu, max_y_mu = np.unravel_index(np.argmax(logProbs_mu), logProbs_mu.shape)
    z_max_mu = logProbs_mu[max_x_mu, max_y_mu]
    z1_mu = np.exp(logProbs_mu-(z_max_mu))

    cmap = plt.cm.coolwarm #define colors

    ax2.pcolormesh(x1_mu,y1_mu,z1_mu, cmap = cmap)
    ax2.set_xscale('log')

    #set labels
    ax2.set_xlabel('\u03BC', fontsize = axislabelfont)
    ax2.set_ylabel('s(%)', fontsize = axislabelfont)

    # calculate best values for theta and phi (max points in 3D space (x,y,z))
    xmax_mu, ymax_mu = np.unravel_index(np.argmax(z1_mu), z1_mu.shape)
    s_max_mu = y1_mu[xmax_mu, ymax_mu]
    mu_max_mu = x1_mu[xmax_mu, ymax_mu]
    z_max_mu = z1_mu[xmax_mu, ymax_mu]

    ax2.xaxis.set_tick_params(width=2, color = grey3, length = 6, labelsize = 16)
    ax2.yaxis.set_tick_params(width=scale, color = grey3, length = 6, labelsize = 16)

    ax2.scatter(mu_max_mu, s_max_mu, marker = '+', s = 1000, color = grey1, lw = 5)

    print('')
    print('\u03BC (ax2) = ', mu_max_mu)
    print('s (ax2) =', s_max_mu)
    print('z max (ax2) = ', z_max_mu)

    ax1.set_title(labelname+' '+gender, fontsize = 16, y = 1.01)
    ax2.set_title(labelname+' '+gender, fontsize = 16, y = 1.01)

    savename = labelname.replace(' ', '_')

    plt.tight_layout()
    plt.savefig('Figures/MLE_heatmaps_sex_differences_mCAs/'+savename+'_s_mu_heatmap_'+gender+'.pdf')
    # plt.show()

    return [x1, y1, z1], [x1_mu, y1_mu, z1_mu], s_max, mu_max

def confidence_interval_95(x1, y1, z1, color, ax, gender): #95% confidence interval for s

    axisfont=17
    titlefont=20
    axislabelfont=21
    m_size=8
    scale = 1

    x1y1z1 = zip(x1, y1, z1)

    xyz_list=[]
    for a, b, c in x1y1z1:
        xyz_list.append([a, b, c])

    total_prob_array=np.array([0.0 for i in range(len(xyz_list[0][0]))])
    for entry in xyz_list:
        s_array=entry[0] #i.e. x1
        prob_array=entry[2] #i.e.z1
        total_prob_array=total_prob_array+prob_array

    total_prob=sum(total_prob_array)
    normalized_prob_array=total_prob_array/total_prob

    cumulative_prob=0.0

    s_95_range=[]
    s_range_probs = []
    s_95CI_range=[]
    s_cumulative_prob_95_range=[]
    for s, p in zip(s_array, normalized_prob_array):
        s_range_probs.append((s,p))
        cumulative_prob=cumulative_prob+p
        if 0.025<cumulative_prob<0.975:
            s_95_range.append(s)
            s_95CI_range.append((s, p))
            s_cumulative_prob_95_range.append((s, p, cumulative_prob))

    lower95_s=min(s_95_range)
    upper95_s=max(s_95_range)

    print('95% confidence interval for s: lower s =', lower95_s)
    print('95% confidence interval for s: upper s =', upper95_s)

    #plotting the most likely s
    xmax, ymax = np.unravel_index(np.argmax(z1), z1.shape)
    s_mle = x1[xmax, ymax]

    #plot distribution
    s_list = []
    probs_list = []
    for (s, probs) in s_range_probs:
        s_list.append(s*100)
        probs_list.append(probs)

    s_listCI = []
    probs_listCI = []
    probs_95_list = []
    for (s, probs) in s_95CI_range:
        s_listCI.append(s*100)
        probs_listCI.append(probs)

    ax.plot(s_list, probs_list, color = color, lw = 2, label = gender)
    ax.fill_between(s_listCI, probs_listCI, color = color, alpha = 0.2)

    #plot confidence interval
    ax.plot([lower95_s*100, lower95_s*100], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([upper95_s*100, upper95_s*100], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([s_mle*100, s_mle*100], [0, 1], linestyle = ':', color = color, lw = 2)

    # Set axis limits
    mins = min(s_list)-0.5
    maxs = max(s_list)+0.5

    # Axis labels
    ax.set_xlabel('s (%)', fontsize = axislabelfont, labelpad = 10, fontweight = 'medium')
    ax.set_ylabel('probability', fontsize = axislabelfont, labelpad = 10, fontweight = 'medium')

    ax.xaxis.set_tick_params(width=scale, color = grey3, length = 6, labelsize = 16, top = False, labeltop = False)
    ax.yaxis.set_tick_params(width=scale, color = grey3, length = 6, labelsize = 16)

    #Only show the required axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)

    for axis in ['bottom','left']:
        ax.spines[axis].set_color(grey3)

    return lower95_s, upper95_s, max(probs_list), [s_list, probs_list]

def confidence_interval_95_mu(x1, y1, z1, color, ax, gender): #95% confidence interval for muation rate increase

    axisfont=17
    titlefont=20
    axislabelfont=21
    m_size=8
    scale = 1

    x1y1z1 = zip(x1, y1, z1)

    xyz_list=[]
    for a, b, c in x1y1z1:
        xyz_list.append([a, b, c])

    total_prob_array=np.array([0.0 for i in range(len(xyz_list[0][0]))])
    for entry in xyz_list:
        mu_array=entry[0] #i.e. x2
        prob_array=entry[2] #i.e.z1
        total_prob_array=total_prob_array+prob_array

    total_prob=sum(total_prob_array)
    normalized_prob_array=total_prob_array/total_prob

    cumulative_prob=0.0

    mu_95_range=[]
    mu_range_probs = []
    mu_95CI_range = []
    mu_cumulative_prob_95_range=[]
    for mu, p in zip(mu_array, normalized_prob_array):
        mu_range_probs.append((mu,p))
        cumulative_prob=cumulative_prob+p
        if 0.025<cumulative_prob<0.975:
            mu_95_range.append(mu)
            mu_95CI_range.append((mu, p))
            mu_cumulative_prob_95_range.append((mu, p, cumulative_prob))

    lower95_mu=min(mu_95_range)
    upper95_mu=max(mu_95_range)

    print('95% confidence interval for \u03BC: lower =', lower95_mu)
    print('95% confidence interval for \u03BC: upper =', upper95_mu)

    #plotting the most likely s
    xmax, ymax = np.unravel_index(np.argmax(z1), z1.shape)
    mu_mle = x1[xmax, ymax]

    #plot distribution
    mu_list = []
    probs_list = []
    for (mu, probs) in mu_range_probs:
        mu_list.append(mu)
        probs_list.append(probs)

    mu_listCI = []
    probs_listCI = []
    probs_95_list = []
    for (mu, probs) in mu_95CI_range:
        mu_listCI.append(mu)
        probs_listCI.append(probs)

    ax.plot(mu_list, probs_list, color = color, lw = 2, label = gender)
    ax.fill_between(mu_listCI, probs_listCI, color = color, alpha = 0.2)

    #plot confidence interval
    ax.plot([lower95_mu, lower95_mu], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([upper95_mu, upper95_mu], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([mu_mle, mu_mle], [0, 1], linestyle = ':', color = color, lw = 2)

    # Axis labels
    ax.set_xlabel('\u03BC', fontsize = axislabelfont, labelpad = 10, fontweight = 'medium')
    ax.set_ylabel('probability', fontsize = axislabelfont, labelpad = 10, fontweight = 'medium')

    ax.xaxis.set_tick_params(width=scale, color = grey3, length = 6, labelsize = 16, top = False, labeltop = False)
    ax.yaxis.set_tick_params(width=scale, color = grey3, length = 6, labelsize = 16)

    #Only show the required axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)

    for axis in ['bottom','left']:
        ax.spines[axis].set_color(grey3)

    return lower95_mu, upper95_mu, max(probs_list), [mu_list, probs_list]

def confidence_intervals_plot(male_s_xyz, male_mu_xyz, female_s_xyz, female_mu_xyz, mCA, labelname):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex = True, figsize=(16, 6))
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    gs.update(wspace=0.4)

    mCA_type = mCA[-1]
    if mCA_type == '+':
        male_color = '#650913'
        female_color = '#ef3d2c'
    if mCA_type == '-':
        male_color = '#213468'
        female_color = '#6badd6'
    if mCA_type == '=':
        male_color = '#cb4e27'
        female_color = neutral_color

    mx1s = male_s_xyz[0]
    my1s = male_s_xyz[1]
    mz1s = male_s_xyz[2]

    mx1mu = male_mu_xyz[0]
    my1mu = male_mu_xyz[1]
    mz1mu = male_mu_xyz[2]

    fx1s = female_s_xyz[0]
    fy1s = female_s_xyz[1]
    fz1s = female_s_xyz[2]

    fx1mu = female_mu_xyz[0]
    fy1mu = female_mu_xyz[1]
    fz1mu = female_mu_xyz[2]

    male_lower95_s, male_upper95_s, male_s_ymax, male_s_probs_list = confidence_interval_95(mx1s, my1s, mz1s, male_color, ax1, 'male')
    male_lower95_mu, male_upper95_mu, male_mu_ymax, male_mu_probs_list = confidence_interval_95_mu(mx1mu, my1mu, mz1mu, male_color, ax2, 'male')

    female_lower95_s, female_upper95_s, female_s_ymax, female_s_probs_list = confidence_interval_95(fx1s, fy1s, fz1s, female_color, ax1, 'female')
    female_lower95_mu, female_upper95_mu, female_mu_ymax, female_mu_probs_list = confidence_interval_95_mu(fx1mu, fy1mu, fz1mu, female_color, ax2, 'female')

    ax1.set_ylim(0, (max([male_s_ymax, female_s_ymax])*1.1))
    ax2.set_ylim(0, (max([male_mu_ymax, female_mu_ymax])*1.1))

    ax2.set_xscale('log') #for mu
    ax1.legend(frameon=False, fontsize = 14)
    ax2.legend(frameon=False, fontsize = 14)

    savename = labelname.replace(' ', '_')

    plt.tight_layout()
    plt.savefig('Figures/Confidence_intervals_sex_differences_mCAs/'+savename+'_s_mu_confidence_intervals_cumulative.pdf')
    # plt.show()

    return male_lower95_s, male_upper95_s, male_lower95_mu, male_upper95_mu, male_s_probs_list, male_mu_probs_list, female_lower95_s, female_upper95_s, female_lower95_mu, female_upper95_mu, female_s_probs_list, female_mu_probs_list

def error_bars(hist, normed_value, widths):

    errors={}
    n=0
    for i in list(hist):
        normalised_hist = i/(normed_value*widths)
        log_hist = np.log(normalised_hist)
        sqrt_hist = math.sqrt(i)
        if sqrt_hist == 1:
            upper_error = 1
            lower_error = 0.9
        if sqrt_hist !=1:
            upper_error = sqrt_hist
            lower_error = sqrt_hist
        normalised_upper_error = upper_error/(normed_value*widths)
        normalised_lower_error = lower_error/(normed_value*widths)
        errors[n]=(normalised_hist[0], normalised_upper_error[0], normalised_lower_error[0])
        n = n+1

    errors_corrected ={}
    for k, v in errors.items():
        binheight = v[0]
        log_binheight = np.log(v[0])
        upper_error = v[1]
        lower_error = v[2]
        log_upper_error = (np.log(upper_error+binheight))-log_binheight
        log_lower_error = log_binheight-(np.log(binheight-lower_error))
        errors_corrected[k] = (log_binheight, log_upper_error, log_lower_error)

    lower_err=[]
    upper_err=[]
    for k, v in errors_corrected.items():
        lower_error = v[2]
        upper_error = v[1]
        lower_err.append(lower_error)
        upper_err.append(upper_error)

    err = [tuple(lower_err),tuple(upper_err)]

    return err

def plot_data_non_cumulative(ax, mu, cell_fractions, total_people, color, gender): #can be log or logit cell fractions
    bin_size = 'doane'

    normed_value = total_people*mu #study_total*mu
    hist, bins = np.histogram(cell_fractions, bins=bin_size, range=(min(cell_fractions),max(cell_fractions)))
    widths = np.diff(bins)
    bin_centres = (bins[:-1] + bins[1:])/2
    hist = np.array(hist, dtype=float)
    normalised_hist = hist/(normed_value*widths)
    log_hist_for_plot = np.log(normalised_hist)

    errors = error_bars(hist, normed_value, widths)

    m_width = 1.5
    c_size = 3
    c_thick = 1.5
    e_width = 1.5

    ax.errorbar(bin_centres, log_hist_for_plot, yerr= errors, fmt = 'o', ecolor = color, \
     elinewidth = e_width, capsize = c_size, capthick = c_thick, markersize = 8, markeredgewidth = 0.5, \
     markeredgecolor = 'k', markerfacecolor = color, zorder=10, label = gender)

    return ax

def plot_data_cumulative(ax, mu, cell_fractions, total_people, color, gender): #can be log or logit cell fractions

    cell_fractions = sorted(cell_fractions, reverse = True)

    normed_value = total_people*mu

    cumulative_number = np.arange(np.size(cell_fractions))/normed_value
    log_cumulative_number = np.log(cumulative_number)

    ax.scatter(cell_fractions, log_cumulative_number, s = 75, color = color, alpha = 0.5, zorder = 10, label = gender)

    return ax


def plot_theory_line_logit(s, mu, ax, color, sex):

    x=np.linspace(scipy.special.logit(0.0001), scipy.special.logit(0.99999999999999), 10000)
    y=[Probtheory_logit(l, [s, mu]) for l in x]
    ax.plot(x, np.log(y), c = color, alpha = 0.75, lw = 3)

    if sex == 'male':
        ax.text(0.05, 0.1, 'male \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.15, 'male s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)
    if sex == 'female':
        ax.text(0.05, 0.25, 'female \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.3, 'female s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)

    return ax

def plot_theory_line_logit_cumulative(s, mu, ax, color, sex, integral_limit):
    x=np.linspace(scipy.special.logit(integral_limit), scipy.special.logit(0.0001), 1000)
    y=[Probtheory_logit(l, [s, mu]) for l in x]
    y_int = 1-(it.cumtrapz(y, x, initial = 0))
    ax.plot(x, np.log(y_int), c = color, alpha = 0.75, lw = 3)

    if sex == 'male':
        ax.text(0.05, 0.1, 'male \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.15, 'male s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)
    if sex == 'female':
        ax.text(0.05, 0.25, 'female \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.3, 'female s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)

    return ax

def plots_with_cumulative_MLE_logit(mCA, integral_limit, labelname, male_s, male_mu, female_s, female_mu, male_logit_cell_fractions, female_logit_cell_fractions):

    total_men = 229122
    total_women = 273383

    ###################################################
    fig, axes = plt.subplots(figsize=(14, 5), nrows=1, ncols=2)
    fig.subplots_adjust(hspace=0.3, wspace = 0.3)

    ax1 = axes[0]
    ax2 = axes[1]

    ## Plot the data ###
    m_width = 1.5
    c_size = 3
    c_thick = 1.5
    e_width = 1.5

    mCA_type = mCA[-1]
    if mCA_type == '+':
        male_color = '#650913'
        female_color = '#ef3d2c'
        integral_limit = 0.99999
    if mCA_type == '-':
        male_color = '#213468'
        female_color = '#6badd6'
        integral_limit = 0.67
    if mCA_type == '=':
        male_color = '#cb4e27'
        female_color = neutral_color
        integral_limit = 0.54

    if len(male_logit_cell_fractions)>=10:
        plot_data_non_cumulative(ax1, male_mu, male_logit_cell_fractions, total_men, male_color, 'male')
        plot_data_cumulative(ax2, male_mu, male_logit_cell_fractions, total_men, male_color, 'male')

    if len(female_logit_cell_fractions)>=10:
        plot_data_non_cumulative(ax1, female_mu, female_logit_cell_fractions, total_women, female_color, 'female')
        plot_data_cumulative(ax2, female_mu, female_logit_cell_fractions, total_women, female_color, 'female')

    x_major_ticks = [scipy.special.logit(0.0001),scipy.special.logit(0.0002),scipy.special.logit(0.0003),scipy.special.logit(0.0004),scipy.special.logit(0.0005),scipy.special.logit(0.0006),scipy.special.logit(0.0007),scipy.special.logit(0.0008), scipy.special.logit(0.0009),\
                     scipy.special.logit(0.001), scipy.special.logit(0.002),scipy.special.logit(0.003),scipy.special.logit(0.004),scipy.special.logit(0.005),scipy.special.logit(0.006),scipy.special.logit(0.007),scipy.special.logit(0.008),scipy.special.logit(0.009), \
                     scipy.special.logit(0.01),scipy.special.logit(0.02),scipy.special.logit(0.03),scipy.special.logit(0.04),scipy.special.logit(0.05),scipy.special.logit(0.06),scipy.special.logit(0.07),scipy.special.logit(0.08),scipy.special.logit(0.09),\
                     scipy.special.logit(0.1),scipy.special.logit(0.2),scipy.special.logit(0.3),scipy.special.logit(0.4),scipy.special.logit(0.5),scipy.special.logit(0.6),scipy.special.logit(0.7),scipy.special.logit(0.8),scipy.special.logit(0.9), scipy.special.logit(0.99), scipy.special.logit(0.999)]
    x_major_tick_labels = ["0.01","","","","","","","","",\
                           "0.1","","","","","","","","",\
                           "1","","","","","","","","",\
                           "10","","","","50","","","","","99","99.9"]

    extended_x_major_ticks = [scipy.special.logit(0.0001),scipy.special.logit(0.0002),scipy.special.logit(0.0003),
                     scipy.special.logit(0.0004),scipy.special.logit(0.0005),scipy.special.logit(0.0006),
                     scipy.special.logit(0.0007),scipy.special.logit(0.0008), scipy.special.logit(0.0009),
                     scipy.special.logit(0.001), scipy.special.logit(0.002),scipy.special.logit(0.003),
                     scipy.special.logit(0.004),scipy.special.logit(0.005),scipy.special.logit(0.006),
                     scipy.special.logit(0.007),scipy.special.logit(0.008),scipy.special.logit(0.009),
                     scipy.special.logit(0.01),scipy.special.logit(0.02),scipy.special.logit(0.03),
                     scipy.special.logit(0.04),scipy.special.logit(0.05),scipy.special.logit(0.06),
                     scipy.special.logit(0.07),scipy.special.logit(0.08),scipy.special.logit(0.09),\
                     scipy.special.logit(0.1),scipy.special.logit(0.2),scipy.special.logit(0.3),
                     scipy.special.logit(0.4),scipy.special.logit(0.5),scipy.special.logit(0.6),
                     scipy.special.logit(0.7),scipy.special.logit(0.8),scipy.special.logit(0.9),
                     scipy.special.logit(0.99), scipy.special.logit(0.999), scipy.special.logit(0.9999), scipy.special.logit(0.99999),
                    scipy.special.logit(0.999999), scipy.special.logit(0.9999999),
                    scipy.special.logit(0.99999999), scipy.special.logit(0.999999999),
                    scipy.special.logit(0.999999999), scipy.special.logit(0.99999999999),
                    scipy.special.logit(0.99999999999), scipy.special.logit(0.9999999999999)]
    extended_x_major_tick_labels = ["0.01","","","","","","","","",\
                           "0.1","","","","","","","","",\
                           "1","","","","","","","","",\
                           "10","","","","50","","","","","99","", "", "",
                           "", "", "", "",
                          "", "",
                          "", "99.99999999999"]

    y_major_ticks = [np.log(1), np.log(2), np.log(3), \
                     np.log(4), np.log(5), np.log(6), \
                     np.log(7), np.log(8), np.log(9), \
                     np.log(10), np.log(20), np.log(30),\
                     np.log(40), np.log(50), np.log(60), \
                     np.log(70), np.log(80), np.log(90),\
                     np.log(100), np.log(200), np.log(300), \
                     np.log(400), np.log(500), np.log(600),\
                     np.log(700), np.log(800), np.log(900), \
                     np.log(1000), np.log(2000), np.log(3000),\
                    np.log(4000), np.log(5000), np.log(6000), \
                     np.log(7000), np.log(8000), np.log(9000),\
                    np.log(10000), np.log(20000), np.log(30000), \
                     np.log(40000), np.log(50000), np.log(60000),\
                    np.log(70000), np.log(80000), np.log(90000), \
                     np.log(100000), np.log(200000),np.log(300000),np.log(400000),np.log(500000),np.log(600000),\
                    np.log(700000),np.log(800000),np.log(900000),np.log(1000000)]
    y_major_tick_labels = ["","","", "", "", "", "", "", "", "$10^{1}$","", "", "", "", "", "", "", "", \
                           "$10^{2}$","", "", "", "", "", "", "", "", "$10^{3}$","", "", "", "", "", "", "", "", \
                           "$10^{4}$","", "", "", "", "", "", "", "", "$10^{5}$","", "", "", "", "", "", "", "", "$10^{6}$"]


    ax1.set_ylabel('relative density of mCAs')
    ax2.set_ylabel('relative density of mCAs (cumulative)')

    ######## plot the theory line ########
    plot_theory_line_logit(male_s, male_mu, ax1, male_color, 'male')
    plot_theory_line_logit(female_s, female_mu, ax1, female_color, 'female')

    ######## plot the cumulative theory line ########
    plot_theory_line_logit_cumulative(male_s, male_mu, ax2, male_color, 'male', integral_limit)
    plot_theory_line_logit_cumulative(female_s, female_mu, ax2, female_color, 'female', integral_limit)

    for ax in axes.flatten():
        ax.set_xlim(0.0001, 1.0)
        ax.set_xlabel('fraction of cells (%)')
        ax.set_xticks(x_major_ticks)
        ax.set_xticklabels(x_major_tick_labels)
        ax.set_yticks(y_major_ticks)
        ax.set_yticklabels(y_major_tick_labels)
        ax.tick_params(axis = 'both', which = 'major', color = grey4)
        ax.tick_params(axis = 'both', which='minor', bottom=False)
        ax.set_ylim(1, np.log(2*10**6))
        ax.set_title(labelname+'\n (male = '+str(len(male_logit_cell_fractions))+', female = '+str(len(female_logit_cell_fractions))+')')

    if (male_s>0.4) or (female_s>0.4):
        ax1.set_xticks(extended_x_major_ticks)
        ax1.set_xticklabels(extended_x_major_tick_labels)

    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')

    plt.tight_layout(w_pad = 5)
    savename = labelname.replace(' ', '_')
    plt.savefig('Figures/Cell_fraction_density_histograms_sex_differences_mCAs/'+str(savename)+'_cell_fraction_density_histogram_cumulative_logit.pdf')

    return print('histogram plotted')


def plot_theory_line_log(s, mu, ax, color, sex):

    x=np.linspace(-10, np.log(0.9999), 100000)
    y=[Probtheory(l, [s, mu]) for l in x]
    ax.plot(x, np.log(y), c = color, alpha = 0.75, lw = 3)

    x=np.linspace(np.log(0.9999), np.log(0.999999), 1000)
    y=[Probtheory(l, [s, mu]) for l in x]
    ax.plot(x, np.log(y), c = color, alpha = 0.75, lw = 3)

    if sex == 'male':
        ax.text(0.05, 0.1, 'male \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.15, 'male s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)
    if sex == 'female':
        ax.text(0.05, 0.25, 'female \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.3, 'female s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)

    return ax

def plot_theory_line_log_cumulative(s, mu, ax, color, sex, integral_limit):
    x=np.linspace(np.log(integral_limit), -10, 1000)
    y=[Probtheory(l, [s, mu]) for l in x]
    y_int = 1-(it.cumtrapz(y, x, initial = 0))
    ax.plot(x, np.log(y_int), c = color, alpha = 0.75, lw = 3)

    if sex == 'male':
        ax.text(0.05, 0.1, 'male \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.15, 'male s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)
    if sex == 'female':
        ax.text(0.05, 0.25, 'female \u03BC = '+str(format(mu, '.2e')), transform=ax.transAxes, fontsize = 12)
        ax.text(0.05, 0.3, 'female s = '+str(round(s*100, 1))+'%', transform=ax.transAxes, fontsize = 12)

    return ax

def plots_with_cumulative_MLE(mCA, integral_limit, labelname, male_s, male_mu, female_s, female_mu, male_log_cell_fractions, female_log_cell_fractions):

    total_men = 229122
    total_women = 273383

    ###################################################
    fig, axes = plt.subplots(figsize=(13.25, 5), nrows=1, ncols=2)
    fig.subplots_adjust(hspace=0.3, wspace = 0.3)

    ax1 = axes[0]
    ax2 = axes[1]

    ## Plot the data ###
    m_width = 1.5
    c_size = 3
    c_thick = 1.5
    e_width = 1.5

    mCA_type = mCA[-1]
    if mCA_type == '+':
        male_color = '#650913'
        female_color = '#ef3d2c'
        integral_limit = 0.99999
    if mCA_type == '-':
        male_color = '#213468'
        female_color = '#6badd6'
        integral_limit = 0.67
    if mCA_type == '=':
        male_color = '#cb4e27'
        female_color = neutral_color
        integral_limit = 0.54

    if len(male_log_cell_fractions)>=10:
        plot_data_non_cumulative(ax1, male_mu, male_log_cell_fractions, total_men, male_color, 'male')
        plot_data_cumulative(ax2, male_mu, male_log_cell_fractions, total_men, male_color, 'male')

    if len(female_log_cell_fractions)>=10:
        plot_data_non_cumulative(ax1, female_mu, female_log_cell_fractions, total_women, female_color, 'female')
        plot_data_cumulative(ax2, female_mu, female_log_cell_fractions, total_women, female_color, 'female')

    x_major_ticks = [np.log(0.0001),np.log(0.0002),np.log(0.0003),np.log(0.0004),np.log(0.0005),np.log(0.0006),np.log(0.0007),np.log(0.0008), np.log(0.0009),\
                     np.log(0.001), np.log(0.002),np.log(0.003),np.log(0.004),np.log(0.005),np.log(0.006),np.log(0.007),np.log(0.008),np.log(0.009), \
                     np.log(0.01),np.log(0.02),np.log(0.03),np.log(0.04),np.log(0.05),np.log(0.06),np.log(0.07),np.log(0.08),np.log(0.09),\
                     np.log(0.1),np.log(0.2),np.log(0.3),np.log(0.4),np.log(0.5),np.log(0.6),np.log(0.7),np.log(0.8),np.log(0.9), np.log(1.0)]
    x_major_tick_labels = ["0.01","","","","","","","","",\
                           "0.1","","","","","","","","",\
                           "1","","","","","","","","",\
                           "10","","","","50","","","","","100"]

    y_major_ticks = [np.log(1), np.log(2), np.log(3), \
                     np.log(4), np.log(5), np.log(6), \
                     np.log(7), np.log(8), np.log(9), \
                     np.log(10), np.log(20), np.log(30),\
                     np.log(40), np.log(50), np.log(60), \
                     np.log(70), np.log(80), np.log(90),\
                     np.log(100), np.log(200), np.log(300), \
                     np.log(400), np.log(500), np.log(600),\
                     np.log(700), np.log(800), np.log(900), \
                     np.log(1000), np.log(2000), np.log(3000),\
                    np.log(4000), np.log(5000), np.log(6000), \
                     np.log(7000), np.log(8000), np.log(9000),\
                    np.log(10000), np.log(20000), np.log(30000), \
                     np.log(40000), np.log(50000), np.log(60000),\
                    np.log(70000), np.log(80000), np.log(90000), \
                     np.log(100000), np.log(200000),np.log(300000),np.log(400000),np.log(500000),np.log(600000),\
                    np.log(700000),np.log(800000),np.log(900000),np.log(1000000)]
    y_major_tick_labels = ["","","", "", "", "", "", "", "", "$10^{1}$","", "", "", "", "", "", "", "", \
                           "$10^{2}$","", "", "", "", "", "", "", "", "$10^{3}$","", "", "", "", "", "", "", "", \
                           "$10^{4}$","", "", "", "", "", "", "", "", "$10^{5}$","", "", "", "", "", "", "", "", "$10^{6}$"]

    for ax in axes.flatten():
        ax.set_xlabel('fraction of cells (%)')
        ax.set_xticks(x_major_ticks)
        ax.set_xticklabels(x_major_tick_labels)
        ax.set_yticks(y_major_ticks)
        ax.set_yticklabels(y_major_tick_labels)
        ax.tick_params(axis = 'both', which = 'major', color = grey4)
        ax.tick_params(axis = 'both', which='minor', bottom=False)
        ax.set_ylim(1, np.log(2*10**6))
        ax.set_xlim(-9, 0)
        ax.set_title(labelname+'\n (male = '+str(len(male_log_cell_fractions))+', female = '+str(len(female_log_cell_fractions))+')')

    ax1.set_ylabel('relative density of mCAs')
    ax2.set_ylabel('relative density of mCAs (cumulative)')

    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')

    ######## plot the theory line ########
    plot_theory_line_log(male_s, male_mu, ax1, male_color, 'male')
    plot_theory_line_log(female_s, female_mu, ax1, female_color, 'female')

    ######## plot the cumulative theory line ########
    plot_theory_line_log_cumulative(male_s, male_mu, ax2, male_color, 'male', integral_limit)
    plot_theory_line_log_cumulative(female_s, female_mu, ax2, female_color, 'female', integral_limit)

    plt.tight_layout(w_pad = 5)

    savename = labelname.replace(' ', '_')

    plt.savefig('Figures/Cell_fraction_density_histograms_sex_differences_mCAs/'+str(savename)+'_cell_fraction_density_histogram_cumulative.pdf')

    return print('histogram plotted')


def g(higher_interpolated_distribution, x):
    return higher_interpolated_distribution(x) #female interpolated function

def h(lower_interpolated_distribution, x, D):
    return lower_interpolated_distribution(x-D) #male interpolated function in terms of y = x - delta (delta = x - y)

def distribution_of_difference(higher_interpolated_distribution, lower_interpolated_distribution, xnew, D):
    return integrate.quad(lambda x: g(higher_interpolated_distribution, x)*h(lower_interpolated_distribution, x, D), xnew[0], xnew[-1], limit=1000)[0] #integrate between lowest x-axis value and highest x-axis value

def p_value(higher_interpolated_distribution, lower_interpolated_distribution, xnew, lower_D): #where lower_D is the lower limit to integrate down to (i.e. p-value = area under the curve between lower D and 0)
    #calculate the full area under the curve:
    upper_full_area = max(xnew)-min(xnew)
    lower_full_area = -upper_full_area
    print('upper full area = ', upper_full_area)
    print('lower full area = ', lower_full_area)
    full_area = integrate.quad(lambda D: distribution_of_difference(higher_interpolated_distribution, lower_interpolated_distribution, xnew, D), lower_full_area, upper_full_area, limit=10000)[0] #integrate between -100 and 100
    under_zero_area = integrate.quad(lambda D: distribution_of_difference(higher_interpolated_distribution, lower_interpolated_distribution,xnew,  D), lower_D, 0, limit=10000)[0] #integrate between lower_D and 0
    print('full area = ', full_area)
    print('under zero area = ', under_zero_area)
    return under_zero_area/full_area

def plot_distributions_with_interpolation(female_x, female_y, female_inter, male_x, male_y, male_inter, s_or_mu, mCA, ax):
    min_x = min([min(female_x), min(male_x)]) #minimum x-axis value from the male and female distributions
    max_x = max([max(female_x), max(male_x)]) #maximum x-axis value from the male and female distributions
    xnew = np.linspace(min_x, max_x, num=100, endpoint=True) #full range of x-axis values for the male and female distributions

    mCA_type = mCA[-1]
    if mCA_type == '+':
        male_color = '#650913'
        female_color = '#ef3d2c'
    if mCA_type == '-':
        male_color = '#213468'
        female_color = '#6badd6'
    if mCA_type == '=':
        male_color = '#cb4e27'
        female_color = neutral_color

    ax.plot(female_x, female_y, color = female_color, lw = 2) #plot the female distribution
    ax.scatter(xnew, female_inter(xnew), color = female_color) #plot the interpolated distribution to check it is working ok
    ax.plot(male_x, male_y, color = male_color, lw = 2) #plot the male distribution
    ax.scatter(xnew, male_inter(xnew), color = male_color) #plot the interpolated distribution to check it is working ok
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    ax.set_xlabel(s_or_mu, fontsize = 14)

    custom_lines = [Line2D([0], [0], color=female_color, lw=4),
                    Line2D([0], [0], color=male_color, lw=4),
                    Line2D([0], [0], color=female_color, lw=0, marker = 'o'),
                    Line2D([0], [0], color=male_color, lw=0, marker = 'o')]
    ax.legend(custom_lines, ['female', 'male', 'female interpolated', 'male interpolated'], frameon = False, loc = 'upper right')

    ax.set_title(mCA+' '+s_or_mu, fontsize = 14)
    ax.xaxis.set_tick_params(width=1, color = grey3, length = 6, labelsize = 14, top = False, labeltop = False)
    ax.yaxis.set_tick_params(width=1, color = grey3, length = 6, labelsize = 14, top = False, labeltop = False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color(grey3)

    return ax

def plot_distribution_differences_with_95_CI(xnew, higher_interpolated_distribution, lower_interpolated_distribution, p_val, s_or_mu, mCA, ax):
    upper_full_area = max(xnew)-min(xnew) #upper limit for integration under the curve = difference between highest and lowest points of the 2 distributions
    lower_full_area = -upper_full_area #lower limit for integration under the curve
    xx=np.array([D for D in np.linspace(lower_full_area, upper_full_area, 100)])
    yy=[distribution_of_difference(higher_interpolated_distribution, lower_interpolated_distribution, xnew, D) for D in xx]

    ax.fill_between(xx, yy, where= xx<=0, zorder = 0, facecolor = 'white', hatch = '///')
    ax.plot(xx, yy, color = grey4, lw = 2, zorder = 1)
    ax.plot([0, 0], [0, max(yy)], color = grey4, lw = 2, zorder = 70)

    #Calculate 95% CI of the distribution of difference
    total_yy = 0
    for entry in yy:
        total_yy = total_yy+entry
    normalised_y_array = np.array(yy)/total_yy

    cumulative_y = 0
    x_95_range = []
    for x, norm_y in zip(xx, normalised_y_array):
        cumulative_y=cumulative_y+norm_y
        if 0.025<cumulative_y<0.975:
            x_95_range.append(x)

    min_x_95=min(x_95_range)
    max_x_95=max(x_95_range)

    ax.plot([min_x_95, min_x_95], [0, max(yy)], color = purple3, lw = 2, zorder = 70)
    ax.plot([max_x_95, max_x_95], [0, max(yy)], color = purple3, lw = 2, zorder = 70)

    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(0, max(yy))
    ax.set_xlabel(s_or_mu+' difference for '+mCA, fontsize = 14)
    ax.text(0, 0.9, 'p-value = '+str("{:.2e}".format(p_val)), horizontalalignment='left', transform=ax.transAxes, fontsize = 12)

    ax.xaxis.set_tick_params(width=1, color = grey3, length = 6, labelsize = 14, top = False, labeltop = False)
    for axis in ['bottom']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color(grey3)

    return ax, min_x_95, max_x_95

def calculate_p_value(s_or_mu, mCA, male_s_or_mu, male_s_or_mu_list, male_probs_list, female_s_or_mu, female_s_or_mu_list, female_probs_list, labelname):

    #Step 1 = generate the functions for the female and male distributions
    female_inter = interp1d(female_s_or_mu_list, female_probs_list, fill_value=0, bounds_error=False) #interpolate the female distribution
    male_inter = interp1d(male_s_or_mu_list, male_probs_list, fill_value=0, bounds_error=False) #interpolate the female distribution

    #Step 2 = Determine which distribution is the one with the suspected higher s or mu value:
    if female_s_or_mu >= male_s_or_mu:
        higher_interpolated_distribution = female_inter
        lower_interpolated_distribution = male_inter
    else:
        higher_interpolated_distribution = male_inter
        lower_interpolated_distribution = female_inter

    #Step 3 = Calculate the p-value
    min_x = min([min(female_s_or_mu_list), min(male_s_or_mu_list)]) #minimum x-axis value from the male and female distributions
    max_x = max([max(female_s_or_mu_list), max(male_s_or_mu_list)]) #maximum x-axis value from the male and female distributions
    xnew = np.linspace(min_x, max_x, num=100000, endpoint=True) #full range of x-axis values for the male and female distributions
    upper_full_area = max(xnew)-min(xnew) #upper limit for integration under the curve = difference between highest and lowest points of the 2 distributions
    lower_full_area = -upper_full_area #lower limit for integration under the curve
    p_val = p_value(higher_interpolated_distribution, lower_interpolated_distribution, xnew, lower_full_area)
    print('p-value = ', p_val)

    #Step 4 = plot the distributions and distribution of differences
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    ax1 = axes[0]
    ax2 = axes[1]
    plot_distributions_with_interpolation(female_s_or_mu_list, female_probs_list, female_inter, male_s_or_mu_list, male_probs_list, male_inter, s_or_mu, mCA, ax1)
    ax2, lower_CI_difference, upper_CI_difference = plot_distribution_differences_with_95_CI(xnew, higher_interpolated_distribution, lower_interpolated_distribution, p_val, s_or_mu, mCA, ax2)

    plt.tight_layout(w_pad = 5)

    savename = labelname.replace(' ', '_')

    plt.savefig('Figures/P_value_plots_sex_differences_mCAs/'+str(savename)+'_distribution_sex_differences_p_value_calculation_'+s_or_mu+'.pdf')

    return p_val, lower_CI_difference, upper_CI_difference

def main():
    # Parameters to be input.
    parser = ArgumentParser()
    parser.add_argument("--mCA", type=str, dest="mCA", help="e.g. 3+", required=True)
    o = parser.parse_args()

    mCA = o.mCA

    total_men = 229122
    total_women = 273383

    start_time = time.time()

    if mCA[-1]=='+':
        labelname = 'gain '+str(mCA[:-1])
        integral_limit = 0.999
    if mCA[-1]=='-':
        labelname = 'loss '+str(mCA[:-1])
        integral_limit = 0.67
    if mCA[-1]=='=':
        labelname = 'CNLOH '+str(mCA[:-1])
        integral_limit = 0.54

    #Step 1 = read in grid ranges
    df_grid = pd.read_csv('Data_files/Heatmap_grid_ranges_sex_differences_mCAs.csv')
    df_grid = df_grid.set_index('mCA')
    df_dict = pd.DataFrame.to_dict(df_grid, orient = 'index')

    grid_size = df_dict[mCA]['GRID']
    male_lower_s = float(df_dict[mCA]['male lower s'])
    male_upper_s = float(df_dict[mCA]['male upper s'])
    male_lower_mu = float(df_dict[mCA]['male lower mu'])
    male_upper_mu = float(df_dict[mCA]['male upper mu'])
    female_lower_s = float(df_dict[mCA]['female lower s'])
    female_upper_s = float(df_dict[mCA]['female upper s'])
    female_lower_mu = float(df_dict[mCA]['female lower mu'])
    female_upper_mu = float(df_dict[mCA]['female upper mu'])

    zf = zipfile.ZipFile('Data_files/Supplementary_data_annotated_cw.csv.zip')
    df = pd.read_csv(zf.open('Supplementary_data_annotated_cw.csv'))

    grid_size = 50

    #MEN
    male_s_range = [male_lower_s, male_upper_s]
    male_mu_range = [male_lower_mu, male_upper_mu]

    male_log_cell_fractions, male_logit_cell_fractions = log_logit_CFs_sex(df, mCA, 'male')
    male_cumulative_densities_list = cumulative_cell_fraction_densities_list(male_logit_cell_fractions, total_men)

    #Calculate s and mu using heatmap
    print('calculating s and mu (using heatmap) for male '+mCA)
    male_s_xyz, male_mu_xyz, male_s, male_mu = heatmap_logit_cumulative(male_cumulative_densities_list, male_s_range, male_mu_range, grid_size, mCA, labelname, integral_limit, 'male')

    #WOMEN
    female_s_range = [female_lower_s, female_upper_s]
    female_mu_range = [female_lower_mu, female_upper_mu]

    female_log_cell_fractions, female_logit_cell_fractions = log_logit_CFs_sex(df, mCA, 'female')
    female_cumulative_densities_list = cumulative_cell_fraction_densities_list(female_logit_cell_fractions, total_women)

    #Calculate s and mu using heatmap
    print('calculating s and mu (using heatmap) for female '+mCA)
    female_s_xyz, female_mu_xyz, female_s, female_mu = heatmap_logit_cumulative(female_cumulative_densities_list, female_s_range, female_mu_range, grid_size, mCA, labelname, integral_limit, 'female')

    #Calculate the confidence intervals and plot on same plot
    print('calculating and plotting 95% confidence intervals for '+mCA)
    male_lower95_s, male_upper95_s, male_lower95_mu, male_upper95_mu, male_s_probs_list, male_mu_probs_list, female_lower95_s, female_upper95_s, female_lower95_mu, female_upper95_mu, female_s_probs_list, female_mu_probs_list = confidence_intervals_plot(male_s_xyz, male_mu_xyz, female_s_xyz, female_mu_xyz, mCA, labelname)

    #Step 5 = plot the histograms
    print('plotting the histograms...')
    plots_with_cumulative_MLE_logit(mCA, integral_limit, labelname, male_s, male_mu, female_s, female_mu, male_logit_cell_fractions, female_logit_cell_fractions)
    plots_with_cumulative_MLE(mCA, integral_limit, labelname, male_s, male_mu, female_s, female_mu, male_log_cell_fractions, female_log_cell_fractions)

    #Step 6 = calculate the p-value difference
    p_val_s, lower_CI_difference_s, upper_CI_difference_s = calculate_p_value('s', mCA, male_s, male_s_probs_list[0], male_s_probs_list[1], female_s, female_s_probs_list[0], female_s_probs_list[1], labelname)
    p_val_mu, lower_CI_difference_mu, upper_CI_difference_mu = calculate_p_value('mu', mCA, male_mu, male_mu_probs_list[0], male_mu_probs_list[1], female_mu, female_mu_probs_list[0], female_mu_probs_list[1], labelname)

    #Step 6 = write the data to output file
    N_male = mCA_count_number_mCAs(mCA, df, 1, 'male')
    N_female = mCA_count_number_mCAs(mCA, df, 1, 'female')

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    mCA_results = open('Data_files/mCA_s_mu_with_confidence_intervals_sex_differences.txt', 'a') #'a' means append to the file, rather than overwrite it
    mCA_results.write(dt_string+'\t'+mCA+'\t'+str(N_male)+'\t'+str(N_female)+'\t'+str(male_s)+'\t'+
                      str(male_lower95_s)+'\t'+str(male_upper95_s)+'\t'+str(female_s)+'\t'+str(female_lower95_s)+'\t'+
                      str(female_upper95_s)+'\t'+str(p_val_s)+'\t'+str(lower_CI_difference_s/100)+'\t'+str(upper_CI_difference_s/100)+'\t'+
                      str(male_mu)+'\t'+str(male_lower95_mu)+'\t'+str(male_upper95_mu)+'\t'+str(female_mu)+'\t'+
                      str(female_lower95_mu)+'\t'+str(female_upper95_mu)+'\t'+str(p_val_mu)+'\t'+
                      str(lower_CI_difference_mu)+'\t'+str(upper_CI_difference_mu)+'\t'+
                      str(male_s_probs_list)+'\t'+str(female_s_probs_list)+'\t'+str(male_mu_probs_list)+'\t'+str(female_mu_probs_list)+'\n')
    mCA_results.close()

    print('MLE completed in %s minutes' % round((time.time() - start_time)/60, 3))
    print('heatmap and confidence interval plots complete for sex differences in '+str(mCA)+' in '+str(int(time.time() - start_time)/3600)+' hours')

    # filename = 'Bell_sound.wav'
    # wave_obj = sa.WaveObject.from_wave_file(filename)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()  # Wait until sound has finished playing

    return

if __name__ == "__main__":
	main()
