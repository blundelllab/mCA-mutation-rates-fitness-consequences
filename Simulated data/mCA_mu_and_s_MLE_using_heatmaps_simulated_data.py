#!/usr/bin/env python

'''''
Watson code for calculing fitness effect (s) and mutation rate (mu) for simulated mCAs with or without an upper limit in cell fraction.
Version 1.0 (Feb 2023)

Input:
    1) fitness
    2) mCA_type
    3) capped_or_not

Outputs:
    1) inferred s and mu
    2) 95% confidence intervals for s and mu
    3) heatmap and confidence interval plots
    4) cell fraction density histograms

Usage:
mCA_mu_and_s_MLE_using_heatmaps_simulated_data.py  --fitness --mCA_type --capped_or_not

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
import simpleaudio as sa

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

total_people = 502411


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

def logit_cell_fractions_CF(df, lower_limit, upper_limit): #multiply by 2 because the simulation generates VAFs rather than cell fractions
    df_copy = df.copy(deep = True)
    VAFs_mCA = df_copy['true_freq'].to_list()
    VAFs = []
    for i in VAFs_mCA:
        if lower_limit <= (i) <= upper_limit:
            VAFs.append(scipy.special.logit(i)) #logit
#             VAFs.append(np.log(float(i*2)/(1-float(i*2)))) #logit
    print('total mCAs = '+str(len(VAFs)))
    return VAFs

def cumulative_cell_fraction_densities_list(logit_cell_fractions, total_people):

    logit_cell_fractions = sorted(logit_cell_fractions, reverse = True)

    cumulative_number = np.arange(np.size(logit_cell_fractions))/total_people
    log_cumulative_number = np.log(cumulative_number)

    densities = []
    for a, b in zip(logit_cell_fractions, log_cumulative_number):
        if math.isinf(b) == False:
            densities.append((a, b))

    return densities

def create_shortened_datapoint_list(logit_cell_fractions):
    sorted_logit_cell_fractions = sorted(logit_cell_fractions)

    hist, bins = np.histogram(logit_cell_fractions, bins='doane', range=(min(logit_cell_fractions),max(logit_cell_fractions)))

    bins_ranges = []
    for i in range(0, len(bins)-1):
        bins_ranges.append((bins[i], bins[i+1]))

    bins_keys = {}
    for cf in sorted_logit_cell_fractions:
        for bins in bins_ranges:
            if bins[0]<=cf<bins[1]:
                if bins in bins_keys.keys():
                    bins_keys[bins].append(cf)
                else:
                    bins_keys[bins]=[cf]
    bins_keys

    shortened_logit_list = []
    for k, v in bins_keys.items(): #take the 1st and middle logit cell fraction from each bin
        shortened_logit_list.append(v[0])
        list_length = len(v)
#         if list_length>=2:
#             shortened_logit_list.append(v[int(len(v)/2)])

    if list_length>1:
        shortened_logit_list.append(sorted_logit_cell_fractions[-1]) #make sure the final datapoint is in the list (i.e. both first and last will be there)

    return shortened_logit_list

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

def ProbDataGivenModel_logit_cumulative_subsample(params, data, integral_limit):
    "This returns the likelihood of the variant being at that cell fraction, given the model"

    x = []
    for datapoint in data:
        logit_cell_fraction = datapoint[0]
        if scipy.special.expit(logit_cell_fraction) <1.0:
            x.append(logit_cell_fraction)

    #just select some of the datapoints to calculate the cumulative for...
    sampled_logit_cell_fractions = create_shortened_datapoint_list(x)

    predicted_cumulative_subsampled = [Probtheory_logit_cumulative(integral_limit, l, params) for l in sampled_logit_cell_fractions]

    predicted_inter = interp1d(sampled_logit_cell_fractions, predicted_cumulative_subsampled, fill_value = 'extrapolate')

    interpolated_predicted_cumulative = predicted_inter(x)

    x_y_dict = {}
    for a, b in zip(x, interpolated_predicted_cumulative):
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

def heatmap_logit_cumulative_subsample(cumulative_densities_list, s_range, mu_range, grid_size, mCA, labelname, integral_limit):
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
    logProbs = np.array([[ProbDataGivenModel_logit_cumulative_subsample([s, mu], cumulative_densities_list, integral_limit) for s in s_list] for mu in mu_list])

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
    logProbs = np.array([[ProbDataGivenModel_logit_cumulative_subsample([s, mu], cumulative_densities_list, integral_limit) for mu in mu_list] for s in s_list])

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
    # theta_phi_max = (x1[xmax, ymax], y1[xmax, ymax], z1.max())
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

    ax1.set_title(labelname, fontsize = 16, y = 1.01)
    ax2.set_title(labelname, fontsize = 16, y = 1.01)

    savename = labelname.replace(' ', '_')

    plt.tight_layout()
    plt.savefig('Logit_heatmaps_cumulative/'+savename+'_s_mu_heatmap.pdf')
    # plt.show()

    return [x1, y1, z1], [x1_mu, y1_mu, z1_mu], s_max, mu_max

def confidence_interval_95(x1, y1, z1, mCA, ax): #95% confidence interval for s

    axisfont=17
    titlefont=20
    axislabelfont=21
    m_size=8
    scale = 1

    if mCA[-1]=='+':
        mCA_color = gain_color
    if mCA[-1]=='-':
        mCA_color = loss_color
    if mCA[-1]=='=':
        mCA_color = neutral_color


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

    ax.plot(s_list, probs_list, color = mCA_color, lw = 2)
    ax.fill_between(s_listCI, probs_listCI, color = mCA_color, alpha = 0.2)

    #plot confidence interval
    ax.plot([lower95_s*100, lower95_s*100], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([upper95_s*100, upper95_s*100], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([s_mle*100, s_mle*100], [0, 1], linestyle = ':', color = mCA_color, lw = 2)

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

def confidence_interval_95_mu(x1, y1, z1, mCA, ax): #95% confidence interval for muation rate increase

    axisfont=17
    titlefont=20
    axislabelfont=21
    m_size=8
    scale = 1

    if mCA[-1]=='+':
        mCA_color = gain_color
    if mCA[-1]=='-':
        mCA_color = loss_color
    if mCA[-1]=='=':
        mCA_color = neutral_color


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

    ax.plot(mu_list, probs_list, color = mCA_color, lw = 2)
    ax.fill_between(mu_listCI, probs_listCI, color = mCA_color, alpha = 0.2)

    #plot confidence interval
    ax.plot([lower95_mu, lower95_mu], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([upper95_mu, upper95_mu], [0, 1], linestyle = ':', color = grey4, lw = 2)
    ax.plot([mu_mle, mu_mle], [0, 1], linestyle = ':', color = mCA_color, lw = 2)

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

def confidence_intervals_plot(s_xyz, mu_xyz, mCA, labelname):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex = True, figsize=(16, 6))
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    gs.update(wspace=0.4)

    x1s = s_xyz[0]
    y1s = s_xyz[1]
    z1s = s_xyz[2]

    x1mu = mu_xyz[0]
    y1mu = mu_xyz[1]
    z1mu = mu_xyz[2]

    lower95_s, upper95_s, s_ymax, s_probs_list = confidence_interval_95(x1s, y1s, z1s, mCA, ax1)
    lower95_mu, upper95_mu, mu_ymax, mu_probs_list = confidence_interval_95_mu(x1mu, y1mu, z1mu, mCA, ax2)

    ax1.set_ylim(0, (s_ymax)*1.1)
    ax2.set_ylim(0, (mu_ymax)*1.1)

    ax2.set_xscale('log') #for mu

    ax1.legend(frameon=False, fontsize = 14)
    ax2.legend(frameon=False, fontsize = 14)

    savename = labelname.replace(' ', '_')

    plt.tight_layout()
    plt.savefig('Logit_confidence_intervals_cumulative/'+savename+'_s_mu_confidence_intervals_cumulative.pdf')
    # plt.show()

    return lower95_s, upper95_s, lower95_mu, upper95_mu, s_probs_list, mu_probs_list

def plot_data_non_cumulative(ax, mu, logit_cell_fractions, total_people, color, label):
    bin_size = 'doane'

    normed_value = total_people*mu #study_total*mu
    hist, bins = np.histogram(logit_cell_fractions, bins=bin_size, range=(min(logit_cell_fractions),max(logit_cell_fractions)))
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
     markeredgecolor = 'k', markerfacecolor = color, zorder=0, label = label)

    return ax

def plot_data_cumulative(ax, mu, logit_cell_fractions, total_people, color, label):

    logit_cell_fractions = sorted(logit_cell_fractions, reverse = True)

    normed_value = total_people*mu

    cumulative_number = np.arange(np.size(logit_cell_fractions))/normed_value
    log_cumulative_number = np.log(cumulative_number)

    ax.scatter(logit_cell_fractions, log_cumulative_number, s = 75, color = color, alpha = 0.5, label = label)

    return ax

def plot_data_cumulative_subsampled(ax, mu, logit_cell_fractions, total_people, color, label):

    reverse_sorted_logit_cell_fractions = sorted(logit_cell_fractions, reverse = True)

    normed_value = total_people*mu

    cumulative_number = np.arange(np.size(reverse_sorted_logit_cell_fractions))/normed_value
    log_cumulative_number = np.log(cumulative_number)

    subsample_logit_cell_fractions = create_shortened_datapoint_list(logit_cell_fractions)

    log_cumulative_number_subsample = []
    sample_logit_cell_fractions_to_plot = []
    for a, b in zip(reverse_sorted_logit_cell_fractions, log_cumulative_number):
        if a in subsample_logit_cell_fractions:
            sample_logit_cell_fractions_to_plot.append(a)
            log_cumulative_number_subsample.append(b)

    ax.scatter(sample_logit_cell_fractions_to_plot, log_cumulative_number_subsample, s = 75, facecolors='none', edgecolors=grey4, alpha = 0.5, label = label)

    return ax

def plots_with_cumulative_MLE_logit_subsample(mCA_type, integral_limit, labelname, s, mu, logit_cell_fractions, total_people):

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

    if mCA_type[-1] == '+':
        color = gain_color
    if mCA_type[-1] == '-':
        color = loss_color
    if mCA_type[-1] == '=':
        color = neutral_color

    plot_data_non_cumulative(ax1, mu, logit_cell_fractions, total_people, color, mCA_type)
    plot_data_cumulative(ax2, mu, logit_cell_fractions, total_people, color, mCA_type)
    plot_data_cumulative_subsampled(ax2, mu, logit_cell_fractions, total_people, color, mCA_type)

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
    x=np.linspace(scipy.special.logit(0.0001), scipy.special.logit(0.99999999999999), 1000)
    y=[Probtheory_logit(l, [s, mu]) for l in x]
    ax1.plot(x, np.log(y), c = color, alpha = 0.75, lw = 3)

    ax1.text(0.05, 0.1, '\u03BC = '+str(format(mu, '.2e')), transform=ax1.transAxes, fontsize = 12)
    ax1.text(0.05, 0.15, 's = '+str(round(s*100, 1))+'%', transform=ax1.transAxes, fontsize = 12)

    ######## plot the cumulative theory line ########
    x=np.linspace(scipy.special.logit(integral_limit), scipy.special.logit(0.0001), 1000)
    y=[Probtheory_logit(l, [s, mu]) for l in x]
    y_int = 1-(it.cumtrapz(y, x, initial = 0))
    ax2.plot(x, np.log(y_int), c = color, alpha = 0.75, lw = 3)

    ax2.text(0.05, 0.1, '\u03BC = '+str(format(mu, '.2e')), transform=ax2.transAxes, fontsize = 12)
    ax2.text(0.05, 0.15, 's = '+str(round(s*100, 1))+'%', transform=ax2.transAxes, fontsize = 12)

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
        ax.set_title(labelname)

    if s>0.4:
        ax1.set_xticks(extended_x_major_ticks)
        ax1.set_xticklabels(extended_x_major_tick_labels)

    plt.tight_layout(w_pad = 5)
    savename = labelname.replace(' ', '_')
    plt.savefig('Histograms_true_freq_logit_heatmaps_cumulative/'+str(savename)+'_cell_fraction_density_histogram_cumulative.pdf')

    return print('histogram plotted')

def main():
    # Parameters to be input.
    parser = ArgumentParser()
    parser.add_argument("--fitness", type=str, dest="fitness", help="e.g. '0.10'", required=True)
    parser.add_argument("--mCA_type", type=str, dest="mCA_type", help="'+'", required=True)
    parser.add_argument("--capped_or_not", type=str, dest="capped_or_not", help="'capped' or 'not capped'", required=True)
    o = parser.parse_args()

    fitness= o.fitness
    mCA_type = o.mCA_type
    capped_or_not = o.capped_or_not

    df_for_ranges = pd.read_csv('Logit_cumulative_heatmap_grid_limits.csv', sep = ',')
    df_for_ranges2 = df_for_ranges[(df_for_ranges['mCA type']==mCA_type) & (df_for_ranges['actual fitness']==float(fitness)) & (df_for_ranges['capped or not']== capped_or_not.replace('_', ' '))]

    lower_s = float(df_for_ranges2['lower s'].tolist()[0])
    upper_s = float(df_for_ranges2['upper s'].tolist()[0])
    lower_mu = float(df_for_ranges2['lower mu'].tolist()[0])
    upper_mu = float(df_for_ranges2['upper mu'].tolist()[0])
    s_range = (lower_s, upper_s)
    mu_range = (lower_mu, upper_mu)

    total_people = 502411

    integral_limit = 0.999

    if mCA_type == '+':
        lower_limit = 0.025
        mCA_type_written = 'gain'
    if mCA_type == '-':
        lower_limit = 0.041
        mCA_type_written = 'loss'
        if capped_or_not == 'capped':
            integral_limit = 0.67
    if mCA_type == '=':
        lower_limit = 0.015
        mCA_type_written = 'CNLOH'
        if capped_or_not == 'capped':
            integral_limit = 0.54

    start_time = time.time()

    df = pd.read_csv('Simulation_results/Simulated_mCAs/biobank_sim_test_u=4.35e-09_s_'+str(fitness)+'01_N100000_u_mCA=4.35e-09_seed3_biobank_variants.csv')

    logit_cell_fractions = logit_cell_fractions_CF(df, lower_limit, integral_limit)
    N = len(logit_cell_fractions)
    print('total number of mCAs = '+str(N))

    logit_cell_fractions_cumulative_density_list = cumulative_cell_fraction_densities_list(logit_cell_fractions, total_people)

    #Calculate s and mu using heatmap
    print('calculating s and mu (using heatmap)...')
    grid_size = 50
    labelname = mCA_type_written+' , upper limit = '+str(integral_limit)+' (s = '+fitness+') (N = '+str(N)+')'
    s_xyz, mu_xyz, s, mu = heatmap_logit_cumulative_subsample(logit_cell_fractions_cumulative_density_list, s_range, mu_range, grid_size, mCA_type, labelname, integral_limit)

    lower95_s, upper95_s, lower95_mu, upper95_mu, s_probs_list, mu_probs_list = confidence_intervals_plot(s_xyz, mu_xyz, mCA_type, labelname)

    print('MLE completed in %s minutes' % round((time.time() - start_time)/60, 3))
    print('s = ', s)
    print('mu = ', mu)

    print('plotting the histograms...')

    plots_with_cumulative_MLE_logit_subsample(mCA_type, integral_limit, labelname, s, mu, logit_cell_fractions, total_people)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    fitness_results = open('Data_files/Fitness_inferences_logit_heatmap_cumulative_with_probs_lists.txt', 'a') #'a' means append to the file, rather than overwrite it
    fitness_results.write(dt_string+'\t'+mCA_type+'\t'+str(fitness)+'\t'+str(lower_limit)+'\t'+str(integral_limit)+'\t'+str(N)+'\t'+str(s)+'\t'+str(lower95_s)+'\t'+str(upper95_s)+'\t'+str(mu)+'\t'+str(lower95_mu)+'\t'+str(upper95_mu)+'\t'+str(s_probs_list)+'\t'+str(mu_probs_list)+'\n')
    fitness_results.close()

    # filename = 'Bell_sound.wav'
    # wave_obj = sa.WaveObject.from_wave_file(filename)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()  # Wait until sound has finished playing

    print('s and mu inference complete for '+str(mCA_type)+' mCA of fitness '+str(fitness)+' ('+str(capped_or_not)+') in'+str(fitness)+' '+str(int(time.time() - start_time)/3600)+' hours')

    return

if __name__ == "__main__":
	main()
