from .peakSelector import peakSelector
from .translations import translation_calibration as transl
from .uncertainty import get_pvalue, print_uncertainty
from .options import lang, style


from numpy.linalg import det

import os
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_regression(x,y,s):
    """
    Fits data into a linear equation  y = a + bx, taking into account
    the uncertainties. It uses chi2 minimization.
    """
    w=1.0/(s*s)
    wy=sum(w*y); wx=sum(w*x);
    wxx=sum(w*x*x); wxy=sum(w*x*y); wyy=sum(w*y*y)
    sw=sum(w)
    d=det(np.array([[sw, wx],[wx, wxx]]))
    a=(wy*wxx-wx*wxy)/d
    b=(sw*wxy-wx*wy)/d
    sa=np.sqrt(wxx/d); sb=np.sqrt(sw/d)
    # r=(sw*wxy-wx*wy)/sqrt((sw*wxx-wx**2)*(sw*wyy-wy**2))

    chi2 = 0
    for xi, yi, si in zip(x,y,s):
        chi2 += (((a + b * xi) - yi) / si) **2

    red_chi2, p_value = get_pvalue(chi2, len(y) - 2)

    return [a, b, sa, sb, red_chi2, p_value]


def calibration(element_list, **kwargs):

    # By default, returning the figure is false
    return_figure = False
    # Check if we want to return matplotlib figure
    for k, val in kwargs.items():
        if k == "return_figure" and val:
            return_figure = True

    # All the elements must have a peak_energy and a peak_channels

    for element in element_list:
        if element.peak_energies is None or element.peak_channels is None:
            print(transl["not all elements have channels and energy"][lang])
            return

    energies = np.array([])
    channels = np.array([])
    channels_uncertainty = np.array([])
    for element in element_list:
        channels = np.append(channels, element.peak_channels)
        channels_uncertainty = np.append(channels_uncertainty, element.peak_channels_uncertainty)
        energies = np.append(energies, element.peak_energies)

    # To add the channel uncertainty, we need to use the y axis for them
    a, b, sa, sb, red_chi2, p_value = linear_regression(energies, channels, channels_uncertainty)

    # We now need to invert the calibration line, because we want to get energy from channels,
    # not backwards. For this, we will need to propagate uncertainties.
    new_a = - a / b
    new_b = 1 / b
    new_sa = np.sqrt(sa ** 2 / b ** 2 + (a / b ** 2) ** 2 * sb ** 2 )
    new_sb = abs(sb / b)

    # Now we plot the new value
    x = np.linspace(min(channels), max(channels), 100)
    y = new_a + new_b * x

    with plt.style.context(style):
        plt.style.use(style)
        plt.rcParams.update({
            'figure.dpi': '100',
            'figure.figsize': [7, 6],
            'figure.constrained_layout.use': True,
            'font.size': 14.0
        })
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y)
        ax.errorbar(channels, energies, xerr = channels_uncertainty, fmt=".")
        ax.set_ylabel(transl["energy"][lang] + " [KeV]")
        ax.set_xlabel(transl["channel"][lang])



        plt.show()

    if return_figure:
        return new_a, new_b, new_sa, new_sb, red_chi2, p_value, fig, ax
    else:
        return new_a, new_b, new_sa, new_sb, red_chi2, p_value



def resolution(element_list, **kwargs):

    # By default, returning the figure is false
    return_figure = False
    # Check if we want to return matplotlib figure
    for k, val in kwargs.items():
        if k == "return_figure" and val:
            return_figure = True

    # All the elements must have a peak_energy and a peak_channels_sigma

    for element in element_list:
        if element.peak_energies is None or element.peak_sigmas is None:
            print(transl["not all elements have channels and energy"][lang])
            return

    energies = np.array([])
    sigmas = np.array([])
    sigmas_uncertainties = np.array([])
    for element in element_list:
        sigmas = np.append(sigmas, element.peak_sigmas)
        sigmas_uncertainties = np.append(sigmas_uncertainties, element.peak_sigmas_uncertainty)
        energies = np.append(energies, element.peak_energies)

    # We need the calibration slope to calculate energies of the sigmas
    # Sigmas are diferences of energy, we only use the slope of the calibration func
    calibration_results = calibration(element_list)

    R = 2.35 * sigmas * calibration_results[1] / energies
    sR = 2.35 / energies * np.sqrt((sigmas_uncertainties * calibration_results[1])**2 + (sigmas * calibration_results[3]) **2)


    # We take logarithms to convert the relation to a linear one an fit it
    a, b, sa, sb, red_chi2, p_value = linear_regression(np.log(energies), np.log(R), abs(sR/R))

    # Now we plot the new value
    x = np.linspace(min(np.log(energies)), max(np.log(energies)), 100)
    y = a + b * x

    with plt.style.context(style):
        plt.style.use(style)
        plt.rcParams.update({
            'figure.dpi': '100',
            'figure.figsize': [7, 6],
            'figure.constrained_layout.use': True,
            'font.size': 14.0
        })
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y)
        ax.errorbar(np.log(energies), np.log(R), yerr = abs(sR/R), fmt=".")
        ax.set_ylabel(transl["log(resolution)"][lang])
        ax.set_xlabel(transl["log(energy)"][lang])

        plt.show()

    if return_figure:
        return a, b, sa, sb, red_chi2, p_value, fig, ax
    else:
        return a, b, sa, sb, red_chi2, p_value








