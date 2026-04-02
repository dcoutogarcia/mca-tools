from .peakSelector import peakSelector
from .translations import translation_calibration as transl
from .uncertainty import get_pvalue, print_uncertainty

from numpy.linalg import det

import os
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mca_tools as mca


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

    print(red_chi2, p_value)
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
            print(transl["not all elements have channels and energy"][mca.lang])
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

    with plt.style.context(mca.style):
        plt.style.use(mca.style)
        plt.rcParams.update({
            'figure.dpi': '100',
            'font.size': 12.0
        })
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y)
        ax.errorbar(channels, energies, xerr = channels_uncertainty, fmt=".")
        ax.set_ylabel(transl["energy"][mca.lang])
        ax.set_xlabel(transl["channel"][mca.lang])
        fig.suptitle(transl["calibration"][mca.lang])



        plt.show()

    if return_figure:
        return new_a, new_b, new_sa, new_sb, red_chi2, p_value, fig, ax
    else:
        return new_a, new_b, new_sa, new_sb, red_chi2, p_value


def calibration_helper(folder_path, bkg_file = None, **kwargs):
    """
    The cached files are stored in .cache/peaks/file_name and .cache/energies/file_name
    """

    # By default, the file extension will be pdf
    fig_ext = ".pdf"
    # Check if we want another plot file extension
    for k, val in kwargs.items():
        if k == "fig_ext":
            fig_ext = val

    # Firstly, we create folders to save the data

    output_path = os.path.join(folder_path, "output/")
    cache_path = os.path.join(folder_path, ".cache/")
    # Cache paths to store peak limits and gamma energies, respectively
    peaks_path = os.path.join(cache_path, "peaks/")
    energies_path = os.path.join(cache_path, "energies/")

    # If two absolute paths are specified in join, only the
    # second one is used.
    background_path = os.path.join(folder_path, bkg_file)

    os.makedirs(output_path, exist_ok=True)
    try:
        os.makedirs(cache_path)
        os.makedirs(peaks_path)
        os.makedirs(energies_path)

    except FileExistsError:
        # Ask the user if it wants to remove cached info
        user_input = input(transl["remove cached peak info"][mca.lang])

        if user_input.lower() == "y":
            rmtree(cache_path)
            os.makedirs(cache_path)
            os.makedirs(peaks_path)
            os.makedirs(energies_path)


    # We save the file names, ignoring them if they are directories
    list_dir = os.listdir(folder_path)
    files = []
    for file in list_dir:
        if os.path.isfile(os.path.join(folder_path, file)):
            files.append(file)

    # We remove the background from the list of processed elements
    # if it is in the same folder, as the others
    if bkg_file in files:
        files.remove(bkg_file)

    print(transl["important info"][mca.lang])

    elements = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_peaks_path = os.path.join(peaks_path, file)
        file_energies_path = os.path.join(energies_path, file)
        file_output_path = os.path.join(output_path, file)

        element = peakSelector(file_path, bkg_file = background_path, fig_path = os.path.join(output_path, "figs"), csv_path = os.path.join(output_path, "csv"), fig_ext = fig_ext)

        if not os.path.isfile(file_peaks_path):

            element.select_peaks()
            input()
            user_input = input(transl["cache peak info"][mca.lang])
            if user_input.lower() != "n":
                element.save_peaks(file_peaks_path)
                element.save_fit_info(file_output_path)

        else:
            element.load_peaks(file_peaks_path)


        # If there is no cached
        if not os.path.isfile(file_energies_path):

            gamma_energies = input(transl["write gamma energy"][mca.lang]).split()

            # Turning them into floats
            for i in range(len(gamma_energies)):
                gamma_energies[i] = float(gamma_energies[i])

            # Setting them in the element.
            element.set_peak_energies(gamma_energies)

            print("")
            user_input = input(transl["cache energy info"][mca.lang])

            if user_input.lower() != "n":
                element.save_peak_energies(file_energies_path)

        else:
            element.load_peak_energies(file_energies_path)


        elements.append(element)

    # We call the calibration function
    a, b, sa, sb, red_chi2, p_value, fig, ax = calibration(elements,
                                                           return_figure = True)


    # We start saving the results. In the plot, we need to use the style.
    with plt.style.context(mca.style):
        plt.style.use(mca.style)
        plt.rcParams.update({
            'figure.dpi': '100',
            'font.size': 12.0
        })
        fig.savefig(os.path.join(output_path, "figs", ("calibration" + fig_ext)))

    # Output data for typst/latex in csv
    dic = {
        "a": print_uncertainty(a, sa),
        "b": print_uncertainty(b, sb),
        "$chi^2_r$": f"{red_chi2:.3f}",
    }

    p_value = p_value * 100 # in percentage
    if p_value  < 1e-3:
        formatted_p_value = (f"{p_value:.2e}")
    else:
        formatted_p_value = (f"{p_value:.2f}")

    dic.update({f"p-value(%)": formatted_p_value})

    df = pd.DataFrame(dic, index = [0])
    df.to_csv(os.path.join(output_path, "csv", "calibration.csv"), index = False)

    return a, b, sa, sb, red_chi2, p_value







