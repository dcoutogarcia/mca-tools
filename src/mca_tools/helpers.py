from .peakSelector import peakSelector
from .translations import translation_calibration as transl
from .uncertainty import print_uncertainty
from .operations import resolution, calibration
from .options import lang, style

import os
from shutil import rmtree
import matplotlib.pyplot as plt
import pandas as pd


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
        user_input = input(transl["remove cached peak info"][lang])

        if user_input.lower() == "y":
            rmtree(cache_path)

    # Try to create them again, even if they existed to ensure all of the
    # folders are present (maybe the user just removed one of them)
    os.makedirs(cache_path, exist_ok = True)
    os.makedirs(peaks_path, exist_ok = True)
    os.makedirs(energies_path, exist_ok = True)



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

    print(transl["important info"][lang])

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
            user_input = input(transl["cache peak info"][lang])
            if user_input.lower() != "n":
                element.save_peaks(file_peaks_path)

        else:
            element.load_peaks(file_peaks_path)


        # If there is no cached
        if not os.path.isfile(file_energies_path):

            gamma_energies = input(transl["write gamma energy"][lang]).split()

            # Turning them into floats
            for i in range(len(gamma_energies)):
                gamma_energies[i] = float(gamma_energies[i])

            # Setting them in the element.
            element.set_peak_energies(gamma_energies)

            print("")
            user_input = input(transl["cache energy info"][lang])

            if user_input.lower() != "n":
                element.save_peak_energies(file_energies_path)

        else:
            element.load_peak_energies(file_energies_path)


        elements.append(element)

    # We call the calibration and resolution functions
    calibration_fit = calibration(elements, return_figure = True)
    resolution_fit = resolution(elements, return_figure = True)


    # We start saving the results. In the plot, we need to use the style.
    calcs_str = ["calibration", "resolution"]
    calcs = [calibration_fit, resolution_fit]
    for i in range(len(calcs)):
        a, b, sa, sb, red_chi2, p_value, fig, ax = calcs[i]
        with plt.style.context(style):
            plt.style.use(style)
            plt.rcParams.update({
                'figure.dpi': '100',
                'font.size': 12.0
            })

            fig.savefig(os.path.join(output_path, "figs", (calcs_str[i] + fig_ext)))


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
        df.to_csv(os.path.join(output_path, "csv", calcs_str[i] + ".csv"), index = False)

    return calcs
