from .peakSelector import peakSelector
from .translations import translation_calibration as transl
from .regresion import regresionPonderada

import os
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

import mca_tools as mca


def calibration(element_list):

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
    a, b, sa, sb, r = regresionPonderada(energies, channels, channels_uncertainty)

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

        plt.show()

    return new_a, new_b, new_sa, new_sb


def calibration_helper(folder_path, bkg_file):
    """
    The cached files are stored in .cache/peaks/file_name and .cache/energies/file_name
    """

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

        element = peakSelector(file_path, bkg_file = background_path)

        if not os.path.isfile(file_peaks_path):

            while True:
                element.select_peaks()


            user_input = input(trans["cache peak info"][mca.lang])
            if user_input.lower() != "n":
                element.save_peaks(file_peaks_path)

        else:
            element.load_peaks(file_peaks_path)


        # If there is no cached
        if not os.path.isfile(file_energies_path):

            gamma_energies = input("").split()

            # Turning them into floats
            for i in range(len(gamma_energies)):
                gamma_energies[i] = float(gamma_energies[i])

            # Setting them in the element.
            element.set_peak_energies(gamma_energies)

            print("")
            user_input = input(trans["cache energy info"][mca.lang])

            if user_input.lower() != "n":
                element.save_peak_energies(file_energies_path)

        else:
            element.load_peak_energies(file_energies_path)


        elements.append(element)



    calibration(elements)








