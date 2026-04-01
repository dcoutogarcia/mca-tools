import os
import pathlib

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import chi2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import matplotlib
import scienceplots

from .translations import translation_peakSelector as transl
from .uncertainty import round_uncertainty

import mca_tools as mca # lang as a global function


# Current Working Directory, used later to get relative paths
CWD = pathlib.Path().resolve()

matplotlib.use("qtagg")
# matplotlib.pyplot.set_loglevel("critical")

"""
Resets the values of the last peak inserted in the plot. It does so
by removing the elements of the sublist that contains the information
of the last peak.

It doesn't have a input because it's suposed to access a variable in
another scope (outside a function)

peak_positions' structure:
Peak positions is a list that contains lists with another list and a
string. The most nested list contains the start and end poins
of the peak. This is inside another list that contains also a string.
The string is the mode of the peak. If two peaks are too close,
they need to be adjusted together in curve_fit.

The main list contains the tuples for every peak in the graph.

Visual expanation:
[ peak1([start point, end point], "single") peak2([start point 1,
end point 1], "single")]
"""
# The code will be like this:
# list = []
# def func():
#   reset_peak_data(list)
#
# We cannot redefine the list inside the function (list = [],
# so we need to remove all the elements by hand



# Find nearest array element to value
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


class peakSelector:
    """
    Class for getting the curve_fit parameters of the Gaussian peaks
    in a spectrogram. Is asks for the user to select the peaks to be
    adjusted, by hand. After that, it computes the curve_fit and returns
    the optimal parameters.

    Input:
    file_path: the path to the mca file that contains spectroscopic info

    **kwargs:
        bins_fused: the amount of bins whose data is being "fused" in
                    the rebining. Setting it to 0 or False cancells
                    automatic rebining.

        bkg_file: the path to a mca file containing the data for the
                 background radiation. If specified, the background
                 will be substracted from the main data.

        fig_path: the path where figures will be automatically saved,
                  *relative* to the CWD. MUST end with a "/" (forward slash)
                  Default: "fig_path/"

        fig_ext: the file extension for the figures. Default: pdf.

    Methods:
    read_mca: reads the number of counts in each channel and the total
              time of the measurement.

    rebining: joins the data counts of a number of channels to reduce
              uncertainty on the parametric optimization. By default,
              it makes a 10 bin rebining. .rebining takes as an argument
              the number of bins fused.

    plot: represents the data in a static, non-interactive way. It's used
          to test if the data was loaded properly.


    select_peaks: lets the user select the desired peaks to be
                  analyzed by curve_fit. It works with matplotlib
                  spicker events.

    fit_peak: fits the peak to a gaussian function plus a polynomial
              background. In case of having a double peak, it adjusts to
              two gaussian funcions and the polynomial background.
              It accepts all scipy.optimize.curve_fit **kwargs.

              The output is the optimal parameters and the covariance
              matrix.

    """
    def __init__(self, file, **kwargs):
        # Basic variables
        self.file_path = file
        self.time = 0
        self.counts = [] # Number of counts per channel
        self.rates = [] # Counts / time
        self.xbins = [] # x values of channels
        self.delta_x = 0 # diference in x units between x distance in bins

        # User options
        self.bins_fused = 10 # Default number of bins fused in rebining
        self.fig_path = "fig_path/"
        self.fig_ext = "pdf"


        # Peak info
        self.peak_positions = [ [ [], None ], ]
        self.peak_energies = None
        self.peak_channels = None
        self.peak_channels_uncertainty = None


        # Background noise filtering
        self.bkg_file = None
        self.bkg_rates = None
        self.bkg_time = None

        # Modify default values with kwargs
        for k, val in kwargs.items():
            if k == "bins_fused":
                self.bins_fused = val
            elif k == "bkg_file":
                self.bkg_file = val
            elif k == "peak_file":
                self.load_peaks(val)
            elif k == "peak_energies":
                self.set_peak_energies(val)
            elif k == "fig_path":
                self.fig_path = val
            elif k == "fig_ext":
                self.fig_ext = val


        # Here we start the methods automaticaly, they can be used
        # anyway to modify default values.

        self.read_mca() # We read the values

        # Rebining can be disbled by usings kwargs. If not,
        # it will be performed.
        if self.bins_fused != 0 and self.bins_fused != False:
            self.rebining(self.bins_fused)


        # If a background file was specified as a kwarg, it will
        # execute the method to remove background noise automaticaly.
        if self.bkg_file is not None:
            self.substract_background_noise(self.bkg_file)

        # If the peak_positions are specified beforehand, the gaussian
        # centroids are initialized with the object
        if self.peak_positions[-1][1] is not None:
            self.fit_peak(plotting = False)


    def read_mca(self):
        """
        Reads the mca file specified in the object inicialization.

        If executed after adding background, it will remove it

        Output: rates: np.ndarray, time: int
        """

        with open(self.file_path, "r") as f:

            time = None
            counts = []
            line = f.readline()
            while line != "":
                if len(line) > 10 and line[:9] == "REAL_TIME":
                    time = int(line.split("-")[1]) # time in seconds

                if (line.strip()).isdigit():
                    counts.append(int(line))

                line = f.readline()

            xbins = np.arange(0, len(counts), 1)

            self.counts = np.array(counts)
            self.xbins = xbins
            self.time = time
            self.delta_x = self.xbins[1] - self.xbins[0]
            self.rates = self.counts / self.time

        return np.array(self.rates), time

    def rebining(self, bins_fused):
        """
        Performs a rebining of bins_fused channels. If called, it changes the previous rebining
        for a new one with the specified bins_fused

        Input: int: bins_fused (number of bins combined in the rebining)
        Output: rates: np.ndarray, time: int
        """

        # We make it posible to update the rebining
        if bins_fused != self.bins_fused:
            self.bins_fused = bins_fused

        # We call the read function again, to ensure that future rebinings
        # made calling the method aren't made over the previous rebining
        rates, time = self.read_mca()
        xbins = np.arange(0, len(rates), 1)

        new_xbins = []
        new_rates = []
        j = 0
        sum_bins = 0

        for i in range(len(rates)):

            # If we are on the first bin of the fused group,
            # we save the x value
            if j == 0:
                mean_x = xbins[i]

            # We keep adding up the bins until we reach
            # j == num_channels_fused.
            sum_bins += rates[i]
            j += 1

            # If we are on the last bin of the fused group,
            # we save the sum of counts and we calculate the
            # x value from the mean of the start and end x.
            if j == self.bins_fused:
                j = 0
                new_rates.append(sum_bins)
                mean_x += xbins[i]
                new_xbins.append(mean_x/2)
                sum_bins = 0

        self.time = time
        self.rates = np.array(new_rates)
        self.xbins = np.array(new_xbins)
        self.delta_x = self.xbins[1] - self.xbins[0]

        self.counts = new_rates * time

        return self.rates, self.xbins


    def substract_background_noise(self, bkg_file):
        """
        Substracts the background noise to the original file. It subtracts the background ratio to
        the main file's ratio. If called again, it recalculates the subtraction from scratch.

        Input: bkg_file: string (path to the background .mca file)
        Output: np.ndarray(): rates, int: time
        """

        Bkg = peakSelector(bkg_file, bins_fused = self.bins_fused)

        # We call the read function again, to ensure that future rebinings
        # made calling the method aren't made over the previous rebining
        selfRates, selfTime = self.rebining(self.bins_fused)

        rates = (selfRates - Bkg.rates)
        self.rates = rates
        self.counts = rates * self.time

        self.bkg_file = bkg_file
        self.bkg_rates = Bkg.rates
        self.bkg_time = Bkg.time

        return self.rates, self.time


    def get_rates_uncertainty(self):
        """
        Computes the uncertainty of the rates based on Poisson Statistics.

        Output: rates_uncertainty: np.ndaray
        """

        time_uncertainty = 1 # Our software only saves time in integers

        if self.bkg_rates is None:
            rates_uncertainty = np.sqrt(abs(self.rates / self.time) +
            (self.rates ** 2 / self.time ** 2) * time_uncertainty ** 2)

        # Needs updating, it is based on a simpified equations
        else:
            rates_uncertainty = np.sqrt(abs(self.rates / self.time) +
                                abs(self.bkg_rates / self.bkg_time))

        for i in range(len(rates_uncertainty)):
            rates_uncertainty[i] = round_uncertainty(rates_uncertainty[i])


        return rates_uncertainty

    def plot(self):
        """
        Plots the current rates.
        """
        with plt.style.context(mca.style):
            plt.style.use(mca.style)
            plt.rcParams.update({
                'figure.dpi': '120', # Suggested by https://github.com/garrettj403/SciencePlots/wiki/Gallery#styles-for-specific-academic-journals
                'font.size': 12.0
            })
            fig, ax = plt.subplots(1,1)
            ax.bar(self.xbins, self.rates, self.delta_x)

            ax.set_xlabel(transl["channels"][mca.lang])
            ax.set_ylabel(transl["rates"][mca.lang])
            fig.suptitle(transl["gamma spectrogram"][mca.lang])

            # Create a directory to store the figures (if it does not exist already)
            if not (CWD / self.fig_path).is_dir():
                os.mkdir(CWD / self.fig_path)

            # Save the plots
            # fig_path/Bismuth_data_PLOT.pdf
            fig_name = pathlib.Path(self.file_path).stem
            fig.savefig(
                str(CWD / self.fig_path / fig_name) + f"_PLOT.{self.fig_ext}"
            )

            plt.show()

    def plot_errorbar(self):
        """
        Plots the current rates with errorbars
        """

        with plt.style.context(mca.style):
            plt.style.use(mca.style)
            plt.rcParams.update({
                'figure.dpi': '120',
                'font.size': 12.0
            })
            fig, ax = plt.subplots(1,1)
            ax.errorbar(self.xbins, self.rates, yerr = self.get_rates_uncertainty(), fmt=".")

            ax.set_xlabel(transl["channels"][mca.lang])
            ax.set_ylabel(transl["rates"][mca.lang])
            fig.suptitle(transl["gamma spectrogram"][mca.lang])

            if not (CWD / self.fig_path).is_dir():
                os.mkdir(CWD / self.fig_path)

            # Save the errorbar plots
            # fig_path/Bismuth_data_ERRORBAR.pdf
            fig_name = pathlib.Path(self.file_path).stem
            fig.savefig(
                str(CWD / self.fig_path / fig_name) + f"_ERRORBAR.{self.fig_ext}"
            )

        plt.show()

    def select_peaks(self):
        """
        Opens an interactive plot to select the peaks to be adjusted with fit_peak.
        When closed, updates the self.peak_positions variable.
        """

        peak_positions = [ [ [], None ], ]
        line_positions = []

        sensibility = 0.01 * max(self.xbins) # 1% of the width


        def center_peak(peak):
            """
            Function that ensures there is the peak is centered for fitting.
            Recieves peak list as an input: [ [peak_limit_1, peak_limit_2], peak_type ]
            Takes the farthest limit and applies that distance to the closer limit.
            """

            idx1 = find_nearest(self.xbins, min(peak[0]))
            idx2 = find_nearest(self.xbins, max(peak[0]))

            # xbins and rates values for the selected zone (peak)
            xbins_peak = self.xbins[idx1:idx2]
            rates_peak = self.rates[idx1:idx2]

            if peak[1] == "single":
                # Here the center must be the highest point of the peak.

                # Finds the index of the highest rate value and looks for
                # its corresponding x value
                x_center = xbins_peak[find_nearest(rates_peak, max(rates_peak))]


            elif peak[1] == "double":
                # The center of a double peak must be the minimmum between
                # the two peaks. To find it, we can search for the min
                # y value in the second third of x values (between 1/3 and 2/3)

                y_center = min(rates_peak[len(xbins_peak) // 3 : 2 * len(xbins_peak) // 3])
                x_center = xbins_peak[find_nearest(rates_peak, y_center)] # we find the equivalent x

                # Now we can repeat the previous process

            # Now, we look for the peak_limit that is farther away.
            # We save the greatest distance and the index of the
            # item that is closer to the peak.

            farthest_distance = max(abs(np.array(peak[0]) - x_center))
            closest_index = np.argmin(abs(np.array(peak[0]) - x_center))

            # If the closest limit is the left one, we subtract
            # the farthest distance, if it is the right one, we add it
            if closest_index == 0: # Left one
                peak[0][0] = x_center - farthest_distance

            elif closest_index == 1: # Right one
                peak[0][1] = x_center + farthest_distance


            return peak


        def save_peak_type(peak_type: str):

            if len(peak_positions[-1][0]) == 0:
                print(transl["two points necessary"][mca.lang])

            else:
                if len(peak_positions[-1][0]) != 2:
                    print(transl["two points necessary"][mca.lang])


                else:
                    peak_positions[-1][0].sort() # We sort the peak
                    peak_positions[-1][1] = peak_type # Saving the peak type
                    # Now we modify the peak limits to center the peak
                    peak_positions[-1] = center_peak(peak_positions[-1])
                    # Add the scheme next element
                    peak_positions.append([[], None])

                    for line in line_positions[-1]:
                        line.set(color = "gray")

                    print(transl["peak selected"][mca.lang])

        def save_peak_data(x):
            """
            Save the data and plot the line
            """

            # Firstly, we check whether the last peak has both points
            if len(peak_positions[-1][0]) == 2 and peak_positions[-1][1] == None:
                print(transl["both points selected"][mca.lang])

            else:
                if len(peak_positions[-1][0]) == 1:
                    if abs(x - peak_positions[-1][0][0]) > sensibility:
                        peak_positions[-1][0].append(x)
                        save_line_data(x)

                else:
                    peak_positions[-1][0].append(x)
                    save_line_data(x)

        def save_line_data(x):

            line = ax.axvline(x=x, ymax=0.5, ls = "--", color = "black")

            # We want the lines in pairs. Each list represents a peak, and each
            # list is supposed to elements, one for each line.
            if len(line_positions) == 0:
                line_positions.append([line])

            else:
                if len(line_positions[-1]) == 1:
                    line_positions[-1].append(line)

                else:
                    line_positions.append([line])

        def reset_peak_data():
            if len(peak_positions[-1][0]) != 0:
                peak_positions.pop(-1)
                peak_positions.append([ [ ], None])

                for line in line_positions[-1]:
                    line.remove()
                line_positions.pop(-1)


        def reset_global_data():
            for lines in line_positions:
                for line in lines:
                    line.remove()

            for i in range(len(peak_positions)):
                # Every time you remove a value, the indexes adapt.
                # index 1 turns to index 0
                peak_positions.pop(0)
                if i < (len(peak_positions) - 2):
                    line_positions.pop(0)

            peak_positions.append([ [], None ])



        def click_event(event):
            # For clicking the histogram
            if isinstance(event.artist, Rectangle):
                x = event.artist.get_x()
                save_peak_data(x)
                fig.canvas.draw()

            # For clicking the text
            elif isinstance(event.artist, Text):
                text = event.artist.get_text()
                if text == transl["reset current peak"][mca.lang]:
                    reset_peak_data()
                    fig.canvas.draw()

                elif text == transl["reset all peaks"][mca.lang]:
                    reset_global_data()
                    fig.canvas.draw()

                elif text == transl["mark as single"][mca.lang]:
                    save_peak_type("single")
                    fig.canvas.draw()

                elif text == transl["mark as double"][mca.lang]:
                    save_peak_type("double")
                    fig.canvas.draw()

        def close_event(event):
            if peak_positions[-1][1] == None:
                self.peak_positions = peak_positions[:-1]

            self.fit_peak()


        # We define the plot. The bar plot and the vertical lines
        fig, ax = plt.subplots()
        ax.bar(self.xbins, self.rates, self.delta_x, picker = True)

        fig.suptitle(self.file_path.split("/")[-1].split(".")[0])


        # Here we add the interactive text
        max_xbins = max(self.xbins)
        max_rates = max(self.rates)

        # Peak options
        ax.text(0.7 * max_xbins , 0.95 * max_rates ,
                transl["confirm peak"][mca.lang], size="x-large")

        ax.text(0.7 * max_xbins, 0.85 * max_rates,
                transl["mark as single"][mca.lang], picker = True,
                size="large", style = "italic")

        ax.text(0.7 * max_xbins, 0.75 * max_rates,
                transl["mark as double"][mca.lang], picker = True,
                size="large", style = "italic")


        # Global options
        ax.text(0.20 * max_xbins, 0.95 * max_rates,
                transl["reset peak"][mca.lang], size="x-large")

        ax.text(0.20 * max_xbins, 0.85 * max_rates,
                transl["reset current peak"][mca.lang], picker = True,
                size="large", style = "italic")

        ax.text(0.20 * max_xbins, 0.75 * max_rates,
                transl["reset all peaks"][mca.lang], picker = True,
                size="large", style = "italic")



        fig.canvas.mpl_connect('pick_event', click_event)
        fig.canvas.mpl_connect('close_event', close_event)
        plt.show()


    def fit_peak(self, **kwargs):
        """
        Uses scipy.optimize.curve_fit to fit the peaks selected with .select_peaks() method.
        It accepts all curve_fit kwargs.

        Output: popt: np.ndarray (optimal parameters), pcov: np.ndaray (covariance matrix)
        """

        # If no peaks were selected, we avoid unnecessary operations
        if len(self.peak_positions) == 0:
            print(transl["no peaks selected"][mca.lang])
            return None

        # We separate the non curve_fit kwargs from plotting:
        non_fit_kwargs = {}
        for k, val in kwargs.items():
            if k == "plotting":
                non_fit_kwargs.update({k: val})

        # We want plotting by default. Appart from that, we need
        # to remove it from kwargs to avoid curve_fit errors.
        if "plotting" in kwargs:
            del kwargs["plotting"]

        else:
            non_fit_kwargs.update({"plotting": True})


        # Functions for single and double peaks.
        # One or two gaussian peaks with polynomial background.

        def gaussian_peak(x, p3, p4, p5):
            """
            Gaussian peak
            """
            return p3 * np.exp(-0.5 * ((x - p4) / p5) ** 2)

        def polynomial_background(x, p0, p1, p2):
            """
            Polynomial background
            """
            return p0 + p1 * x + p2 * x ** 2

        def single_peak(x, p0, p1, p2, p3, p4, p5):
            """
            Functional form of a gaussian with polynomial background
            """
            bkg_func = polynomial_background(x, p0, p1, p2)
            gaussian_func = gaussian_peak(x, p3, p4, p5)
            return bkg_func + gaussian_func

        def double_peak(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
            """
            Functional form of a double gaussian with polynomial background
            """
            bkg_func = polynomial_background(x, p0, p1, p2)
            gaussian_func1 = gaussian_peak(x, p3, p4, p5)
            gaussian_func2 = gaussian_peak(x, p6, p7, p8)
            return bkg_func + gaussian_func1 + gaussian_func2

        def get_FWHM(x_center, xbins, rates):
            """
            Computes an aproximate value of FWHM
            (Full Width at Half Maximum) of a peak.
            Inputs: x_center (center of the peak in bins),
                    xbins: (x values of the peak),
                    rates: (rates of the peak)

            Output: FWHM
            """
            # It's not at the half because it helps fit.
            y_FWHM = max(rates) * 0.65
            x_FWHM = x[find_nearest(rates, y_FWHM)]
            FWHM = abs(x_FWHM - x_center)
            return FWHM

        list_popt = []
        list_pcov = []
        list_chi2 = []

        for peak in self.peak_positions:

            idx1 = find_nearest(self.xbins, min(peak[0]))
            idx2 = find_nearest(self.xbins, max(peak[0]))

            # x, y and uncertainty in y for curve_fit
            x = self.xbins[idx1:idx2]
            y = self.rates[idx1:idx2]
            sy = self.get_rates_uncertainty()[idx1:idx2]

            x_fit = np.linspace(x[0], x[-1], 100)
            y_fit = np.zeros(len(x_fit))
            y_gauss_1 = np.zeros(len(x_fit))
            y_gauss_2 = np.zeros(len(x_fit))
            y_background = np.zeros(len(x_fit))

            try:
                if peak[1] == "single":

                    # Firsly, we get some data of the peak
                    x_center = x[len(x) // 2] # The peak is centered

                    # Now we compute the FWHM (Full Width at Half Maximum).
                    # Sigma is related to the FWHM: 2.35 * sigma = FWHM
                    FWHM = get_FWHM(x_center, x, y)

                    # We estimate the parameters. The most important are
                    # the center and sigma, the other ones can be initialized as 1
                    p0 = [1,1,1,1, x_center, FWHM]


                    # Now we compute the curve_fit (with additional kwargs or with auto p0)
                    if len(kwargs) == 0:
                        popt, pcov, infodict, mseg, ier = curve_fit(single_peak, x, y, p0=p0, sigma = sy, full_output = True)
                    else:
                    # TODO: If full output is introduced as a kwarg, the program is going to raise an error
                        popt, pcov = curve_fit(single_peak, x, y, sigma = sy, **kwargs)

                    # We calculate the plotting points of the theoretical function.
                    y_fit = single_peak(x_fit, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
                    y_background = polynomial_background(x_fit, popt[0], popt[1], popt[2])
                    y_gauss_1 =  gaussian_peak(x_fit, popt[3], popt[4], popt[5])

                elif peak[1] == "double":
                    # Firstly, we need the x values of the peak centers.

                    # We split the list in two, one for each peak
                    x_1 = x[: len(x) // 2]
                    x_2 = x[len(x) // 2 :]

                    y_1 = y[: len(y) // 2]
                    y_2 = y[len(y) // 2 :]

                    # We locate the x values of each peak:
                    x_center_1 = x_1[find_nearest(y_1, max(y_1))]
                    x_center_2 = x_2[find_nearest(y_2, max(y_2))]

                    # Now we compute the FWHM (Full Width at Half Maximum).
                    # Sigma is related to the FWHM: 2.35 * sigma = FWHM
                    FWHM_1 = get_FWHM(x_center_1, x_1, y_1)
                    FWHM_2 = get_FWHM(x_center_1, x_1, y_1)

                    # We estimate the parameters. The most important are
                    # the center and sigma, the other ones can be initialized as 1

                    p0 = [1,1,1,1, x_center_1, FWHM_1, 1, x_center_2, FWHM_2]


                    # Now we compute the curve_fit (with additional kwargs or with auto p0)
                    if len(kwargs) == 0:
                        popt, pcov, infodict, mseg, ier = curve_fit(double_peak, x, y, p0=p0, sigma = sy, full_output = True)
                    else:
                    # TODO: If full output is introduced as a kwarg, the program is going to raise an error
                        popt, pcov = curve_fit(double_peak, x, y, sigma = sy, **kwargs)


                    # We calculate the plotting points of the theoretical function.
                    y_fit = double_peak(x_fit, popt[0], popt[1], popt[2], popt[3],
                                           popt[4], popt[5], popt[6], popt[7], popt[8])
                    y_background = polynomial_background(x_fit, popt[0], popt[1], popt[2])
                    y_gauss_1 = gaussian_peak(x_fit, popt[3], popt[4], popt[5])
                    y_gauss_2 = gaussian_peak(x_fit, popt[6], popt[7], popt[8])


                # We only plotting it's true (default value)
                for k, val in non_fit_kwargs.items():
                    if k == "plotting" and val:
                        # We plot the result

                        # Set the style. Style selection and latex installation
                        # check is performed in __init__.py
                        with plt.style.context(mca.style):
                            plt.style.use(mca.style)
                            print(plt.rcParams.keys())
                            plt.rcParams.update({
                                'figure.dpi': '250',
                                'figure.figsize': [7.5, 6.5],
                                'figure.constrained_layout.use': True,
                                'font.size': 12.0
                            })
                            fig, ax = plt.subplots(1,1)
                            ax.plot(x_fit, y_fit, label = transl["fit"][mca.lang])
                            ax.plot(x_fit, y_background, label = transl["background"][mca.lang])
                            ax.errorbar(x,y, yerr=sy ,fmt=".", label = transl["points"][mca.lang])

                            if peak[1] == "double":
                                ax.plot(x_fit, y_gauss_1, label = transl["gauss 1"][mca.lang])
                                ax.plot(x_fit, y_gauss_2, label = transl["gauss 2"][mca.lang])

                            else:
                                ax.plot(x_fit, y_gauss_1, label = transl["gauss"][mca.lang])


                            ax.legend()
                            ax.set_xlabel(transl["channels"][mca.lang])
                            ax.set_ylabel(transl["rates"][mca.lang])
                            fig.suptitle(transl["gamma spectrogram"][mca.lang])

                            if not (CWD / self.fig_path).is_dir():
                                os.mkdir(CWD / self.fig_path)

                            # Save the plots
                            # fig_path/Bismuth_data_FITPEAK_2560_2998_single.pdf
                            fig_name = pathlib.Path(self.file_path).stem
                            fig.savefig(
                                str(CWD / self.fig_path / fig_name)
                                + f"_FITPEAK_{peak[0][0]:.0f}_{peak[0][1]:.0f}_{peak[1]}.{self.fig_ext}"
                            )

                            # Showing the plots
                            plt.show()

                # We compute chi2 and its degrees of freedom
                chi2 = sum(infodict["fvec"]**2) # fvec = (y - f(x)) / sigma_i
                deg_free = len(y) - len(popt) # Number of points - number of fitted params

                # We append the values to the return list
                list_pcov.append(pcov)
                list_popt.append(popt)
                list_chi2.append((chi2, deg_free))

            except RuntimeError:
                print(transl["optimal parameters not found"][mca.lang])
                popt = None
                pcov = None


        # Before returning, and for calibration purposes, we will find
        # all gaussian centroids and append them into a list
        peak_channels = []
        peak_channels_uncertainty = []

        for i in range(len(list_pcov)):
            if peak[1] == "single":
                peak_channels.append(list_popt[i][4])
                peak_channels_uncertainty.append(list_pcov[i][4,4])

            elif peak[1] == "double":
                peak_channels.append(list_popt[i][4])
                peak_channels_uncertainty.append(list_pcov[i][4,4])

                peak_channels.append(list_popt[i][7])
                peak_channels_uncertainty.append(list_pcov[i][7,7])


        sorted_pairs = sorted(zip(peak_channels, peak_channels_uncertainty))
        self.peak_channels = [v1 for v1, v2 in sorted_pairs]
        self.peak_channels_uncertainty = [v2 for v1, v2 in sorted_pairs]


        # We return the data
        return list_popt, list_pcov, list_chi2


    def save_peaks(self, file_path):
        """
        Saves peak info to specified text file
        """

        # We check if the file exists
        if os.path.exists(file_path):
            user_input = (tranls["overwrite file?"][mca.lang])
            if user_input.lower() != "y":
                print(transl["cancelling operation"][mca.lang])
                return
            else:
                print(transl["overwritting file"][mca.lang])


        with open(file_path, "w") as f:
            for peak in self.peak_positions:
                f.write(f"{peak[0][0]}, {peak[0][1]}, {peak[1]}\n")


    def load_peaks(self, file_path):
        """
        Load peak data from specified file
        """

        with open(file_path, "r") as f:

            peak_positions = []
            line = f.readline()
            while line != "":
                peak_limit_1, peak_limit_2, peak_type = line.rstrip("\n").split(",")
                peak_positions.append([[float(peak_limit_1), float(peak_limit_2)], peak_type.lstrip()])
                line = f.readline()


        # We save the loaded peak_positions
        self.peak_positions = peak_positions
        self.fit_peak(plotting = False)
        return peak_positions

    def set_peak_energies(self, energy_list):
        """
        Sets the real energy value of the peaks
        """
        energy_list.sort()
        self.peak_energies = energy_list


    def save_peak_energies(self, file_path):
        """
        Saves the real energy value of the peaks
        """

         # We check if the file exists
        if os.path.exists(file_path):
            user_input = (tranls["overwrite file?"][mca.lang])
            if user_input.lower() != "y":
                print(transl["cancelling operation"][mca.lang])
                return
            else:
                print(transl["overwritting file"][mca.lang])


        with open(file_path, "w") as f:
            for value in self.peak_energies:
                f.write(f"{value}\n")


    def load_peak_energies(self, file_path):

        with open(file_path, "r") as f:

            peak_energies = []
            line = f.readline()
            while line != "":
                peak_energies.append(float(line))
                line = f.readline()

        self.set_peak_energies(peak_energies)


    def save_fit_info(self, file_path):
        list_popt, list_pcov, list_chi2 = self.fit_peak(plotting = False)

        # We check if the file exists
        if os.path.exists(file_path):
            user_input = (transl["overwrite file?"][mca.lang])
            if user_input.lower() != "y":
                print(transl["cancelling operation"][mca.lang])
                return
            else:
                print(transl["overwritting file"][mca.lang])


        with open(file_path, "w") as f:

            for i in range(len(list_popt)):
                sigmas = np.sqrt(np.diag(list_pcov[i]))
                f.write(f"Pico {i}:\n")
                f.write("Parametros óptimos\n")
                f.write(f"Fondo: {list_popt[i][0]}({sigmas[0]}) + {list_popt[i][1]}({sigmas[1]})*x + {list_popt[i][2]}({sigmas[2]})*x^2\n")

                # Simple peak
                if len(list_popt[i]) == 6:
                    f.write("Pico:\n")
                    f.write(f"Integral: {list_popt[i][3]}({sigmas[3]})\n") # I need to calculate the integral real value
                    f.write(f"Centroide: {list_popt[i][4]}({sigmas[3]})\n")
                    f.write(f"Sigma = {list_popt[i][5]}({sigmas[3]})\n")

                # Double peak
                elif len(list_popt[i]) == 9:
                    f.write("Pico a:\n")
                    f.write(f"Integral: {list_popt[i][3]}({sigmas[3]})\n") # I need to calculate the integral real value
                    f.write(f"Centroide: {list_popt[i][4]}({sigmas[3]})\n")
                    f.write(f"Sigma = {list_popt[i][5]}({sigmas[3]})\n")

                    f.write("Pico b:\n")
                    f.write(f"Integral: {list_popt[i][6]}({sigmas[6]})\n") # I need to calculate the integral real value
                    f.write(f"Centroide: {list_popt[i][7]}({sigmas[7]})\n")
                    f.write(f"Sigma = {list_popt[i][8]}({sigmas[8]})\n")

                # Triple peak (pending implemetation)
                elif len(list_popt[i]) == 12:
                    f.write("Pico a:\n")
                    f.write(f"Integral: {list_popt[i][3]}({sigmas[3]})\n") # I need to calculate the integral real value
                    f.write(f"Centroide: {list_popt[i][4]}({sigmas[3]})\n")
                    f.write(f"Sigma = {list_popt[i][5]}({sigmas[3]})\n")

                    f.write("Pico b:\n")
                    f.write(f"Integral: {list_popt[i][6]}({sigmas[6]})\n") # I need to calculate the integral real value
                    f.write(f"Centroide: {list_popt[i][7]}({sigmas[7]})\n")
                    f.write(f"Sigma = {list_popt[i][8]}({sigmas[8]})\n")

                    f.write("Pico c:\n")
                    f.write(f"Integral: {list_popt[i][9]}({sigmas[9]})\n") # I need to calculate the integral real value
                    f.write(f"Centroide: {list_popt[i][10]}({sigmas[10]})\n")
                    f.write(f"Sigma = {list_popt[i][11]}({sigmas[11]})\n")


                f.write(f"Reduced Chi squared: {list_chi2[i][0] / list_chi2[i][1]}\n")
                f.write(f"p-value(%): {100 * (1 - chi2.cdf(list_chi2[i][0], list_chi2[i][1]))}\n")




    def print_fit_info(self):
        list_popt, list_pcov, list_chi2 = self.fit_peak(plotting = False)

        for i in range(len(list_popt)):
            sigmas = np.sqrt(np.diag(list_pcov[i]))
            print(f"Pico {i}:")
            print("Parametros óptimos")
            print(f"Fondo: {list_popt[i][0]}({sigmas[0]}) + {list_popt[i][1]}({sigmas[1]})*x + {list_popt[i][2]}({sigmas[2]})*x^2")

            # Simple peak
            if len(list_popt[i]) == 6:
                print("Pico:")
                print(f"Integral: {list_popt[i][3]}({sigmas[3]})") # I need to calculate the integral real value
                print(f"Centroide: {list_popt[i][4]}({sigmas[3]})")
                print(f"Sigma = {list_popt[i][5]}({sigmas[3]})")

            # Double peak
            elif len(list_popt[i]) == 9:
                print("Pico a:")
                print(f"Integral: {list_popt[i][3]}({sigmas[3]})") # I need to calculate the integral real value
                print(f"Centroide: {list_popt[i][4]}({sigmas[3]})")
                print(f"Sigma = {list_popt[i][5]}({sigmas[3]})")

                print("Pico b:")
                print(f"Integral: {list_popt[i][6]}({sigmas[6]})") # I need to calculate the integral real value
                print(f"Centroide: {list_popt[i][7]}({sigmas[7]})")
                print(f"Sigma = {list_popt[i][8]}({sigmas[8]})")

            # Triple peak (pending implemetation)
            elif len(list_popt[i]) == 12:
                print("Pico a:")
                print(f"Integral: {list_popt[i][3]}({sigmas[3]})") # I need to calculate the integral real value
                print(f"Centroide: {list_popt[i][4]}({sigmas[3]})")
                print(f"Sigma = {list_popt[i][5]}({sigmas[3]})")

                print("Pico b:")
                print(f"Integral: {list_popt[i][6]}({sigmas[6]})") # I need to calculate the integral real value
                print(f"Centroide: {list_popt[i][7]}({sigmas[7]})")
                print(f"Sigma = {list_popt[i][8]}({sigmas[8]})")

                print("Pico c:")
                print(f"Integral: {list_popt[i][9]}({sigmas[9]})") # I need to calculate the integral real value
                print(f"Centroide: {list_popt[i][10]}({sigmas[10]})")
                print(f"Sigma = {list_popt[i][11]}({sigmas[11]})")


            print(f"Reduced Chi squared: {list_chi2[i][0] / list_chi2[i][1]}")
            print(f"p-value(%): {100 * (1 - chi2.cdf(list_chi2[i][0], list_chi2[i][1]))}")
