import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from scipy.optimize import curve_fit

from uncertainty import round_uncertainty
from translations import valid_languages
from translations import translation_peakSelector as transl

# Util funcions to reduce cluttering in the class

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

    Methods:
    read_mca: reads the number of counts in each channel and the total
              time of the measurement.

    rebining: joins the data counts of a number of channels to reduce
              uncertainty on the parametric optimization. By default,
              it makes a 10 bin rebining. .rebining takes as an argument
              the number of bins fused.

    plot: represents the data in a static, non-interactive way. It's used
          to test if the data was loaded properly.

    interactive_plot: lets the user select the desired peaks to be
                      analyzed by curve_fit. It works with matplotlib
                      picker events.

    """
    def __init__(self, file, **kwargs):
        """
        Indicate self variables here:
        filePath
        bins_fused
        time
        bins
        xbins
        """

        # Basic variables
        self.file_path = file
        self.time = 0
        self.bins = []
        self.xbins = []
        self.delta_x = 0
        self.bins_uncertainty = []

        # User options
        self.bins_fused = 10 # Default number of bins fused in rebining
        self.lang = "en" # Plots language

        # Interactive plot
        self.peak_positions = [ [ [], None ], ]

        # Background noise filtering
        self.bkg_file = None
        self.bkg_bins = None
        self.bkg_time = None

        # Modify default values with kwargs
        for k, val in kwargs.items():
            if k == "bins_fused":
                self.bins_fused = val
            elif k == "lang":
                if val in valid_languages:
                    self.lang = val
                else:
                    print("Not a valid language, using English!")
            elif k == "bkg_file":
                self.bkg_file = val


        self.read_mca()

        # Rebining can be disbled by usings kwargs.
        if self.bins_fused != 0 and self.bins_fused != False:
            self.rebining(self.bins_fused)

        if self.bkg_file is not None:
            self.substract_background_noise(self.bkg_file, self.bins_fused)

        self.get_bins_uncertainty()


    def read_mca(self):
        f = open(self.file_path, "r")

        time = None
        bins = []
        line = f.readline()
        while line != "":
            if len(line) > 10 and line[:9] == "REAL_TIME":
                time = int(line.split("-")[1]) # time in seconds

            if (line.strip()).isdigit():
                bins.append(int(line))

            line = f.readline()

        xbins = np.arange(0, len(bins), 1)

        self.bins = np.array(bins)
        self.xbins = xbins
        self.time = time
        self.delta_x = self.xbins[1] - self.xbins[0]


        return np.array(bins), time

    def rebining(self, bins_fused):
        """
        Performs a rebining of bins_fused channels
        """

        # We make it posible to update the rebining
        if bins_fused != self.bins_fused:
            self.bins_fused = bins_fused


        new_xbins = []
        new_bins = []
        j = 0
        sum_bins = 0

        for i in range(len(self.bins)):

            # If we are on the first bin of the fused group,
            # we save the x value
            if j == 0:
                mean_x = self.xbins[i]

            # We keep adding up the bins until we reach
            # j == num_channels_fused.
            sum_bins += self.bins[i]
            j += 1

            # If we are on the last bin of the fused group,
            # we save the sum of counts and we calculate the
            # x value from the mean of the start and end x.
            if j == self.bins_fused:
                j = 0
                new_bins.append(sum_bins)
                mean_x += self.xbins[i]
                new_xbins.append(mean_x/2)
                sum_bins = 0

        self.bins = np.array(new_bins)
        self.xbins = np.array(new_xbins)
        self.delta_x = self.xbins[1] - self.xbins[0]


        return np.array(new_bins), np.array(new_xbins)

    def substract_background_noise(self, bkg_file, bins_fused):

        Bkg = peakSelector(bkg_file, bins_fused = self.bins_fused)

        ratioBkg = Bkg.bins / Bkg.time
        ratioSelf = self.bins / self.time

        self.bkg_bins = Bkg.bins
        self.bkg_time = Bkg.time
        bins = (ratioSelf - ratioBkg) * self.time
        # We need to remove the negative bins, they cause problems with
        # uncertainty.
        for i in range(len(bins)):
            if bins[i] < 0:
                bins[i] = 0

        self.bins = bins


    def get_bins_uncertainty(self):

        # This is the simplified equation
        if self.bkg_bins is None:
            bins_uncertainty = np.sqrt(self.bins / self.time ** 2)

        else:
            bins_uncertainty = np.sqrt(self.bins / self.time ** 2 +
                                    self.bkg_bins / self.bkg_time ** 2)

        for i in range(len(bins_uncertainty)):
            bins_uncertainty[i] = round_uncertainty(bins_uncertainty[i])

        self.bins_uncertainty = bins_uncertainty

        return bins_uncertainty

    def plot(self):

        fig, ax = plt.subplots(1,1)
        ax.bar(self.xbins, self.bins, self.delta_x)

        ax.set_xlabel(transl["channels"][self.lang])
        ax.set_ylabel(transl["counts"][self.lang])
        fig.suptitle(transl["gamma spectrogram"][self.lang])

        plt.show()

    def interactive_plot(self):

        peak_positions = [ [ [], None ], ]
        line_positions = []

        sensibility = 0.01 * max(self.xbins) # 1% of the width


        def save_peak_type(peak_type: str):

            if len(peak_positions[-1][0]) == 0:
                print(transl["two points necessary"][self.lang])

            else:
                if len(peak_positions[-1][0]) != 2:
                    print(transl["two points necessary"][self.lang])


                else:
                    peak_positions[-1][1] = peak_type
                    # Add the scheme next element
                    peak_positions.append([[], None])

                    for line in line_positions[-1]:
                        line.set(color = "gray")

                    print(transl["peak selected"][self.lang])

        def save_peak_data(x):
            """
            Save the data and plot the line
            """

            # Firstly, we check whether the last peak has both points
            if len(peak_positions[-1][0]) == 2 and peak_positions[-1][1] == None:
                print(transl["both points selected"][self.lang])

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
            if isinstance(event.artist, Rectangle):
                x = event.artist.get_x()
                save_peak_data(x)
                fig.canvas.draw()

            elif isinstance(event.artist, Text):
                text = event.artist.get_text()
                if text == transl["reset current peak"][self.lang]:
                    reset_peak_data()
                    fig.canvas.draw()

                elif text == transl["reset all peaks"][self.lang]:
                    reset_global_data()
                    fig.canvas.draw()

                elif text == transl["mark as single"][self.lang]:
                    save_peak_type("single")
                    fig.canvas.draw()

                elif text == transl["mark as double"][self.lang]:
                    save_peak_type("double")
                    fig.canvas.draw()

        def close_event(event):
            if peak_positions[-1][1] == None:
                self.peak_positions = peak_positions[:-1]

            # self.fit_peak()


        # We define the plot. The bar plot and the vertical lines
        fig, ax = plt.subplots()
        ax.bar(self.xbins, self.bins, self.delta_x, picker = True)


        # Here we add the interactive text

        # Peak options
        max_xbins = max(self.xbins)
        max_bins = max(self.bins)

        ax.text(0.7 * max_xbins , 0.95 * max_bins ,
                transl["confirm peak"][self.lang], size="x-large")

        ax.text(0.7 * max_xbins, 0.85 * max_bins,
                transl["mark as single"][self.lang], picker = True,
                size="large", style = "italic")

        ax.text(0.7 * max_xbins, 0.75 * max_bins,
                transl["mark as double"][self.lang], picker = True,
                size="large", style = "italic")


        # Global options
        ax.text(0.20 * max_xbins, 0.95 * max_bins,
                transl["reset peak"][self.lang], size="x-large")

        ax.text(0.20 * max_xbins, 0.85 * max_bins,
                transl["reset current peak"][self.lang], picker = True,
                size="large", style = "italic")

        ax.text(0.20 * max_xbins, 0.75 * max_bins,
                transl["reset all peaks"][self.lang], picker = True,
                size="large", style = "italic")



        fig.canvas.mpl_connect('pick_event', click_event)
        fig.canvas.mpl_connect('close_event', close_event)
        fig.show()


    def fit_peak(self):

        # If no peaks were selected, we avoid unnecessary operations
        if self.peak_positions[-1][1] is None:
            return None


        # Functions for single and double peaks.
        # One or two gaussian peaks with polynomial backgroud.
        def single_peak(x, p0, p1, p2, p3, p4, p5):
            bkg_func = p0 + p1 * x + p2 * x ** 2
            gaussian_func = p3 * np.exp(-0.5 * ((x - p4) / p5) ** 2)
            return bkg_func + gaussian_func

        def double_peak(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
            bkg_func = p0 + p1 * x + p2 * x ** 2
            gaussian_func1 = p3 * np.exp(-0.5 * ((x - p4) / p5) ** 2)
            gaussian_func2 = p6 * np.exp(-0.5 * ((x - p7) / p8) ** 2)
            return bkg_func + gaussian_func1 + gaussian_func2


        # Find nearest array element to value
        def find_nearest(array, value):
            return (np.abs(array - value)).argmin()


        for peak in self.peak_positions:

            idx1 = find_nearest(self.xbins, min(peak[0]))
            idx2 = find_nearest(self.xbins,max(peak[0]))

            # x, y and uncertainty in y for curve_fit
            x = self.xbins[idx1:idx2]
            y = self.bins[idx1:idx2]
            sy = self.bins_uncertainty[idx1:idx2]

            x_fit = np.linspace(x[0], x[-1], 100)
            y_fit = np.zeros(len(x_fit))
            try:
                if peak[1] == "single":
                    p0 = [1,1,1,
                        sum(y) / np.sqrt(2*np.pi),
                        np.mean(peak[0]),
                        abs(x[0] - x[len(x) // 2])]
                    popt, pcov = curve_fit(single_peak, x, y, p0=p0, sigma = sy)

                    for i in range(len(x_fit)):
                        y_fit[i] = single_peak(x_fit[i], popt[0], popt[1],
                                        popt[2], popt[3], popt[4], popt[5])
                elif peak[1] == "double":
                    p0 = [1,1,1,
                        -sum(y) / np.sqrt(2*np.pi),
                        np.mean(peak[0]) * 1/3,
                        abs(x[0] - x[len(x) // 4]),
                        -sum(y) / np.sqrt(2*np.pi),
                        np.mean(peak[0]) * 2/3,
                        abs(x[0] - x[len(x) // 4])]


                    popt, pcov = curve_fit(double_peak, x, y, p0=p0, sigma = sy)
                    print(popt)
                    for i in range(len(x_fit)):
                        y_fit[i] = double_peak(x_fit[i], popt[0], popt[1],
                                    popt[2], popt[3], popt[4], popt[5],
                                    popt[6], popt[7], popt[8])



                fig, ax = plt.subplots(1,1)
                ax.plot(x_fit, y_fit)
                ax.plot(x,y, ".")
                print(popt)

            except RuntimeError:
                print("Optimal parameters not found")


        plt.show()
        return popt, pcov




# Test
if __name__ == "__main__":

    Bi = peakSelector("Test/Bi.mca", bkg_file = "Test/Background.mca")
    Co = peakSelector("Test/Co.mca", bkg_file="Test/Background.mca", bins_fused = 20)

    Bi.interactive_plot()
    Bi.fit_peak





