# MCA-tools
MCA-tools is a toolset designed for processing the output of multichannel analyzers in gamma spectroscopy studies. It is currently work in progress. 
The expected feature set is:

- Obtaining the parameters of the gaussian function that characterizes the peaks (mostly implemented).
- Calibrating the detector using known energies of gamma rays
- Characterizing the geometrical eficiency of the detector (including a Monte Carlo simulation)

## Depencencies
Currently it depends on:
- Matplotlib
- Numpy
- Scipy.optimize

## `peakSelector` class
Currently, the only thing that it implemented is the `peakSelector` class. This class takes the mca file path as an input and processes the data to get
the parameters that fit the peak to a gaussian function. The class works by:

1. *Reading the bins:* it also asign them a x value. It reads the total time in order to get the rates.
2. *Doing a rebining:* by default it merges 10 channels. It can be modified using `.rebining(num_bins)` method.
3. *Removing the background noise:* if specified using kwarg `bkg_file` in definiton or using the `.remove_background_noise(bkg_file)` method, it
   calculates the rates of both files and removes the contibution of background radiation. It returns the total number of counts, not the rates.
4. *Interactive plot*: when you call the `.interactive_plot()` method, you open a matplotlib `plot()` with interactivee capabilities. You can select
    the start and end points in a peak and select if the peak is double of single. When all of the peaks are selected and the window closed, the
    method calls `.fit_peak()`.
5. *Fitting the peak*: the last step is calling scipy's `curve_fit` function to get the optimal parameters.

## Translations
The file `translations.py` contains a dictionary with multiple texts in 3 languages (Galician, Spanish and English). The rest of the documentation will
also be translated in the future.
