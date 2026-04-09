import numpy as np
from scipy.stats import chi2

def round_uncertainty(x):
    # The 1 turns 1 siginficant figure into 2 significant figures.
    if x != 0:
        return round(x, 1-int(np.floor(np.log10(abs(x)))))
    else:
        return 0

def get_pvalue(chi2_val, df):
    p_value = 1 - chi2.cdf(chi2_val, df)
    reduced_chi2 = chi2_val / df
    return (reduced_chi2, p_value)

def print_uncertainty(value, uncertainty):

    non_decimal = str(value).split(".")[0] # Non decimal part of the value (string)

    # Position of the most siginficant figure of the value and the uncertainty
    uncert_signif_fig = int(np.floor(np.log10(abs(uncertainty))))
    value_signif_fig = int(np.floor(np.log10(abs(value))))


    # Position of the 2 most significant figures of the uncertainty
    uncert_round_fig = 1 - uncert_signif_fig

    # Getting the significant figures in the string
    # Getting the value this way makes the formatting round the uncertainty
    # automaticaly
    uncert_str = f"{uncertainty:.1e}"


    # If the uncertainty has the same significant figures as
    # its value, the uncertainty has to be written differently
    # This is because we are using scientific notation
    if value_signif_fig == uncert_signif_fig:
        uncert_str = f"({uncert_str[0]}.{uncert_str[2]})"
        significant_values = 1 # We print 2 significant values in both the value and uncertainty

    else:
        uncert_str = f"({uncert_str[0]}{uncert_str[2]})"

        # The last check we need is if the  number of significant values on the
        # uncertainty is larger than the main value, that needs to raise an error

        # The first number doesn't count, so we have to subtract 1
        if int(non_decimal) == 0:
            significant_values = len(non_decimal) + uncert_round_fig - 2

        else:
            significant_values = len(non_decimal) + uncert_round_fig - 1

    string_formatter = f".{significant_values}e"
    rounded_value = f"{value:{string_formatter}}"


    # If the uncertainty is bigger that the magnitude, we don't
    # express it as the last two digits, we set the first one, and after
    # that, the uncertainty
    if uncert_signif_fig > value_signif_fig:
        return rounded_value + f"({uncertainty:.1e})"

    else:
        # We add the uncertainty in parenthesis in betweeen the rounded value
        splitted_rounded_value = rounded_value.split("e")
        return splitted_rounded_value[0] + uncert_str + "e" + splitted_rounded_value[1]

