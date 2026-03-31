from .translations import valid_languages
import shutil

# Default value
lang = "en"
def select_language(new_lang):
    global lang
    if new_lang in valid_languages:
        lang = new_lang
    else:
        print("Not a valid language, using English!")
        lang = "en"


# Setting defaul style
# Check if latex is present
if shutil.which('latex'):
    style = ['science', 'ieee']
else:
    style = ['science',"no-latex",'ieee']

# Add an option to manualy select plot style
def select_plot_style(plot_style):
    global style
    style = plot_style
    # An option to check all avaliable styles should be added

# These are loaded after the lang to avoid circular import
from .peakSelector import peakSelector
from .calibration import calibration, calibration_helper


__all__ = ["peakSelector",
            "calibration",
            "calibration_helper"
            "select_language"]







