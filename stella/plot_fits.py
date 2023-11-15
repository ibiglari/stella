import os, gc, sys
from threading import Thread

from tqdm import tqdm

import numpy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter

from lightkurve.search import search_lightcurve

ROOT_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../"
print("Initializing environment...", end = "", flush = True)
sys.path.insert(1, ROOT_PATH)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("done", flush = True)

print("Importing packages...", end = "", flush = True)
import lightkurve.io as lkio

from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.table import unique

import stella
import tensorflow as tf
#tf.keras.utils.disable_interactive_logging()
matplotlib.use("agg")

print("done", flush = True)

MODEL_WIDTH = "50"
MODEL_EPOCHS = "400"
MODELS_FOLDER = ROOT_PATH + "/models/" + MODEL_WIDTH + "-" + MODEL_EPOCHS + "/"
OUTPUT_FOLDER = ROOT_PATH + "/output/comparision/"

COLOR_YELLOW = 'tab:olive'
COLOR_RED = 'tab:red'
FAST_PLOTS_FOLDER_NAME_TEMPLATE = "{0:09d}/fast/"
SHORT_PLOTS_FOLDER_NAME_TEMPLATE = "{0:09d}/short/"
OVERVIEW_FILENAME_TEMPLATE = "(1) - overview_{0:09d}_sector{1:02d}.png"
PREDICTIONS_FILENAME_TEMPLATE = "(2) - predictions_{0:09d}_model_{1:s}_sector_{2:02d}.png"
PREDICTION_FILENAME_TEMPLATE = "(3) - pred_sector{0:02d}_model_{1:s}_part_{2:05d}.png"
FLARE_FILENAME_TEMPLATE = "(4) - flare_sector{0:02d}_model_{1:s}_peak_{2:10.10f}.png"
MISMATCH_FILENAME_TEMPLATE = "sector{0:02d}_peak_{1:10.10f}_{2:s}.png"
PARTITION_SIZE = 100
CUTOFF_PREDICTION = 0.5
CHART_DPI = 150

#TIC_TO_PLOT = 5656273
TIC_TO_PLOT = -1

class Curve:
    def __init__(self, cnn):
        self.cnn = cnn

    def load(self, fits_filename, overwrite = False):
        with fits.open(fits_filename) as hdul:
            header = hdul[0].header
        tic = header["TICID"]
        if (TIC_TO_PLOT > -1) and (not (tic == TIC_TO_PLOT)):
            return False
        self.fits_filename = fits_filename

        lc = lkio.read(self.fits_filename)
        return self.load_curve(lc, OUTPUT_FOLDER + FAST_PLOTS_FOLDER_NAME_TEMPLATE, overwrite = overwrite)

    def load_curve(self, lc, folder_name_template, overwrite = False):
        self.id = lc.meta.get("TICID")
        if (TIC_TO_PLOT > -1) and (not self.id == TIC_TO_PLOT):
            return False
        lc = lc.remove_nans().normalize()
        self.sector = int(lc.meta.get("SECTOR"))
        self.tstart = lc.meta.get("TSTART")
        self.tstop = lc.meta.get("TSTOP")
        self.charts_output_folder = folder_name_template.format(self.id)
        if overwrite or (not os.path.exists(self.charts_output_folder)):
            os.makedirs(self.charts_output_folder, exist_ok = True)
            self.flux = lc.flux.value
            self.flux_err = lc.flux_err.value
            self.times = lc.time.value
            return True
        else:
            return False

    def search_lightcurve(self, models):
        self.sd_flux = numpy.std(self.flux)
        self.min_flux = numpy.min(self.flux) - (self.sd_flux * 0.2)
        self.max_flux = numpy.max(self.flux) + (self.sd_flux * 0.2)

        preds = numpy.zeros((len(models), len(self.flux)))
        pred_errors = numpy.zeros((len(models), len(self.flux)))

        for j in range(len(models)):
            cnn.predict(modelname = models[j],
                    times = self.times,
                    fluxes = self.flux,
                    errs = self.flux_err)
            
            preds[j] = cnn.predictions[0]
            pred_errors[j] = cnn.predict_err[0]

        self.avg_pred = numpy.nanmedian(preds, axis=0)
        self.ff = stella.FitFlares(id = [self.id],
                        time = [self.times],
                        flux = [self.flux],
                        flux_err = [self.flux_err],
                        predictions = [self.avg_pred])
        self.ff.identify_flare_peaks(threshold = CUTOFF_PREDICTION)

    def plot_everything(self):
        self.plot_lightcurve()
        self.plot_fitted_flares()
        self.plot_probabilities()

    def plot_lightcurve(self):
        plt.style.use('default')

        fig, ax1 = plt.subplots()
        ax1.title.set_text("")
        ax1.set_xlabel("Time (BJD + 2457000)") 
        ax1.set_ylabel("Flux")
        ax1.set_ylim([self.min_flux, self.max_flux])
        ax1.plot(self.times, self.flux)

        for tpeak in self.ff.flare_table['tpeak']:
            plt.vlines(tpeak, 0,2, color='k', alpha=0.5, linewidth=5, zorder=0)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Flare Probability", color = COLOR_YELLOW)
        ax2.set_ylim([-1.05, 1.05])
        ax2.plot(self.times, self.avg_pred, color = COLOR_YELLOW)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
        plt.title(f"TIC {self.id} Sector {self.sector}")

        fig.tight_layout()
            
        figure = plt.gcf() # get current figure
        figure.set_size_inches(16, 8)

        plt.savefig(self.charts_output_folder + "/" + PREDICTIONS_FILENAME_TEMPLATE.format(self.id, MODEL_WIDTH, self.sector), dpi = CHART_DPI)
        plt.clf()
        plt.close("all")
        del fig, figure, ax1, ax2

        plt.style.use(astropy_mpl_style)

        fig, ax1 = plt.subplots()
        ax1.title.set_text("")
        ax1.set_xlabel("Time (BJD + 2457000)") 
        ax1.set_ylabel("Flux")
        ax1.set_ylim([self.min_flux, self.max_flux])
        ax1.plot(self.times, self.flux)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Error", color = COLOR_RED)
        ax2.plot(self.times, self.flux_err, color = COLOR_RED)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
        plt.title(f"TIC {self.id} Sector {self.sector}")

        fig.tight_layout()
            
        figure = plt.gcf() # get current figure
        figure.set_size_inches(16, 8)
        plt.savefig(self.charts_output_folder + "/" + OVERVIEW_FILENAME_TEMPLATE.format(self.id, self.sector), dpi = CHART_DPI)
        plt.clf()
        plt.close("all")
        del fig, figure, ax1, ax2    

    def plot_fitted_flares(self):
        flares = self.ff.flare_table
        for row in flares:
            tpeak = row["tpeak"]
            index_np = self.find_index_np(tpeak)
            if index_np != -1:
                fn = self.charts_output_folder + "/" + FLARE_FILENAME_TEMPLATE.format(self.sector, MODEL_WIDTH, tpeak)
                start = max(0, index_np - (PARTITION_SIZE // 2))
                end = min(len(self.times), index_np + (PARTITION_SIZE // 2))
                sub_pred = self.avg_pred[start:end]
                self.plot_flare(fn, start, end, sub_pred)

    def plot_probabilities(self):
        for k in range((len(self.times) // PARTITION_SIZE) + 1):
            fn = self.charts_output_folder + "/" + PREDICTION_FILENAME_TEMPLATE.format(self.sector, MODEL_WIDTH, k)
            start = k * PARTITION_SIZE
            if k * PARTITION_SIZE < len(self.times):
                end = (k + 1) * PARTITION_SIZE
            else:
                end = len(self.times)
            sub_pred = self.avg_pred[start:end]
            if any(value > CUTOFF_PREDICTION for value in sub_pred):
                self.plot_flare(fn, start, end, sub_pred)

    def plot_flare(self, fn, start, end, sub_pred, highlight = None):
        sub_time = self.times[start:end]
        sub_flux = self.flux[start:end]

        fig, ax1 = plt.subplots()
        ax1.title.set_text("")
        ax1.set_xlabel("Time (BJD + 2457000)") 
        ax1.set_ylabel("Flux")
        ax1.set_ylim([self.min_flux, self.max_flux])
        ax1.plot(sub_time, sub_flux)

        ax2 = ax1.twinx()
        ax2.set_ylim([-1.05, 1.05])
        ax2.set_ylabel("Flare Probability", color = COLOR_YELLOW)
        ax2.plot(sub_time, sub_pred, color = COLOR_YELLOW)
        if highlight is not None:
            plt.axvspan(highlight[0], highlight[1], alpha=0.2, color='red')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
        plt.title(f"TIC {self.id} Sector {self.sector}")

        fig.tight_layout()
                        
        figure = plt.gcf() # get current figure
        figure.set_size_inches(16, 8)
        plt.savefig(fn, dpi = CHART_DPI)
        plt.clf()
        plt.close("all")

        del fig, ax1, ax2, figure
        gc.collect()

    def find_index_np(self, x):
        index = numpy.searchsorted(self.times, x, side='right') - 1
        if 0 <= index < len(self.times) - 1 and self.times[index] <= x < self.times[index + 1]:
            return index
        return -1
    
def find_centered_index(timestamps, center_timestamp, spread_width = 100):
        # Convert the timestamp to a numpy array for easier manipulation
        timestamps = numpy.array(timestamps)
        # Find the index of the center timestamp
        center_index = numpy.searchsorted(timestamps, center_timestamp)
        # Determine the range of indices for the selected data
        return (max(0, center_index - spread_width), min(len(timestamps), center_index + spread_width + 1))

cnn = stella.ConvNN(output_dir = OUTPUT_FOLDER)
ds = stella.DownloadSets(tess_download_dir = OUTPUT_FOLDER)
ds.download_models()

short_models = ds.models
fast_models = [os.path.join(root, name)
          for root, dirs, files in os.walk(MODELS_FOLDER)
          for name in files
          if name.endswith((".h5"))]

fits_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(ROOT_PATH + "/var/flares/Fast - Sector 42/")
             for name in files
             if name.endswith((".fits"))]

def plot_prediction_mismatches(fast, short, flares):
    for row in flares:
        tpeak = row["tpeak"]
        
        range = find_centered_index(fast.times, tpeak)
        fn = fast.charts_output_folder + "/../" + MISMATCH_FILENAME_TEMPLATE.format(fast.sector, tpeak, "fast")
        sub_pred = fast.avg_pred[range[0]:range[1]]
        fast.plot_flare(fn, range[0], range[1], sub_pred)
        
        range = find_centered_index(short.times, tpeak)
        fn = short.charts_output_folder + "/../" + MISMATCH_FILENAME_TEMPLATE.format(short.sector, tpeak, "short")
        sub_pred = short.avg_pred[range[0]:range[1]]
        short.plot_flare(fn, range[0], range[1], sub_pred, (tpeak - 1.157407407407407e-3, tpeak + 1.157407407407407e-3))

def search_fits():
    thread = None
    for i in tqdm(range(len(fits_files)), position = 0):
        fits_filename = fits_files[i]
        if fits_filename.endswith((".fits")):
            fast_curve = Curve(cnn)
            if not fast_curve.load(fits_filename):
                continue
            fast_curve.search_lightcurve(fast_models)

            short_curve = Curve(cnn)
            lc = search_lightcurve(target = "tic" + str(fast_curve.id), mission = "TESS", sector = 42, exptime = 120)
            lc = lc.download().remove_nans().normalize()
            if short_curve.load_curve(lc, OUTPUT_FOLDER + SHORT_PLOTS_FOLDER_NAME_TEMPLATE):
                short_curve.search_lightcurve(short_models)
                # Extract the 'tpeak' columns as numpy arrays
                tpeak_F = numpy.array(fast_curve.ff.flare_table['tpeak'])
                tpeak_S = numpy.array(short_curve.ff.flare_table['tpeak'])
                
                # Find rows in table F that are not in table S based on 'tpeak'
                rows_not_in_S = fast_curve.ff.flare_table[~numpy.isclose(tpeak_F[:, None], tpeak_S, atol=0.1).any(axis=1)]
                if len(rows_not_in_S) == 0:
                    continue
                rows_not_in_S = unique(rows_not_in_S, 'tpeak')
                if thread is not None:
                    thread.join()
                thread = Thread(target = plot_prediction_mismatches, args = (fast_curve, short_curve, rows_not_in_S,))
                thread.start()

    if thread is not None:
        thread.join()

search_fits()
