import os, sys, shutil, time
ROOT_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../"
WORKING_DIR = ROOT_PATH + "/var"
print("Initializing environment...", end = "", flush = True)
sys.path.insert(1, ROOT_PATH)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("done")

print("Importing packages and initializing TensorFlow...", end = "", flush = True)
import stella
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20

print("done")

##### SETUP

# short = 2 minutes
# fast  = 20 second
#flares = "short"
flares = "fast"

num_flares = 0
#num_flares = 3792

download_required = False
#download_required = True

epochs = 400
#####

if flares == "short":
    fcn = None
    time_correction = 2457000.0
    sectors = [1, 2]
else:
    fcn = ROOT_PATH + "/fast_flares.csv"
    time_correction = 0.0
    sectors = [27,28,29,30,31,32,33,34,35,36,37,38,39]

WORKING_DIR = WORKING_DIR + "/" + flares + "-"

if sectors != None:
    WORKING_DIR = WORKING_DIR + "[" + ",".join(str(s) for s in sectors) + "]"
else:
    WORKING_DIR = WORKING_DIR + "ALL SECTORS"
WORKING_DIR = WORKING_DIR + "-"
if num_flares != 0:
    WORKING_DIR = WORKING_DIR + str(num_flares)
else:
    WORKING_DIR = WORKING_DIR + "ALL"

if download_required:
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)

print("Downloading catalog...", end = "", flush = True)
download = stella.DownloadSets(ROOT_PATH, fn_dir = WORKING_DIR, flare_catalog_name = fcn)
if flares == "short":
    download.download_catalog()
else:
    download.load_flares()
print("done")

print("Downloading light curves...")
if num_flares != 0:
    download.flare_table = download.flare_table[0:num_flares]

if download_required:
    download.download_lightcurves(remove_fits = False, exptime = flares, sector = sectors)

c = 200
time.sleep(15)  
ds = stella.FlareDataSet(downloadSet = download, time_offset = time_correction, cadences=200)
cnn = stella.ConvNN(output_dir = WORKING_DIR, ds = ds)
cnn.train_models(seeds = [1, 2, 3, 5, 8, 13, 21], epochs = epochs, save = True, custom_name = "_cadences=" + str(c))

#ind_pc = np.where(ds.train_labels == 1)[0]
#ind_nc = np.where(ds.train_labels == 0)[0]

#positive_cadences = np.hstack(np.hstack(ds.train_data[ind_pc[10]]))
#negative_cadences = np.hstack(np.hstack(ds.train_data[ind_nc[10]]))

#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,3), 
#                               sharex=True, sharey=True)
#ax1.plot(positive_cadences, 'r')
#ax1.set_title('Flare')
#ax1.set_xlabel('Cadences')
#ax2.plot(negative_cadences, 'k')
#ax2.set_title('No Flare')
#ax2.set_xlabel('Cadences')
#plt.show()

# Create a 2x2 grid of subplots
plt.figure(figsize=(12, 8))

# Subplot 1: Scatter plot (occupies the entire first row)
plt.subplot(212)
plt.scatter(cnn.val_pred_table['tpeak'], cnn.val_pred_table['pred_s0002'],
            c=cnn.val_pred_table['gt'], vmin=0, vmax=1)
plt.xlabel('Tpeak [BJD - 2457000]')
plt.ylabel('Probability of Flare')
plt.colorbar(label='Ground Truth')
plt.title('Scatter Plot')

# Subplot 2: Training and validation loss (first column of the second row)
plt.subplot(221)
plt.plot(cnn.history_table['loss_s0002'], 'k', label='Training', lw=3)
plt.plot(cnn.history_table['val_loss_s0002'], 'darkorange', label='Validation', lw=3)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Plot')

# Subplot 3: Accuracy plot (second column of the second row)
plt.subplot(222)
plt.plot(cnn.history_table['accuracy_s0002'], 'k', label='Training', lw=3)
plt.plot(cnn.history_table['val_accuracy_s0002'], 'darkorange', label='Validation', lw=3)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Plot')

# Adjust spacing between subplots
plt.tight_layout()

# Show the grid of subplots
plt.savefig(WORKING_DIR + str(c) + "-" +str(epochs) + ".png")
