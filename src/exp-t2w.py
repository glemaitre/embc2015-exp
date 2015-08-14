# In order to plot some stuff
import matplotlib.pyplot as plt
# In order to manipulate some stuff
import numpy as np
# In order to classifiy some stuff
### Random forest
from sklearn.ensemble import RandomForestClassifier
# In order to quantify the performances of our stuff
### ROC Curve
from sklearn.metrics import roc_curve
### AUC Curve
from sklearn.metrics import roc_auc_score
# Confusion matrix
from sklearn.metrics import confusion_matrix

# Define the function to compute the Normalised Mean Intensity
def nmi(data):
    # get the minimum 
    #min_data = np.min(data)
    min_data = -1.
    print 'mini: {}'.format(min_data)

    # get the maximum
    #max_data = np.max(data)
    max_data = 1.
    print 'maxi: {}'.format(max_data)

    # find the mean
    mean_data = np.mean(data)
    print 'mean: {}'.format(mean_data)

    # return the nmi
    return mean_data / (max_data - min_data)

# Load the data file from the numpy npz file
data_norm_rician = np.load('../data/clean/data_norm_rician.npy')
data_norm_gaussian = np.load('../data/clean/data_norm_gaussian.npy')
data_t2w_norm = np.load('../data/clean/data_raw_norm.npy')
label = np.load('../data/clean/label.npy')
patient_sizes = np.load('../data/clean/patient_sizes.npy')

print '---> Data loaded'
fig, axes = plt.subplots(nrows=3, ncols=2)
# Make the classification for each patient
global_hist_t2w = np.zeros((200,))
global_norm_gaussian = np.zeros((200,))
global_norm_rician = np.zeros((200,))
global_hist_t2w_cap = np.zeros((200,))
global_norm_gaussian_cap = np.zeros((200,))
global_norm_rician_cap = np.zeros((200,))

# Initialise the array in order to store the value of the nmi
nmi_raw = []
nmi_gaussian = []
nmi_rician = []
for pt in xrange(len(patient_sizes)):
    
    # Find the index of the current patients
    if (pt == 0):
        start_idx = 0
        end_idx = patient_sizes[pt]
    else:
        start_idx = np.sum(patient_sizes[0 : pt])
        end_idx = np.sum(patient_sizes[0 : pt + 1])

    ##### RAW DATA #####
    # Compute the histogram for the whole data
    nb_bins = 200
    hist, bin_edges = np.histogram(data_t2w_norm[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[0, 0].plot(bin_edges[0 : -1], hist, label='Patient '+str(pt))
    global_hist_t2w = np.add(global_hist_t2w, hist)

    # Compute the NMI for the raw data
    nmi_raw.append(nmi(data_t2w_norm[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = 200
    sub_data = data_t2w_norm[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[0, 1].plot(bin_edges[0 : -1], hist)
    global_hist_t2w_cap = np.add(global_hist_t2w_cap, hist)

    ##### GAUSSIAN NORMALISATION #####
    # Compute the histogram for the whole data
    nb_bins = 200
    hist, bin_edges = np.histogram(data_norm_gaussian[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[1, 0].plot(bin_edges[0 : -1], hist)    
    global_norm_gaussian = np.add(global_norm_gaussian, hist)

    # Compute the NMI for the gaussian data
    nmi_gaussian.append(nmi(data_norm_gaussian[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = 200
    sub_data = data_norm_gaussian[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[1, 1].plot(bin_edges[0 : -1], hist)
    global_norm_gaussian_cap = np.add(global_norm_gaussian_cap, hist)

    ##### RICIAN NORMALISATION #####
    # Compute the histogram for the whole data
    nb_bins = 200
    hist, bin_edges = np.histogram(data_norm_rician[start_idx : end_idx], bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[2, 0].plot(bin_edges[0 : -1], hist)
    global_norm_rician = np.add(global_norm_rician, hist)

    # Compute the NMI for the rician data
    nmi_rician.append(nmi(data_norm_rician[start_idx : end_idx]))

    # Compute the histogram for the cancer data
    nb_bins = 200
    sub_data = data_norm_rician[start_idx : end_idx]
    cap_data = sub_data[np.nonzero(label[start_idx : end_idx] == 1)[0]]
    hist, bin_edges = np.histogram(cap_data, bins=nb_bins, range=(-1., 1.), density=True)
    hist = np.divide(hist, np.sum(hist))
    axes[2, 1].plot(bin_edges[0 : -1], hist)
    global_norm_rician_cap = np.add(global_norm_rician_cap, hist)

# Decorate the plot with the proper annotations
axes[0, 0].set_ylabel('Probabilities')
axes[0, 0].set_title('PDFs for non-normalise data (CaP + healthy)')

axes[0, 1].set_ylabel('Probabilities')
axes[0, 1].set_title('PDFs for non-normalise data (CaP)')

axes[1, 0].set_ylabel('Probabilities')
axes[1, 0].set_title('PDFs for Gaussian normalise data (CaP + healthy)')

axes[1, 1].set_ylabel('Probabilities')
axes[1, 1].set_title('PDFs for Gaussian normalise data (CaP)')

axes[2, 0].set_ylabel('Probabilities')
axes[2, 0].set_title('PDFs for Rician normalise data (CaP + healthy)')

axes[2, 1].set_ylabel('Probabilities')
axes[2, 1].set_title('PDFs for Rician normalise data (CaP)')

# Normalise the histogram
global_hist_t2w = np.divide(global_hist_t2w, float(len(patient_sizes)))
global_norm_gaussian = np.divide(global_norm_gaussian, float(len(patient_sizes)))
global_norm_rician = np.divide(global_norm_rician, float(len(patient_sizes)))

global_hist_t2w_cap = np.divide(global_hist_t2w_cap, float(len(patient_sizes)))
global_norm_gaussian_cap = np.divide(global_norm_gaussian_cap, float(len(patient_sizes)))
global_norm_rician_cap = np.divide(global_norm_rician_cap, float(len(patient_sizes)))

# Compute the entropy
from scipy.stats import entropy

plt.figure()
plt.plot(bin_edges[0 : -1], global_hist_t2w, label="No norm - Ent=" + str(entropy(global_hist_t2w)))
plt.plot(bin_edges[0 : -1], global_norm_gaussian, label="Gaussian norm - Ent=" + str(entropy(global_norm_gaussian)))
plt.plot(bin_edges[0 : -1], global_norm_rician, label="Rician norm - Ent=" + str(entropy(global_norm_rician)))
plt.title('Accumulation of the PDFs for prostate')
plt.legend()
#plt.show()

plt.figure()
plt.plot(bin_edges[0 : -1], global_hist_t2w_cap, label="No norm - Ent=" + str(entropy(global_hist_t2w_cap)))
plt.plot(bin_edges[0 : -1], global_norm_gaussian_cap, label="Gaussian norm - Ent=" + str(entropy(global_norm_gaussian_cap)))
plt.plot(bin_edges[0 : -1], global_norm_rician_cap, label="Rician norm - Ent=" + str(entropy(global_norm_rician_cap)))
plt.title('Accumulation of the PDFs for CaP')
plt.legend()
#plt.show()

print 'Ratio entropy prostate vs cap'
print 'Raw data: {}'.format(entropy(global_hist_t2w_cap) / entropy(global_hist_t2w))
print 'Gaussian Normalisation: {}'.format(entropy(global_norm_gaussian_cap) / entropy(global_norm_gaussian))
print 'Rician Normalisation: {}'.format(entropy(global_norm_rician_cap) / entropy(global_norm_rician))

print ''
print 'Information regarding the NMI'
print 'std of raw NMI: {}'.format(np.std(nmi_raw))
print 'std of gaussian NMI: {}'.format(np.std(nmi_gaussian))
print 'std of rician NMI: {}'.format(np.std(nmi_rician))

# fig, axes = plt.subplots(nrows=1, ncols=3)
# axes[0, 0].plot(bin_edges[0 : -1], global_hist_t2w)
# axes[0, 1].plot(bin_edges[0 : -1], global_norm_gaussian)
# axes[0, 2].plot(bin_edges[0 : -1], global_norm_rician)
# plt.show()
