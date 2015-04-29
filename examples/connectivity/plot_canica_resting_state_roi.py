"""
Group analysis of resting-state fMRI with ICA: CanICA
=====================================================

An example applying CanICA to resting-state data. This example applies it
to 40 subjects of the ADHD200 datasets.

CanICA is an ICA method for group-level analysis of fMRI data. Compared
to other strategies, it brings a well-controlled group model, as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal. The reference papers are:

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

Pre-prints for both papers are available on hal
(http://hal.archives-ouvertes.fr)
"""

import numpy as np
import nibabel
from nilearn import datasets
from nilearn.roi.extraction import HardExtractor
from nilearn.roi.extraction import RandomWalkerExtractor

### Load ADHD rest dataset ####################################################

adhd_dataset = datasets.fetch_adhd()
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

### Apply CanICA ##############################################################
from nilearn.decomposition.canica import CanICA

#n_components = 20
#canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
               # memory="nilearn_cache", memory_level=5,
                #threshold=3., verbose=10, random_state=0)
#canica.fit(func_filenames)

# Retrieve the independent components in brain space
#components_img = canica.masker_.inverse_transform(canica.components_)
#print 'components_img', components_img
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
#components_img.to_filename('canica_resting_state.nii.gz')

#### Region of Interest Extraction #############################\
ext = HardExtractor(n_regions=10)
print 'finished Hard Extractor'
RWE = RandomWalkerExtractor(seed_ratio=1.5, n_regions=10)
print 'finished RWE Extractor'
RWE.fit('canica_resting_state.nii.gz')
ext.fit('canica_resting_state.nii.gz')
roi_maps = ext.regions_img_
print 'roi_maps', roi_maps 
RWE_maps = RWE.regions_img_
print 'RWE_maps', RWE_maps
#print 'roi_maps_brainspace', roi_maps_brainspace

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_roi
from nilearn.image import iter_img

for i, cur_img in enumerate(iter_img(roi_maps,RWE_maps)):
    plot_roi(cur_img, display_mode="z", title="IC %d" % i, cut_coords=1,
		 colorbar=False)
plt.show()


