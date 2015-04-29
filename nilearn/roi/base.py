import numpy as np
import nibabel
import traceback

from sklearn.base import BaseEstimator, TransformerMixin
from nilearn._utils.niimg_conversions import check_niimg

from .utils import is_gm


def _ratio_threshold(maps, ratio):
    raveled = np.abs(maps).ravel()
    argsort = np.argsort(raveled)
    n_voxels = int(ratio * maps[0].size)
    threshold = raveled[argsort[- n_voxels]]
    return threshold


class RegionAccumulator:
    def __init__(self, affine, min_size=10, n_regions=None, only_gm=False):
        self.min_size = min_size
        self.n_regions = n_regions
        self.regions = []
        self.index = []
        self.sizes = []
        self.gm_index = []
        self.only_gm = only_gm
        self.affine = affine

    def add(self, region, index):
        size = np.sum(region != 0.)
        is_gm_ = is_gm(region, self.affine)
        if size < self.min_size or (self.only_gm and not is_gm_):
            return
        regions_so_far = len(self.regions)
        if self.n_regions is None or regions_so_far < self.n_regions:
            self.regions.append(region)
            self.index.append(index)
            self.sizes.append(size)
            self.gm_index.append(is_gm_)
            return
        argmin = np.argmin(self.sizes)
        if size < self.sizes[argmin]:
            return
        self.regions[argmin] = region
        self.index[argmin] = index
        self.sizes[argmin] = size
        self.gm_index[argmin] = is_gm_


class BaseROIExtractor(BaseEstimator, TransformerMixin):

      
    def __init__(self, value_ratio=1., min_size=10, n_regions=None,
                 only_gm=False):
        self.value_ratio = value_ratio
        self.min_size = min_size
        self.n_regions = n_regions
        self.only_gm = only_gm

    def _compute_low_threshold(self, maps):
        return _ratio_threshold(maps, self.value_ratio)

    def _compute_regions(maps):
        raise NotImplementedError('This method should be subclassed')

    def _extract_regions_labels(self, maps, label_maps):
        
        accumulator = RegionAccumulator(self.affine, min_size=self.min_size,
                                        n_regions=self.n_regions,
                                        only_gm=self.only_gm)
        for i, (map_, label_map) in enumerate(zip(maps, label_maps)):            
            print 'processing map ' + str(i)
            if np.all(label_map == 0):
                continue
            bg = label_map[0, 0, 0]
            l_comps = np.unique(np.asarray(label_map))
            print 'l_comps'+ str(i)
            for label in l_comps:
                if label == bg:
                    continue
                region = (label_map == label) * map_
                accumulator.add(region, i)
        self.regions_ = accumulator.regions
        self.index_ = accumulator.index
        self.sizes_ = accumulator.sizes
        self.gm_index_ = accumulator.gm_index

    def _extract_regions_prob(self, maps, prob_maps_list):

        print 'Entered into extract regions prob'
        accumulator = RegionAccumulator(self.affine, min_size=self.min_size,
                                        n_regions=self.n_regions,
                                        only_gm=self.only_gm)
        for i, (map_, prob_maps) in enumerate(zip(maps, prob_maps_list)):
            print 'processing map ' + str(i)
            for prob_map in prob_maps:
                if np.all(prob_map == 0):
                    continue
                region = prob_map * map_
                accumulator.add(region, i)
        self.regions_ = accumulator.regions
        self.index_ = accumulator.index
        self.sizes_ = accumulator.sizes
        self.gm_index_ = accumulator.gm_index

    def fit(self, maps_img):

        # Load the maps
        maps_img = check_niimg(maps_img)
        maps = maps_img.get_data()
        print 'fit'
        # For automatic random walker
        self.affine = maps_img.get_affine()

        # Put the map index in the first dimension for convenience
        maps = np.rollaxis(maps.view(), 3)

        if self.value_ratio is not None:
            low_threshold = self._compute_low_threshold(maps)
            maps[np.abs(maps) < low_threshold] = 0.

        regions = self._compute_regions(maps)
        if isinstance(regions[0], list):
            self._extract_regions_prob(maps, regions)
        else:
            self._extract_regions_labels(maps, regions)

        if len(self.regions_) > 0:
            regions = np.rollaxis(np.asarray(self.regions_), 0, 4)
        else:
            regions = np.empty(maps_img.shape[:3] + (0,))

        self.regions_img_ = \
                nibabel.Nifti1Image(regions, maps_img.get_affine())
        return self
