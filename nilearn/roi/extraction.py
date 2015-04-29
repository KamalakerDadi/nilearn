import numpy as np
from scipy.ndimage.measurements import label, maximum
from ..masking import _load_mask_img
from ..image.image import _smooth_array
from .._utils.niimg_conversions import check_niimg

from .base import BaseROIExtractor, _ratio_threshold
from .utils import hard_assignment
from .utils import is_gm
from ..external.skimage import random_walker


class NoopExtractor(BaseROIExtractor):

    def __init__(self):
        pass

    def fit(self, maps_img):
        maps_img = check_niimg(maps_img)
        data = maps_img.get_data()
        self.regions_img_ = maps_img
        self.index_ = range(data.shape[3])
        self.gm_index_ = [is_gm(data[..., i], maps_img.get_affine())
                          for i in range(data.shape[3])]
        return self


class HardExtractor(BaseROIExtractor):
    """ Make a hard assignment of the regions and take the biggest connected
        components
    """
    print 'Entered into HardExtractor' 
    def __init__(self, n_regions=None, only_gm=False):
        super(HardExtractor, self).__init__(
                value_ratio=None, min_size=10,
                n_regions=n_regions, only_gm=only_gm)
          

    def _compute_regions(self, maps):
        # Set maps to unit variance
        # Take the hard assignment
        print 'Before hard assignment'
        hard = hard_assignment(maps)
        print 'After hard assignment'
        label_maps = np.zeros_like(maps, dtype=int)
        for i, map_ in enumerate(maps):
            # Take the hard regions
            map_[hard != i] = 0.
            # Take the connected components
            labels, n = label(map_)
            label_maps[i] = labels
          
	print 'End of hard extraction'
         
        # Sorting and region selection will be done by the base extractor
        return label_maps
        


class VoxelRatioExtractor(BaseROIExtractor):

    def _compute_regions(self, maps):
        # Maps are already thresholded by the first step. We just have to
        # extract connected components.
        label_maps = []
        for map_ in maps:
            if np.all(map_ == 0):
                continue
            label_maps.append(label(map_)[0])
        return label_maps


class HystheresisExtractor(BaseROIExtractor):

    def __init__(self, seed_ratio, value_ratio=1., min_size=1, n_regions=None,
                 only_gm=False):
        super(HystheresisExtractor, self).__init__(
                value_ratio=value_ratio, min_size=min_size,
                n_regions=n_regions, only_gm=only_gm)
        self.seed_ratio = seed_ratio

    def _compute_regions(self, maps):
        label_maps = []
        seed_thr = _ratio_threshold(maps, self.seed_ratio)
        for map_ in maps:
            if np.all(map_ == 0):
                continue

            # Hysteresis operates on positive images. We switch negative
            # components to positive
            map_ = np.abs(map_)

            # Extraction of connected components
            label_map, n_labels = label(map_)

            if n_labels == 0:
                # No regions, we keep the map for later indexing purpose
                label_maps.append(label_map)
                continue

            # Extract maximum of each region
            maxima = maximum(map_, labels=label_map,
                             index=range(1, n_labels + 1))

            # Nullify regions which maximum is inferior to the seed ratio
            for i in range(1, n_labels + 1):
                if maxima[i - 1] < seed_thr:
                    label_map[label_map == i] = 0
            label_maps.append(label_map)
        return label_maps


def find_poi(data, affine, smooth=10.):
    from skimage.feature.peak import peak_local_max
    # Smooth the data to remove meaningless local maxima
    markers_ind = _smooth_array(data, affine, fwhm=smooth)
    markers_ind = peak_local_max(markers_ind, exclude_border=False,
            min_distance=6)
    return markers_ind


class RandomWalkerExtractor(BaseROIExtractor):

    
    def __init__(self, seed_ratio, value_ratio=1., min_size=1, n_regions=None,
                 only_gm=False):
        super(RandomWalkerExtractor, self).__init__(
                value_ratio=value_ratio, min_size=min_size,
                n_regions=n_regions, only_gm=only_gm)
        self.seed_ratio = seed_ratio

    def _compute_regions(self, maps):
        label_maps = []
        if isinstance(self.seed_ratio, float):
            seed_thr = _ratio_threshold(maps, self.seed_ratio)
        if self.seed_ratio == 'hard':
            # Use hard thresholding to init seeds
            ht = hard_assignment(maps)

        for i, map_ in enumerate(maps):
            if np.all(map_ == 0):
                continue

            # RW operates on positive images. We switch negative
            # components to positive
            map_ = np.abs(map_)

            if self.seed_ratio == 'auto':
                seeds_indices = find_poi(map_, self.affine)
                seeds = np.zeros(map_.shape)
                # Mark seeds
                for coord in seeds_indices:
                    seeds[tuple(coord)] = 1
            elif self.seed_ratio == 'hard':
                seeds = (ht == i)
            elif isinstance(self.seed_ratio, float):
                seeds = (map_ >= seed_thr)
            else:
                raise ValueError("Seed ratio must be 'auto', 'hard', or float")
            if not np.any(seeds):
                map_.fill(0)
                label_maps.append(map_)
                continue
            seeds, n_blobs = label(seeds)

            # Problem here: if using 'auto', some components of maps may not
            # have a seed, this make RW fail. We tag as '-1' all components
            # that have no seed

            components, nc = label(map_)
            seeded_comp = np.unique(components[seeds > 0])
            for c in range(nc):
                if not c in seeded_comp:
                    map_[components == c] = 0.

            seeds[map_ == 0.] = -1

            # Random Walker needs value to be between -1 and 1
            max_ = np.max(map_)
            if max_ > 1.:
                map_ /= max_

            try:
                label_map = random_walker(map_, seeds, mode='cg_mg',
                        spacing=np.diag(self.affine)[:3])

            except Exception as e:
                print 'Random Walker failed for map %d: %s' % (i, str(e))
                map_.fill(0)
                label_maps.append(map_)
                continue
            label_maps.append(label_map)
        return label_maps


class BlobExtractor(BaseROIExtractor):

    def __init__(self, mask, value_ratio=1., min_size=1, n_regions=None,
                 only_gm=False):
        super(BlobExtractor, self).__init__(
                value_ratio=value_ratio, min_size=min_size,
                n_regions=n_regions, only_gm=only_gm)
        self.mask = mask

    def _compute_regions(self, maps):
        try:
            import nipy.labs.spatial_models.hroi as hroi
            from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image
        except:
            return []
        
        mask_img = check_niimg(self.mask)
        mask = _load_mask_img(mask_img)[0]
        domain = grid_domain_from_image(mask_img)
        if self.value_ratio is not None:
            thr = _ratio_threshold(maps, self.value_ratio)
        label_maps = []
        for i, map_ in enumerate(maps):
            nroi = hroi.HROI_as_discrete_domain_blobs(domain,
                    map_[mask], threshold=thr, smin=self.min_size)
            nroi.reduce_to_leaves()

            descrip = "blob image extracted from map %d" % i
            wim = nroi.to_image('id', roi=True, descrip=descrip)
            label_maps = wim.get_data()
        return label_maps
