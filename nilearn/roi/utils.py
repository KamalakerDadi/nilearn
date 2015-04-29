from scipy import linalg
import numpy as np

from .. import datasets
from ..image.resampling import resample_img


ref_atlas = None


def hard_assignment(maps, normalize=False):
    if normalize:
        for m in maps:
            m /= np.max(np.abs(m))
    # Add a map with low values to absorb other maps '0'
    bg = np.empty(maps[0].shape)
    bg.fill(0)
    # Take absolute value, implicit copy
    maps = np.append([bg], np.abs(maps), axis=0)
    return np.argmax(maps, axis=0) - 1


def is_gm(region_data, affine, acceptance_ratio=.5, gm_map=None):
    if gm_map is None:
        global ref_atlas
        if ref_atlas is None:
            ref_atlas = datasets.fetch_icbm152_2009()['gm']
        gm_map = ref_atlas
    gm_map = resample_img(gm_map, target_affine=affine,
            target_shape=region_data.shape).get_data()
    return ((gm_map * region_data).sum() >=
            (region_data.sum() * acceptance_ratio))


def learn_time_series(data, maps):
    if maps.shape[0] == 0:
        return np.ndarray((data.shape[0], 0))
    serie = linalg.lstsq(maps.T, data.T)[0]
    return serie.T
