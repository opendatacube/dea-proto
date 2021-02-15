"""
Wofs Summary
"""
from typing import Optional, Tuple
import xarray as xr
from odc.stats.model import Task
from odc.algo.io import load_with_native_transform
from odc.algo import safe_div, apply_numexpr
from .model import StatsPluginInterface
from . import _plugins


class StatsWofs(StatsPluginInterface):
    NAME = "ga_ls_wo_summary"
    SHORT_NAME = NAME
    VERSION = '0.0.0'
    PRODUCT_FAMILY = "wo_summary"

    def __init__(
        self, resampling: str = "bilinear",
    ):
        self.resampling = resampling

    @property
    def measurements(self) -> Tuple[str, ...]:
        return ("count_wet", "count_clear", "frequency")

    def _native_tr(self, xx):
        wet = xx.water == 128
        dry = xx.water == 0
        # Cloud is
        #  - valid observation bits 0,1 == 0
        #  - is cloud shadow or cloud (one of bits 5,6 is non-zero)
        cloud = apply_numexpr("((water%4)==0) & ((water>>5)%4 > 0)", xx, dtype="bool")
        cloud.attrs.pop("nodata", None)
        return xr.Dataset(dict(wet=wet, dry=dry, cloud=cloud))

    def _fuser(self, xx):
        from odc.algo._masking import _or_fuser

        # Merge everything with OR first
        xx = xx.map(_or_fuser)

        # Ensure all 3 bits are exclusive
        #  cloud=T, wet=?, dry=? => (wet'=F  , dry'=F)
        #  cloud=F, wet=T, dry=T => (wet'=F  , dry'=F)
        #  else                  => (wet'=wet, dry'=dry)
        wet = apply_numexpr("wet & (~dry) & (~cloud)", xx, dtype="bool")
        dry = apply_numexpr("dry & (~wet) & (~cloud)", xx, dtype="bool")

        return xr.Dataset(dict(wet=wet, dry=dry, cloud=xx.cloud))

    def input_data(self, task: Task) -> xr.Dataset:
        chunks = {"y": -1, "x": -1}
        groupby = "solar_day"

        xx = load_with_native_transform(
            task.datasets,
            bands=["water"],
            geobox=task.geobox,
            native_transform=self._native_tr,
            fuser=self._fuser,
            groupby=groupby,
            resampling=self.resampling,
            chunks=chunks,
        )

        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        count_wet = xx.wet.sum(axis=0, dtype="uint16")
        count_dry = xx.dry.sum(axis=0, dtype="uint16")
        count_clear = count_wet + count_dry
        frequency = safe_div(count_wet, count_clear, dtype="float32")

        return xr.Dataset(
            dict(count_wet=count_wet, count_clear=count_clear, frequency=frequency)
        )

    def rgba(self, xx: xr.Dataset) -> Optional[xr.DataArray]:
        return None


_plugins.register("wofs-summary", StatsWofs)
