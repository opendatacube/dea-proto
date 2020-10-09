import math
import re
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Iterable
from uuid import UUID

from datacube.utils.geometry import Geometry
from odc.index import odc_uuid


Document = Dict[str, Any]

KNOWN_CONSTELLATIONS = [
    'sentinel-2'
]

LANDSAT_PLATFORMS = [
    'landsat-5', 'landsat-7', 'landsat-8'
]

# Mapping between EO3 field names and STAC properties object field names
MAPPING_STAC_TO_EO3 = {
    "end_datetime": "dtr:end_datetime",
    "start_datetime": "dtr:start_datetime",
    "gsd": "eo:gsd",
    "instruments": "eo:instrument",
    "platform": "eo:platform",
    "constellation": "eo:constellation",
    "view:off_nadir": "eo:off_nadir",
    "view:azimuth": "eo:azimuth",
    "view:sun_azimuth": "eo:sun_azimuth",
    "view:sun_elevation": "eo:sun_elevation",
}


def _stac_product_lookup(item: Document) -> Tuple[Optional[str], str, Optional[str], str]:
    properties = item['properties']

    product_label = None
    product_name = properties.pop('odc:product', None) or properties['platform']
    region_code = None
    default_grid = None

    # Maybe this should be the default product_name
    constellation = properties.get('constellation')

    if constellation in KNOWN_CONSTELLATIONS:
        if constellation == 'sentinel-2':
            product_label = properties['sentinel:product_id']
            product_name = 's2_l2a'
            region_code = '{}{}{}'.format(
                str(properties['proj:epsg'])[-2:],
                properties['sentinel:latitude_band'],
                properties['sentinel:grid_square']
            )
            default_grid = "g10m"
    elif properties.get('platform') in LANDSAT_PLATFORMS:
        self_href = _find_self_href(item)
        product_label = Path(self_href).stem.replace(".stac-item", "")
        region_code = properties.get('odc:region_code')
        default_grid = "g30m"

    # On many stac documents the 'id' is a usable label.
    # (But we don't want to use it if it's only an unreadable uuid)
    if not product_label:
        if not _looks_like_an_id(item['id']):
            product_label = item['id']

    return product_label, product_name, region_code, default_grid


def _find_self_href(item: Document) -> str:
    """
    Extracting product label from filename of the STAC document 'self' URL
    """
    self_uri = [link.get('href', '')
                for link in item.get("links", [])
                if link.get('rel') == 'self']

    if len(self_uri) < 1:
        raise ValueError("Can't find link for 'self'")
    if len(self_uri) > 1:
        raise ValueError("Too many links to 'self'")
    return self_uri[0]


def _get_stac_bands(item: Document,
                    default_grid: str,
                    relative: bool = False) -> Tuple[Document, Document]:
    bands = {}
    grids = {}
    assets = item.get('assets', {})

    for asset_name, asset in assets.items():
        # Ignore items that are not actual COGs/geotiff
        if asset.get('type') not in ['image/tiff; application=geotiff; profile=cloud-optimized',
                                     'image/tiff; application=geotiff']:
            continue

        transform = asset.get('proj:transform')
        grid = f'g{transform[0]:g}m'

        if grid not in grids:
            grids[grid] = {
                'shape': asset.get('proj:shape'),
                'transform': asset.get('proj:transform')
            }

        path = asset['href']
        if relative:
            path = Path(path).name

        band_info = {
            'path': path
        }

        if grid != default_grid:
            band_info['grid'] = grid

        bands[asset_name] = band_info

    # If we don't specify a default grid, label the first grid 'default'
    if not default_grid:
        default_grid = list(grids.keys())[0]

    if default_grid in grids:
        grids['default'] = grids.pop(default_grid)

    return bands, grids


def _geographic_to_projected(geometry, crs):
    """ Transform from WGS84 to the target projection, assuming Lon, Lat order
    """

    geom = Geometry(geometry, 'EPSG:4326')
    geom = geom.to_crs(crs, resolution=math.inf)

    if geom.is_valid:
        return geom.json
    else:
        return None


def stac_transform_absolute(input_stac):
    return stac_transform(input_stac, relative=False)


def _convert_value_to_eo3_type(key: str, value):
    """
    Convert return type as per EO3 specification.
    Return type is String for "instrument" field in EO3 metadata.

    """
    if key == "instruments":
        return _stac_to_eo3_instruments(value)
    if key == 'platform':
        return _normalise_platform(value)

    return value


def _normalise_platform(s: str) -> str:
    """
    EO3's eo:platform field is formatted according to stac examples circa version 0.6.

    Stac doccuments on the web use a few different cases and separators, but ODC's matching
    is case-sensitive and exact, so we want need to be consisent.

    >>> _normalise_platform('LANDSAT_8')
    'landsat-8'
    """
    return s.lower().replace("_", "-")


def _stac_to_eo3_instruments(value: Iterable[str]) -> str:
    """
    EO3's eo:instrument field follows the stac examples circa stac version 0.6.

    For landsat, this matches the exact instrument formatting in USGS's MTL files.

    >>> _stac_to_eo3_instruments(['tm'])
    'TM'
    >>> _stac_to_eo3_instruments(['oli'])
    'OLI'
    >>> _stac_to_eo3_instruments(['tirs', 'oli'])
    'OLI_TIRS'
    """
    return '_'.join(sorted(v.strip().upper() for v in value))


def _get_stac_properties_lineage(input_stac: Document) -> Tuple[Document, Any]:
    """
    Extract properties and lineage field
    """
    properties = input_stac['properties']
    prop = {MAPPING_STAC_TO_EO3.get(key, key): _convert_value_to_eo3_type(key, val)
            for key, val in properties.items()}
    if prop.get('odc:processing_datetime') is None:
        prop['odc:processing_datetime'] = (
            # Stac's 'created' property.
            prop.pop('created', None) or
            # TODO: This is not ideal. Perhaps the file ctime?
            properties['datetime'].replace("000+00:00", "Z")
        )
    if prop.get('odc:file_format') is None:
        prop['odc:file_format'] = 'GeoTIFF'

    # Extract lineage
    lineage = prop.pop('odc:lineage', None)

    return prop, lineage


def _check_valid_uuid(uuid_string: str) -> bool:
    """
    Check if provided uuid string is a valid UUID.
    """
    try:
        UUID(str(uuid_string))
        return True
    except ValueError:
        return False


def stac_transform(input_stac: Document, relative: bool = True) -> Document:
    """ Takes in a raw STAC 1.0 dictionary and returns an ODC dictionary
    """

    product_label, product_name, region_code, default_grid = _stac_product_lookup(input_stac)

    # Generating UUID for products not having UUID.
    # Checking if provided id is valid UUID.
    # If not valid, creating new deterministic uuid using odc_uuid function based on product_name and product_label.
    # TODO: Verify if this approach to create UUID is valid.
    if _check_valid_uuid(input_stac["id"]):
        deterministic_uuid = input_stac["id"]
    else:
        stable_identifier = product_label or input_stac['id']

        if product_name in ["s2_l2a"]:
            deterministic_uuid = str(odc_uuid("sentinel-2_stac_process", "1.0.0", [stable_identifier]))
        else:
            deterministic_uuid = str(odc_uuid(f"{product_name}_stac_process", "1.0.0", [stable_identifier]))

    bands, grids = _get_stac_bands(input_stac, default_grid, relative=relative)

    stac_properties, lineage = _get_stac_properties_lineage(input_stac)

    properties = input_stac['properties']
    epsg = properties['proj:epsg']
    native_crs = f"epsg:{epsg}"

    geometry = _geographic_to_projected(input_stac['geometry'], native_crs)

    product_label = product_label or input_stac.pop('title', None)

    optional_properties = {}
    if product_label:
        optional_properties['label'] = product_label

    stac_odc = {
        '$schema': 'https://schemas.opendatacube.org/dataset',
        'id': deterministic_uuid,
        **optional_properties,

        'crs': native_crs,
        'grids': grids,
        'product': {
            'name': product_name.lower()
        },
        'properties': stac_properties,
        'measurements': bands,
        'lineage': {}
    }

    if region_code:
        stac_odc['properties']['odc:region_code'] = region_code

    if geometry:
        stac_odc['geometry'] = geometry

    if lineage:
        stac_odc['lineage'] = lineage

    return stac_odc


def _looks_like_an_id(s: str) -> bool:
    """
    Is this purely numeric or a uuid?

    >>> _looks_like_an_id('123')
    True
    >>> _looks_like_an_id('23d8a399-137d-45b3-9ba1-fd610bca2a13')
    True
    >>> _looks_like_an_id('DA6E7559-8957-482C-B389-4A90263655D0')
    True
    >>> _looks_like_an_id('ga_ls5t_ard_3-1-20200605_113081_1988-03-30_final')
    False
    >>> _looks_like_an_id('LT05_L1TP_113081_19880330_20170209_01_T1')
    False
    >>> _looks_like_an_id('beef-feed-34')
    False
    """
    # Numeric?
    if re.fullmatch(r'[0-9]+', s) is not None:
        return True

    # UUID?
    if re.fullmatch(r'[0-9A-Fa-f-]+(-[0-9A-Fa-f]+){4}', s) is not None:
        return True

    return False
