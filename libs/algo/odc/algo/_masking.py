""" Mostly masking related, also converting between float[with nans] and int[with nodata]

"""

from typing import Dict, Tuple, Any, Iterable, Union
import numpy as np
import xarray as xr
import dask
import dask.array as da
import numexpr as ne
from ._dask import randomize


def default_nodata(dtype):
    """ Default `nodata` for a given dtype
        - nan for float{*}
        - 0   for any other type
    """
    if dtype.kind == 'f':
        return dtype.type(np.nan)
    return dtype.type(0)


def keep_good_np(xx, where, nodata):
    yy = np.full_like(xx, nodata)
    np.copyto(yy, xx, where=where)
    return yy


def keep_good_only(x, where,
                   inplace=False,
                   nodata=None):
    """ Return a copy of x, but with some pixels replaced with `nodata`.

    This function can work on dask arrays, in which case output will be a dask array as well.

    If x is a Dataset then operation will be applied to all data variables.

    :param x: xarray.DataArray with `nodata` property
    :param where: xarray.DataArray<bool> True -- keep, False -- replace with `x.nodata`
    :param inplace: Modify pixels in x directly, not valid for dask arrays.

    For every pixel of x[idx], output is:

     - nodata  if where[idx] == False
     - x[idx]  if where[idx] == True
    """
    if isinstance(x, xr.Dataset):
        return x.apply(lambda x: keep_good_only(x, where, inplace=inplace),
                       keep_attrs=True)

    assert x.shape == where.shape
    if nodata is None:
        nodata = getattr(x, 'nodata', 0)

    if inplace:
        if dask.is_dask_collection(x):
            raise ValueError("Can not perform inplace operation on a dask array")

        np.copyto(x.data, nodata, where=~where.data)
        return x

    if dask.is_dask_collection(x):
        data = da.map_blocks(keep_good_np,
                             x.data, where.data, nodata,
                             name=randomize('keep_good'),
                             dtype=x.dtype)
    else:
        data = keep_good_np(x.data, where.data, nodata)

    return xr.DataArray(data,
                        dims=x.dims,
                        coords=x.coords,
                        attrs=x.attrs,
                        name=x.name)


def from_float_np(x, dtype, nodata, scale=1, offset=0, where=None):
    scale = np.float32(scale)
    offset = np.float32(offset)

    out = np.empty_like(x, dtype=dtype)

    params = dict(x=x,
                  nodata=nodata,
                  scale=scale,
                  offset=offset)

    # `x == x` is equivalent to `~np.isnan(x)`

    if where is not None:
        assert x.shape == where.shape
        params['m'] = where
        expr = 'where((x == x)&m, x*scale + offset, nodata)'
    else:
        expr = 'where(x == x, x*scale + offset, nodata)'

    ne.evaluate(expr,
                local_dict=params,
                out=out,
                casting='unsafe')

    return out


def to_float_np(x, nodata=None, scale=1, offset=0, dtype='float32'):
    float_type = np.dtype(dtype).type

    _nan = float_type(np.nan)
    scale = float_type(scale)
    offset = float_type(offset)

    params = dict(_nan=_nan,
                  scale=scale,
                  offset=offset,
                  x=x,
                  nodata=nodata)
    out = np.empty_like(x, dtype=dtype)

    if nodata is None:
        return ne.evaluate('x*scale + offset',
                           out=out,
                           casting='unsafe',
                           local_dict=params)
    elif scale == 1 and offset == 0:
        return ne.evaluate('where(x == nodata, _nan, x)',
                           out=out,
                           casting='unsafe',
                           local_dict=params)
    else:
        return ne.evaluate('where(x == nodata, _nan, x*scale + offset)',
                           out=out,
                           casting='unsafe',
                           local_dict=params)


def to_f32_np(x, nodata=None, scale=1, offset=0):
    return to_float_np(x, nodata=nodata, scale=scale, offset=offset, dtype='float32')


def to_float(x, scale=1, offset=0, dtype='float32'):
    if isinstance(x, xr.Dataset):
        return x.apply(to_float,
                       scale=scale,
                       offset=offset,
                       dtype=dtype,
                       keep_attrs=True)

    attrs = x.attrs.copy()
    nodata = attrs.pop('nodata', None)

    if dask.is_dask_collection(x.data):
        data = da.map_blocks(to_float_np,
                             x.data, nodata, scale, offset, dtype,
                             dtype=dtype,
                             name=randomize('to_float'))
    else:
        data = to_float_np(x.data,
                           nodata=nodata,
                           scale=scale,
                           offset=offset,
                           dtype=dtype)

    return xr.DataArray(data,
                        dims=x.dims,
                        coords=x.coords,
                        name=x.name,
                        attrs=attrs)


def to_f32(x, scale=1, offset=0):
    return to_float(x, scale=scale, offset=offset, dtype='float32')


def from_float(x, dtype, nodata, scale=1, offset=0):
    if isinstance(x, xr.Dataset):
        return x.apply(from_float, keep_attrs=True,
                       args=(dtype, nodata, scale, offset))

    attrs = x.attrs.copy()
    attrs['nodata'] = nodata

    if dask.is_dask_collection(x.data):
        data = da.map_blocks(from_float_np,
                             x.data, dtype, nodata,
                             scale=scale, offset=offset,
                             dtype=dtype,
                             name=randomize('from_float'))
    else:
        data = from_float_np(x.data, dtype, nodata,
                             scale=scale, offset=offset)

    return xr.DataArray(data,
                        dims=x.dims,
                        coords=x.coords,
                        name=x.name,
                        attrs=attrs)


def _impl_to_bool(x, m):
    return ((1 << x) & m) > 0


def _impl_to_bool_inverted(x, m):
    return ((1 << x) & m) == 0


def _flags_invert(flags: Dict[str, Any]) -> Dict[str, Any]:
    _out = dict(**flags)
    _out['values'] = {n: int(v)
                      for v, n in flags['values'].items()}
    return _out


def _get_enum_values(names: Iterable[str],
                     flags_definition: Dict[str, Dict[str, Any]],
                     flag: str = '') -> Tuple[int, ...]:
    """
    Lookup enum values in flags definition library

    :param names: enum value to lookup e.g. ("cloud", "shadow")
    :param flags_definition: Flags definition dictionary as used by Datacube
    :param flag: Name of the enum (for example "fmask", auto-guessed if omitted)
    """
    if flag != '':
        flags_definition = {flag: flags_definition[flag]}

    names = list(names)
    names_set = set(names)
    unmatched = set()
    for ff in flags_definition.values():
        values = _flags_invert(ff)['values']
        unmatched = names_set - set(values.keys())
        if len(unmatched) == 0:
            return tuple(values[n] for n in names)

    if len(flags_definition) > 1:
        raise ValueError("Can not find flags definitions that match query")
    else:
        unmatched_human = ",".join(f'"{name}"' for name in unmatched)
        raise ValueError(f"Not all enumeration names were found: {unmatched_human}")

def _mk_ne_isin_condition(values: Tuple[int,...],
                          var_name: str = 'a',
                          invert: bool = False) -> str:
    """
    Produce numexpr expression equivalent to numpys `.isin`

     - ((a==v1)|(a==v2)|..|a==vn)   when invert=False
     - ((a!=v1)&(a!=v2)&..&a!=vn)   when invert=True
    """
    op1, op2 = ('!=', '&') if invert else ('==', '|')
    parts = [f'({var_name}{op1}{val})' for val in values]
    return f'({op2.join(parts)})'


def _enum_to_mask_numexpr(mask: np.ndarray,
                          classes: Tuple[int, ...],
                          invert: bool = False,
                          value_true: int = 1,
                          value_false: int = 0,
                          dtype: Union[str, np.dtype] = 'bool') -> np.ndarray:
    cond = _mk_ne_isin_condition(classes, 'm', invert=invert)
    expr = f"where({cond}, {value_true}, {value_false})"
    out = np.empty_like(mask, dtype=dtype)

    ne.evaluate(expr,
                local_dict=dict(m=mask),
                out=out,
                casting='unsafe')

    return out


def fmask_to_bool(mask: xr.DataArray,
                  categories: Iterable[str],
                  invert: bool = False,
                  flag_name: str = '') -> xr.DataArray:
    """
    This method works for fmask and other "enumerated" masks

    It is equivalent to `np.isin(mask, categories)`

    example:
        xx = dc.load(.., measurements=['fmask', ...])
        no_cloud = fmask_to_bool(xx.fmask, ('valid', 'snow', 'water'))

        xx.where(no_cloud).isel(time=0).nbar_red.plot.imshow()

    """

    flags = getattr(mask, 'flags_definition', None)
    if flags is None:
        raise ValueError('Missing flags_definition attribute')

    classes = _get_enum_values(categories, flags, flag=flag_name)

    bmask = xr.apply_ufunc(_enum_to_mask_numexpr,
                           mask,
                           kwargs=dict(classes=classes, invert=invert),
                           keep_attrs=True,
                           dask='parallelized',
                           output_dtypes=['bool'])
    bmask.attrs.pop('flags_definition', None)
    bmask.attrs.pop('nodata', None)

    return bmask


def _gap_fill_np(a, fallback, nodata):
    params = dict(a=a,
                  b=fallback,
                  nodata=a.dtype.type(nodata))

    out = np.empty_like(a)

    if np.isnan(nodata):
        # a==a equivalent to `not isnan(a)`
        expr = 'where(a==a, a, b)'
    else:
        expr = 'where(a!=nodata, a, b)'

    return ne.evaluate(expr,
                       local_dict=params,
                       out=out,
                       casting='unsafe')


def gap_fill(x: xr.DataArray,
             fallback: xr.DataArray,
             nodata=None,
             attrs=None):
    """ Fill missing values in `x` with values from `fallback`.

        x,fallback are expected to be xarray.DataArray with identical shape and dtype.

            out[pix] = x[pix] if x[pix] != x.nodata else fallback[pix]
    """

    if nodata is None:
        nodata = getattr(x, 'nodata', None)

    if nodata is None:
        nodata = default_nodata(x.dtype)
    else:
        nodata = x.dtype.type(nodata)

    if attrs is None:
        attrs = x.attrs.copy()

    if dask.is_dask_collection(x):
        data = da.map_blocks(_gap_fill_np,
                             x.data, fallback.data, nodata,
                             name=randomize('gap_fill'),
                             dtype=x.dtype)
    else:
        data = _gap_fill_np(x.data, fallback.data, nodata)

    return xr.DataArray(data,
                        attrs=attrs,
                        dims=x.dims,
                        coords=x.coords,
                        name=x.name)


def test_gap_fill():
    a = np.zeros((5,), dtype='uint8')
    b = np.empty_like(a)
    b[:] = 33

    a[0] = 11
    ab = _gap_fill_np(a, b, 0)
    assert ab.dtype == a.dtype
    assert ab.tolist() == [11, 33, 33, 33, 33]

    xa = xr.DataArray(a,
                      name='test_a',
                      dims=('t',),
                      attrs={'p1': 1, 'nodata': 0},
                      coords=dict(t=np.arange(a.shape[0])))
    xb = xa + 0
    xb.data[:] = b
    xab = gap_fill(xa, xb)
    assert xab.name == xa.name
    assert xab.attrs == xa.attrs
    assert xab.data.tolist() == [11, 33, 33, 33, 33]

    xa.attrs['nodata'] = 11
    assert gap_fill(xa, xb).data.tolist() == [33, 0, 0, 0, 0]

    a = np.zeros((5,), dtype='float32')
    a[1:] = np.nan
    b = np.empty_like(a)
    b[:] = 33
    ab = _gap_fill_np(a, b, np.nan)

    assert ab.dtype == a.dtype
    assert ab.tolist() == [0, 33, 33, 33, 33]

    xa = xr.DataArray(a,
                      name='test_a',
                      dims=('t',),
                      attrs={'p1': 1},
                      coords=dict(t=np.arange(a.shape[0])))
    xb = xa + 0
    xb.data[:] = b
    xab = gap_fill(xa, xb)
    assert xab.name == xa.name
    assert xab.attrs == xa.attrs
    assert xab.data.tolist() == [0, 33, 33, 33, 33]

    xa = xr.DataArray(da.from_array(a),
                      name='test_a',
                      dims=('t',),
                      attrs={'p1': 1},
                      coords=dict(t=np.arange(a.shape[0])))

    xb = xr.DataArray(da.from_array(b),
                      name='test_a',
                      dims=('t',),
                      attrs={'p1': 1},
                      coords=dict(t=np.arange(b.shape[0])))

    assert dask.is_dask_collection(xa)
    assert dask.is_dask_collection(xb)
    xab = gap_fill(xa, xb)

    assert dask.is_dask_collection(xab)
    assert xab.name == xa.name
    assert xab.attrs == xa.attrs
    assert xab.compute().values.tolist() == [0, 33, 33, 33, 33]


def test_fmask_to_bool():
    import pytest

    def _fake_flags(prefix='cat_', n = 65):
        return dict(bits=list(range(8)),
                    values={str(i): f'{prefix}{i}' for i in range(0, n)})

    flags_definition = dict(fmask=_fake_flags())

    fmask = xr.DataArray(np.arange(0, 65, dtype='uint8'),
                         attrs=dict(flags_definition=flags_definition))

    mm = fmask_to_bool(fmask, ("cat_1", "cat_3"))
    ii, = np.where(mm)
    assert tuple(ii) == (1, 3)

    # upcast to uint16 internally
    mm = fmask_to_bool(fmask, ("cat_0", "cat_15"))
    ii, = np.where(mm)
    assert tuple(ii) == (0, 15)

    # upcast to uint32 internally
    mm = fmask_to_bool(fmask, ("cat_1", "cat_3", "cat_31"))
    ii, = np.where(mm)
    assert tuple(ii) == (1, 3, 31)

    # upcast to uint64 internally
    mm = fmask_to_bool(fmask, ("cat_0", "cat_32", "cat_37", "cat_63"))
    ii, = np.where(mm)
    assert tuple(ii) == (0, 32, 37, 63)

    with pytest.raises(ValueError):
        fmask_to_bool(fmask, ('cat_64'))

    mm = fmask_to_bool(fmask.chunk(3), ("cat_0",)).compute()
    ii, = np.where(mm)
    assert tuple(ii) == (0,)

    mm = fmask_to_bool(fmask.chunk(3), ("cat_31", "cat_63")).compute()
    ii, = np.where(mm)
    assert tuple(ii) == (31, 63)

    # check _get_enum_values
    flags_definition = dict(cat=_fake_flags("cat_"),
                            dog=_fake_flags("dog_"))
    assert _get_enum_values(("cat_0",), flags_definition) == (0,)
    assert _get_enum_values(("cat_0", "cat_12"), flags_definition) == (0, 12)
    assert _get_enum_values(("dog_0", "dog_13"), flags_definition) == (0, 13)
    assert _get_enum_values(("dog_0", "dog_13"), flags_definition, flag='dog') == (0, 13)

    with pytest.raises(ValueError) as e:
        _get_enum_values(("cat_10", "_nope"), flags_definition)
    assert "Can not find flags definitions" in str(e)

    with pytest.raises(ValueError) as e:
        _get_enum_values(("cat_10", "bah", "dog_0"), flags_definition, flag="dog")
    assert "cat_10" in str(e)


def test_enum_to_mask_numexpr():
    elements = (1, 4, 23)
    mm = np.asarray([1,2,3,4,5,23], dtype='uint8')

    np.testing.assert_array_equal(_enum_to_mask_numexpr(mm, elements),
                                  np.isin(mm, elements))
    np.testing.assert_array_equal(_enum_to_mask_numexpr(mm, elements, invert=True),
                                  np.isin(mm, elements, invert=True))

    bb8 = _enum_to_mask_numexpr(mm, elements, dtype='uint8', value_true=255)
    assert bb8.dtype == 'uint8'

    np.testing.assert_array_equal(
        _enum_to_mask_numexpr(mm, elements, dtype='uint8', value_true=255) == 255,
        np.isin(mm, elements))


