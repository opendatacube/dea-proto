"""
Generic dask helpers
"""

from typing import Tuple, Union, cast, Iterator, Optional, List, Any, Dict, Hashable
from random import randint
from datetime import datetime
from bisect import bisect_right, bisect_left
import numpy as np
import xarray as xr
import dask
from dask.core import get_dependencies, ishashable, istask
from dask.distributed import wait as dask_wait, TimeoutError
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
from dask import is_dask_collection
import functools
import toolz
from toolz import partition_all
import networkx as nx
from ._tools import ROI, roi_shape, slice_in_out


def chunked_persist(data, n_concurrent, client, verbose=False):
    """
    Force limited concurrency when persisting a large collection.

    This is useful to control memory usage when operating close to capacity.

    Sometimes `client.persist(data)` will run out of memory, not because
    fully-realized data is large, but because of intermediate data memory
    requirements. This is particularly common when using local dask cluster
    with only one worker.

    This function forces evaluation order of the dask graph to control peak
    memory usage.

    Say you have a largish task graph of 10x10 top-level sub-tasks, you have
    enough memory to process 5 sub-tasks concurrently, but Dask might decide
    to schedule more than that and will cause worker restarts due to out of
    memory errors. With this function you can force dask scheduler to
    persist this collection in batches of 5 concurrent sub-tasks, keeping
    the computation within the memory budget.
    """
    delayed = data.to_delayed().ravel()

    persisted = []
    for chunk in partition_all(n_concurrent, delayed):
        chunk = client.persist(chunk)
        _ = dask_wait(chunk)
        persisted.extend(chunk)
        if verbose:
            print(".", end="")

    # at this point it should be almost no-op
    return client.persist(data)


def chunked_persist_da(
    xx: xr.DataArray, n_concurrent, client, verbose=False
) -> xr.DataArray:
    data = chunked_persist(xx.data, n_concurrent, client=client, verbose=verbose)
    return xr.DataArray(data, dims=xx.dims, coords=xx.coords, attrs=xx.attrs)


def chunked_persist_ds(xx: xr.Dataset, client, verbose: bool = False) -> xr.Dataset:
    names = list(xx.data_vars)
    data = [xx[n].data for n in names]
    delayed = [d.to_delayed().ravel() for d in data]
    delayed = list(zip(*delayed))

    persisted = []
    for chunk in delayed:
        chunk = client.persist(chunk)
        _ = dask_wait(chunk)
        persisted.extend(chunk)
        if verbose:
            print(".", end="")

    # at this point it should be almost no-op
    data = client.persist(data)

    # reconstruct xr.Dataset from persisted chunks
    _vars = {}
    for n, d in zip(names, data):
        dv = xx[n]
        _vars[n] = xr.DataArray(data=d, dims=dv.dims, coords=dv.coords, name=n)

    return xr.Dataset(_vars)


def randomize(prefix: str) -> str:
    """
    Append random token to name
    """
    return "{}-{:08x}".format(prefix, randint(0, 0xFFFFFFFF))


@dask.delayed
def with_deps(value, *deps):
    return value


def list_reshape(x: List[Any], shape: Tuple[int, ...]) -> List[Any]:
    """
    similar to numpy version of x.reshape(shape), but only works on flat list on input.
    """
    for n in shape[1:][::-1]:
        x = list(map(list, toolz.partition(n, x)))
    return x


def unpack_chunksize(chunk: int, N: int) -> Tuple[int, ...]:
    """
    Compute chunk sizes
    Example: 4, 11 -> (4, 4, 3)
    """
    if chunk >= N or chunk < 0:
        return (N,)

    nb = N // chunk
    last_chunk = N - chunk * nb
    if last_chunk == 0:
        return tuple(chunk for _ in range(nb))

    return tuple(chunk for _ in range(nb)) + (last_chunk,)


def unpack_chunks(
    chunks: Tuple[int, ...], shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], ...]:
    """
    Expand chunks
    """
    assert len(chunks) == len(shape)
    return tuple(unpack_chunksize(ch, n) for ch, n in zip(chunks, shape))


def _roi_from_chunks(chunks: Tuple[int, ...]) -> Iterator[slice]:
    off = 0
    for v in chunks:
        off_next = off + v
        yield slice(off, off_next)
        off = off_next


def _split_chunks(
    chunks: Tuple[int, ...], max_chunk: int
) -> Iterator[Tuple[int, int, slice]]:
    """
    For every input chunk split it into smaller chunks.
    Return a list of tuples describing output chunks and their relation to input chunks.

    Output: [(dst_idx: int, src_idx: int, src_roi: slice]

    Note that every output chunk has only one chunk on input,
    so chunking might be irregular on output. This is by design, to avoid
    creating cross chunk dependencies.
    """
    dst_idx = 0
    for src_idx, src_sz in enumerate(chunks):
        off = 0
        while off < src_sz:
            sz = src_sz - off
            if max_chunk > 0:
                sz = min(sz, max_chunk)
            yield (dst_idx, src_idx, slice(off, off + sz))
            dst_idx += 1
            off += sz


def _get_chunks_asarray(xx: da.Array) -> np.ndarray:
    """
    Returns 2 ndarrays of equivalent shapes

    - First one contains dask tasks: (name: str, idx0:int, idx1:int)
    - Second one contains sizes of blocks (Tuple[int,...])
    """
    shape_in_chunks = xx.numblocks
    name = xx.name

    chunks = np.ndarray(shape_in_chunks, dtype="object")
    shapes = np.ndarray(shape_in_chunks, dtype="object")
    for idx in np.ndindex(shape_in_chunks):
        chunks[idx] = (name, *idx)
        shapes[idx] = tuple(xx.chunks[k][i] for k, i in enumerate(idx))
    return chunks, shapes


def _get_chunks_for_all_bands(xx: xr.Dataset):
    """
    Equivalent to _get_chunks_asarray(xx.to_array('band').data)
    """
    blocks = []
    shapes = []

    for dv in xx.data_vars.values():
        b, s = _get_chunks_asarray(dv.data)
        blocks.append(b)
        shapes.append(s)

    blocks = np.stack(blocks)
    shapes = np.stack(shapes)
    return blocks, shapes


def _get_all_chunks(xx: da.Array, flat: bool = True) -> List[Any]:
    shape_in_chunks = xx.numblocks
    name = xx.name
    chunks = [(name, *idx) for idx in np.ndindex(shape_in_chunks)]
    if flat:
        return chunks
    return list_reshape(chunks, shape_in_chunks)


def is_single_chunk_xy(x: da.Array):
    """
    True if last 2 dimensions are 1x1 sized in blocks
    """
    return x.numblocks[-2:] == (1, 1)


def empty_maker(fill_value, dtype, dsk, name="empty"):
    cache = {}

    def mk_empty(shape: Tuple[int, ...]) -> str:
        x = cache.get(shape, None)
        if x is not None:
            return x

        b_name = name + "_" + "x".join(str(i) for i in shape)
        b_name = randomize(b_name)
        cache[shape] = b_name
        dsk[b_name] = (np.full, shape, fill_value, dtype)
        return b_name

    return mk_empty


def _stack_2d_np(shape_in_blocks, *blocks, out=None, axis=0):
    """
    Stack a bunch of blocks into one plane.

    Takes a flat sequence of blocks in row major order an rearranges them onto a plane.

    Example:
      (2, 3) [a0, a1, a2, a3, a4, a5]
      >>
            [[a0, a1, a2],
             [a3, a4, a5]]

    By default assume that y,x dimensions are first, i.e. a[y,x] or a[y, x, band],
    but one can also stack blocks with extra dimensions by supplying axis= parameter,
    Example: for blocks like this: a[t, y, x, band] use axis=1

    :param shape_in_blocks: (ny, nx) number of blocks
    :param blocks: Blocks in row major order
    :param out: Allows re-use of memory, it must match dtype and output shape exactly
    :param axis: Index of y axis, x axis is then axis+1 (default is axis=0)
    """
    assert len(blocks) > 0
    assert len(shape_in_blocks) == 2
    assert shape_in_blocks[0] * shape_in_blocks[1] == len(blocks)

    dtype = blocks[0].dtype
    bshape = blocks[0].shape
    dims1 = bshape[:axis]
    dims2 = bshape[axis + 2 :]
    idx1 = tuple(slice(0, None) for _ in range(len(dims1)))
    idx2 = tuple(slice(0, None) for _ in range(len(dims2)))

    h, w = shape_in_blocks

    chunk_y = [b.shape[axis + 0] for b in blocks[0 : h * w : w]]
    chunk_x = [b.shape[axis + 1] for b in blocks[:w]]
    offset = [np.cumsum(x) for x in ([0] + chunk_y, [0] + chunk_x)]
    ny, nx = [x[-1] for x in offset]

    if out is None:
        out = np.empty((*dims1, ny, nx, *dims2), dtype=dtype)
    else:
        pass  # TODO: verify out shape is ok

    for block, idx in zip(blocks, np.ndindex(shape_in_blocks)):
        ny, nx = block.shape[axis : axis + 2]
        _y, _x = (offset[i][j] for i, j in zip([0, 1], idx))

        idx = (*idx1, slice(_y, _y + ny), slice(_x, _x + nx), *idx2)

        out[idx] = block

    return out


def _extract_as_one_block(axis, crop, shape_in_blocks, *blocks):
    out = _stack_2d_np(shape_in_blocks, *blocks, axis=axis)
    if crop is None:
        return out
    return out[crop]


def _chunk_getter(xx: da.Array):
    """
    _chunk_getter(xx)(np.s_[:3, 2:4]) -> (
    (xx.name, 0, 2),
    (xx.name, 0, 3),
    (xx.name, 1, 2),
    ...)
    """
    shape_in_chunks = xx.numblocks
    name = xx.name
    xx = np.asarray([{"v": tuple(idx)} for idx in np.ndindex(shape_in_chunks)]).reshape(
        shape_in_chunks
    )

    def getter(roi):
        return tuple((name, *x["v"]) for x in xx[roi].ravel())

    return getter


def _rechunk_2x2(xx, name="2x2"):
    """
    this is for testing only, ignore it, it's not robust
    """
    assert xx.ndim == 2
    name = randomize(name)
    ny, nx = (len(ch) // 2 for ch in xx.chunks[:2])

    dsk = {}
    chunks = _chunk_getter(xx)

    for r, c in np.ndindex((ny, nx)):
        r2 = r * 2
        c2 = c * 2
        ch_idx = np.s_[r2 : r2 + 2, c2 : c2 + 2]
        _xx = chunks(ch_idx)
        dsk[(name, r, c)] = (_stack_2d_np, (2, 2), *_xx)

    chy = tuple(xx.chunks[0][i * 2] + xx.chunks[0][i * 2 + 1] for i in range(ny))
    chx = tuple(xx.chunks[1][i * 2] + xx.chunks[1][i * 2 + 1] for i in range(nx))

    chunks = (chy, chx)
    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=(xx,))

    return da.Array(dsk, name, chunks=chunks, dtype=xx.dtype, shape=xx.shape)


def _compute_chunk_range(
    span: slice, chunks: Tuple[int, ...], summed: bool = False
) -> Tuple[slice, slice]:
    """
    Compute slice in chunk space and slice after taking just those chunks

    :param span: example: `np.s_[:10]`
    :param chunks: example: xx.chunks[0]

    """
    cs = chunks if summed else tuple(np.cumsum(chunks))
    n = cs[-1]

    _in, _out = slice_in_out(span, n)

    b_start = bisect_right(cs, _in)
    b_end = bisect_left(cs, _out) + 1

    offset = _in - (0 if b_start == 0 else cs[b_start - 1])
    sz = _out - _in

    return slice(b_start, b_end), slice(offset, offset + sz)


def compute_chunk_range(
    roi: ROI,
    chunks: Union[Tuple[int, ...], Tuple[Tuple[int, ...]]],
    summed: bool = False,
) -> Tuple[ROI, ROI]:
    """
    Convert ROI in pixels to ROI in blocks (broi) + ROI in pixels (crop) such that

       xx[roi] == stack_blocks(blocks(xx)[broi])[crop]

    Returns
    =======
      broi, crop
    """
    if isinstance(roi, slice):
        chunks = cast(Tuple[int, ...], chunks)
        return _compute_chunk_range(roi, chunks, summed)

    chunks = cast(Tuple[Tuple[int, ...]], chunks)
    assert len(roi) == len(chunks)
    broi = []
    crop = []

    for span, _chunks in zip(roi, chunks):
        bspan, pspan = _compute_chunk_range(span, _chunks)
        broi.append(bspan)
        crop.append(pspan)

    return tuple(broi), tuple(crop)


def crop_2d_dense(
    xx: da.Array, yx_roi: Tuple[slice, slice], name: str = "crop_2d", axis: int = 0
) -> da.Array:
    """
    xx[.., yx_roi, ..] -> Dask array with 1 single chunk in y,x dimension
    """
    assert len(yx_roi) == 2

    yx_broi, yx_crop = compute_chunk_range(yx_roi, xx.chunks[axis : axis + 2])
    assert isinstance(yx_crop, tuple)
    assert isinstance(yx_broi, tuple)

    xx_chunks = _chunk_getter(xx)
    bshape = roi_shape(yx_broi)

    #  tuple(*dims1, y, x, *dims2) -- complete shape in blocks
    dims1 = tuple(map(len, xx.chunks[:axis]))
    dims2 = tuple(map(len, xx.chunks[axis + 2 :]))

    # Adjust crop to include non-yx dimensions
    crop = (
        tuple(slice(0, None) for _ in dims1)
        + yx_crop
        + tuple(slice(0, None) for _ in dims2)
    )

    name = randomize(name)
    dsk = {}
    for ii1 in np.ndindex(dims1):
        roi_ii1 = tuple(slice(i, i + 1) for i in ii1)
        for ii2 in np.ndindex(dims2):
            roi_ii2 = tuple(slice(i, i + 1) for i in ii2)
            broi = roi_ii1 + yx_broi + roi_ii2
            blocks = xx_chunks(broi)
            assert len(blocks) == bshape[0] * bshape[1]
            dsk[(name, *ii1, 0, 0, *ii2)] = (
                _extract_as_one_block,
                axis,
                crop,
                bshape,
                *blocks,
            )

    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=(xx,))
    yx_shape = roi_shape(yx_crop)
    yx_chunks = tuple((n,) for n in yx_shape)
    chunks = xx.chunks[:axis] + yx_chunks + xx.chunks[axis + 2 :]
    shape = (*xx.shape[:axis], *yx_shape, *xx.shape[axis + 2 :])

    return da.Array(dsk, name, chunks=chunks, dtype=xx.dtype, shape=shape)


def _reshape_yxbt_impl(blocks, crop_yx=None, dtype=None):
    """
    input axis order : (band, time,) (blocks: y, x)
    output axis order: y, x, band, time
    """

    def squeeze_to_yx(x):
        idx = tuple(0 for _ in x.shape[:-2])
        return x[idx]

    assert len(blocks) > 0
    nb = len(blocks)
    nt = len(blocks[0])
    b = squeeze_to_yx(blocks[0][0])

    if dtype is None:
        dtype = b.dtype

    if crop_yx:
        b = b[crop_yx]
    ny, nx = b.shape

    dst = np.empty((ny, nx, nb, nt), dtype=dtype)
    for (it, ib) in np.ndindex((nt, nb)):
        b = squeeze_to_yx(blocks[ib][it])
        if crop_yx is not None:
            b = b[crop_yx]

        assert b.shape == (ny, nx)
        dst[:, :, ib, it] = b

    return dst


def reshape_yxbt(
    xx: xr.Dataset,
    name: str = "reshape_yxbt",
    yx_chunks: Union[int, Tuple[int, int]] = -1,
) -> xr.DataArray:
    """
    Reshape Dask-backed ``xr.Dataset[Time,Y,X]`` into
    ``xr.DataArray[Y,X,Band,Time]``. On the output DataArray there is
    exactly one chunk along both Time and Band dimensions.

    :param xx: Dataset with 3 dimensional bands, dimension order (time, y, x)

    :param name: Dask name of the output operation

    :param yx_chunks: If supplied subdivide YX chunks of input into smaller
                      sections, note that this can only make yx chunks smaller
                      not bigger. Every output chunk depends on one input chunk
                      only, so output chunks might not be regular, for example
                      if input chunk sizes are 10, and yx_chunks=3, you'll get
                      chunks sized 3,3,3,1,3,3,3,1... (example only, never use chunks
                      that small)

    .. note:

       Chunks along first dimension ought to be of size 1 exactly (default for
       time dimension when using dc.load).
    """
    if isinstance(yx_chunks, int):
        yx_chunks = (yx_chunks, yx_chunks)

    if not is_dask_collection(xx):
        raise ValueError("Currently this code works only on Dask inputs")

    if not all(
        dv.data.numblocks[0] == dv.data.shape[0] for dv in xx.data_vars.values()
    ):
        raise ValueError("All input bands should have chunk=1 for the first dimension")

    name0 = name
    name = randomize(name)

    blocks, _ = _get_chunks_for_all_bands(xx)
    b0, *_ = xx.data_vars.values()

    attrs = dict(b0.attrs)
    nb = len(xx.data_vars.values())
    nt, ny, nx = b0.shape

    deps = [dv.data for dv in xx.data_vars.values()]
    shape = (ny, nx, nb, nt)
    dtype = b0.dtype
    dims = b0.dims[1:] + ("band", b0.dims[0])

    maxy, maxx = yx_chunks
    ychunks, xchunks = b0.data.chunks[1:3]
    _yy = list(_split_chunks(ychunks, maxy))
    _xx = list(_split_chunks(xchunks, maxx))
    ychunks = tuple(roi.stop - roi.start for _, _, roi in _yy)
    xchunks = tuple(roi.stop - roi.start for _, _, roi in _xx)

    chunks = [ychunks, xchunks, (nb,), (nt,)]

    dsk = {}
    for iy, iy_src, y_roi in _yy:
        for ix, ix_src, x_roi in _xx:
            crop_yx = (y_roi, x_roi)
            _blocks = blocks[:, :, iy_src, ix_src].tolist()
            dsk[(name, iy, ix, 0, 0)] = (
                functools.partial(_reshape_yxbt_impl, crop_yx=crop_yx),
                _blocks,
            )

    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    data = da.Array(dsk, name, chunks=chunks, dtype=dtype, shape=shape)

    coords: Dict[Hashable, Any] = {k: c for k, c in xx.coords.items()}
    coords["band"] = list(xx.data_vars)

    return xr.DataArray(data=data, dims=dims, coords=coords, name=name0, attrs=attrs)


def flatten_kv(xx):
    """
    Turn dictionary into a flat list: [k0, v0, k1, v1, ...].

    Useful for things like map_blocks when passing Dict[str, da.Array] for example.
    """

    def _kv(xx):
        for k, v in xx.items():
            yield k
            yield v

    return list(_kv(xx))


def unflatten_kv(xx):
    """
    Reverse operation of `flatten_kv`
    """
    return {k: v for k, v in toolz.partition_all(2, xx)}


def wait_for_future(
    future, poll_timeout: float = 1.0, t0: Optional[datetime] = None
) -> Iterator[Tuple[float, datetime]]:
    """
    Generate a sequence of (time_passed, timestamp) tuples, stop when future becomes ready.

    :param future: Dask future
    :param poll_timeout: Controls how often
    :param t0: From what point to start counting (defaults to right now)
    """
    if t0 is None:
        t0 = datetime.utcnow()

    while not future.done():
        try:
            dask_wait(future, timeout=poll_timeout, return_when="FIRST_COMPLETED")
            return
        except TimeoutError:
            pass
        t_now = datetime.utcnow()

        yield ((t_now - t0).total_seconds(), t_now)


def to_nx_graph(dsk):
    """
    Converts a dask graph to a networkx graph. This code is adapted from dask.dot.to_graphviz
    """

    edges = []

    for k, v in dsk.items():
        if istask(v):
            for dep in get_dependencies(dsk, k):
                edges.append((dep, k))

        elif ishashable(v) and v in dsk:
            v_name = v
            edges.append((v, k))

    return nx.DiGraph(edges)


def _get_terminal_nodes(nx_graph, leaf=True):
    if leaf:
        return [node for node in nx_graph.nodes() if len(list(nx_graph.successors(node))) == 0]
    else:
        return [node for node in nx_graph.nodes() if len(list(nx_graph.predecessors(node))) == 0]


def parition_graph(nx_graph):
    """
    Partitions the input graph into smaller subgraphs and returns the leaf nodes of the subgraphs.

    This assumes a single output node. It removes nodes that are common ancenstors of to create
    a disconnected graph of multiple connected components. The leaf nodes of each connected 
    component is returned.

           o
        __/ \__
       /       \
      g         h     ======>      g         h     =====> returns nodes g and h
    / | \     / | \              / | \     / | \
    | | | ... | | |              | | | ... | | |
    a b c     d e f              a b c     d e f
    
    """
    root_nodes = _get_terminal_nodes(nx_graph, leaf=False)
    common_descendants = nx.algorithms.dag.descendants(digraph, root_nodes[0])
    for node in root_nodes:
        common_descendants = common_descendants.intersection(nx.algorithms.dag.descendants(digraph, node))

    disconnected_graph = digraph.copy()
    disconnected_graph.remove_nodes_from(common_descendants)

    components = nx.weakly_connected_components(disconnected_graph)
    components = (_get_terminal_nodes(disconnected_graph.subgraph(component), leaf=True) for component in components)
    
    return components


def auto_persist(dsk, n_concurrent):
    """
    Persists automatically determined dask nodes to minimize peak memory usage.
    """

    nx_graph = to_nx_graph(dsk)
    components = parition_graph(nx_graph)

    for chunk in partition_all(n_concurrent, components):
        chunk = [
            Delayed(key, dsk)
            for component in chunk
            for key in component
        ]
    
        chunk = client.persist(chunk)
        _ = dask_wait(chunk)

    # assumes only one output node
    leaf = _get_terminal_nodes(nx_graph, leaf=True)[0]
    
    # this needs to be computed instead of the original output array 
    out = Delayed(leaf, dsk)
    return out
