"""
Various I/O adaptors
"""

from typing import Dict, Any, Optional, List, Union
import json
from urllib.parse import urlparse
from dask.delayed import Delayed
from pathlib import Path
import xarray as xr

from datacube.utils.aws import get_creds_with_retry, mk_boto_session, s3_client
from odc.aws import s3_head_object  # TODO: move it to datacube
from datacube.utils.dask import save_blob_to_s3, save_blob_to_file
from datacube.utils.cog import to_cog
from datacube.model import Dataset
from botocore.credentials import ReadOnlyCredentials
from .model import Task, EXT_TIFF


DEFAULT_COG_OPTS = dict(
    compress='deflate',
    predict=2,
    zlevel=6,
    blocksize=512,
)


def load_creds(profile: Optional[str] = None) -> ReadOnlyCredentials:
    session = mk_boto_session(profile=profile)
    creds = get_creds_with_retry(session)
    if creds is None:
        raise ValueError("Failed to obtain credentials")

    return creds.get_frozen_credentials()


def dump_json(meta: Dict[str, Any]) -> str:
    return json.dumps(meta, separators=(',', ':'))


class S3COGSink:
    def __init__(self,
                 creds: Union[ReadOnlyCredentials, str, None] = None,
                 cog_opts: Optional[Dict[str, Any]] = None,
                 public: bool = False):

        if cog_opts is None:
            cog_opts = dict(**DEFAULT_COG_OPTS)

        self._creds = creds
        self._cog_opts = cog_opts
        self._meta_ext = 'json'
        self._meta_contentype = 'application/json'
        self._band_ext = EXT_TIFF
        self._public = public

    def uri(self, task: Task) -> str:
        return task.metadata_path('absolute', ext=self._meta_ext)

    def _get_creds(self) -> ReadOnlyCredentials:
        if self._creds is None:
            self._creds = load_creds()
        if isinstance(self._creds, str):
            self._creds = load_creds(self._creds)
        return self._creds

    def verify_s3_credentials(self, test_uri: Optional[str] = None) -> bool:
        try:
            _ = self._get_creds()
        except ValueError:
            return False
        if test_uri is None:
            return True
        path, ok = self._write_blob(b"verifying S3 permissions", test_uri).compute()
        assert path == test_uri
        return ok

    def _write_blob(self,
                    data,
                    url: str,
                    ContentType: Optional[str] = None,
                    with_deps=None) -> Delayed:
        _u = urlparse(url)
        if _u.scheme == 's3':
            kw = dict(creds=self._get_creds())
            if ContentType is not None:
                kw['ContentType'] = ContentType
            if self._public:
                kw['ACL'] = 'public-read'

            return save_blob_to_s3(data, url, with_deps=with_deps, **kw)
        elif _u.scheme == 'file':
            _dir = Path(_u.path).parent
            if not _dir.exists():
                _dir.mkdir(parents=True, exist_ok=True)
            return save_blob_to_file(data, _u.path, with_deps=with_deps)
        else:
            raise ValueError(f"Don't know how to save to '{url}'")

    def _ds_to_cog(self,
                   ds: xr.Dataset,
                   paths: Dict[str, str]) -> List[Delayed]:
        out = []
        for band, dv in ds.data_vars.items():
            url = paths.get(band, None)
            if url is None:
                raise ValueError(f"No path for band: '{band}'")
            cog_bytes = to_cog(dv, **self._cog_opts)
            out.append(self._write_blob(cog_bytes,
                                        url,
                                        ContentType='image/tiff'))
        return out

    def write_cog(self, da: xr.DataArray, url: str) -> Delayed:
        cog_bytes = to_cog(da, **self._cog_opts)
        return self._write_blob(cog_bytes, url, ContentType='image/tiff')

    def exists(self, task: Union[Task, str]) -> bool:
        if isinstance(task, str):
            uri = task
        else:
            uri = self.uri(task)
        _u = urlparse(uri)
        if _u.scheme == 's3':
            s3 = s3_client(creds=self._get_creds(), cache=True)
            meta = s3_head_object(uri, s3=s3)
            return meta is not None
        elif _u.scheme == 'file':
            return Path(_u.path).exists()
        else:
            raise ValueError(f"Can't handle url: {uri}")

    def dump(self,
             task: Task,
             ds: Dataset) -> Delayed:
        paths = task.paths('absolute', ext=self._band_ext)
        cogs = self._ds_to_cog(ds, paths)

        json_url = task.metadata_path('absolute', ext=self._meta_ext)
        meta = task.render_metadata(ext=self._band_ext)

        json_txt = dump_json(meta)

        return self._write_blob(json_txt.encode('utf8'),
                                json_url,
                                ContentType=self._meta_contentype,
                                with_deps=cogs)
