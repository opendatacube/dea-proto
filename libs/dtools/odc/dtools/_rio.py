""" rasterio environment management tools
"""
import threading
import rasterio
from rasterio.session import AWSSession
import rasterio.env

_local = threading.local()

SECRET_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def _sanitize(opts, keys):
    return {k: (v if k not in keys else "xx..xx") for k, v in opts.items()}


def get_rio_env(sanitize=True):
    """Get GDAL params configured by rasterio for the current thread.

    :param sanitize: If True replace sensitive Values with 'x'
    """

    env = rasterio.env.local._env
    if env is None:
        return {}
    opts = env.get_config_options()
    if sanitize:
        opts = _sanitize(opts, SECRET_KEYS)

    return opts


def activate_rio_env(aws=None, defaults=True, **kwargs):
    """Inject activated rasterio.Env into current thread.

    This de-activates previously setup environment.

    :param aws: Dictionary of options for rasterio.session.AWSSession
                OR False in which case session won't be setup
                OR None -- session = rasterio.session.AWSSession()

    :param defaults: Supply False to not inject COG defaults
    :param **kwargs: Passed on to rasterio.Env(..) constructor
    """
    env_old = getattr(_local, "env", None)

    if env_old is not None:
        env_old.__exit__(None, None, None)
        _local.env = None

    if aws is False:
        session = None
    else:
        aws = {} if aws is None else dict(**aws)
        region_name = aws.get("region_name", "auto")

        if region_name == "auto":
            from odc.aws import auto_find_region

            try:
                aws["region_name"] = auto_find_region()
            except Exception as e:
                # only treat it as error if it was requested by user
                if "region_name" in aws:
                    raise e

        session = AWSSession(**aws)

    opts = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR") if defaults else {}

    opts.update(**kwargs)

    env = rasterio.Env(session=session, **opts)
    env.__enter__()
    _local.env = env
    return get_rio_env()
