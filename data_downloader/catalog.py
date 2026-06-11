from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    access: str
    time_mode: str = "none"  # none | range | year_range | fireseason | weekly_archive | latest
    aoi_mode: str = "unsupported"  # unsupported | native | postfilter
    default_years: tuple[int, int] | None = None
    notes: str = ""
    provider_urls: List[str] = field(default_factory=list)


DATASETS: Dict[str, DatasetSpec] = {
    "hrrr_fireseason": DatasetSpec(
        name="hrrr_fireseason",
        family="weather",
        access="public_direct",
        time_mode="fireseason",
        default_years=(2024, 2024),
        notes="NOAA HRRR forecast files from the public S3 archive. Defaults to 00/06/12/18 UTC fire-season f00 surface files.",
        provider_urls=[
            "https://registry.opendata.aws/noaa-hrrr-pds/",
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/",
        ],
    ),
    "aqs_pm25": DatasetSpec(
        name="aqs_pm25",
        family="air_quality",
        access="public_direct",
        time_mode="year_range",
        default_years=(2020, 2025),
        notes="EPA AirData hourly PM2.5 files for parameter codes 88101 and 88502 plus site/monitor metadata.",
        provider_urls=["https://aqs.epa.gov/aqsweb/airdata/download_files.html"],
    ),
    "hms_smoke": DatasetSpec(
        name="hms_smoke",
        family="smoke",
        access="public_direct",
        time_mode="year_range",
        default_years=(2020, 2025),
        notes="NOAA HMS annual smoke-polygon and fire-point shapefile bundles.",
        provider_urls=[
            "https://www.ospo.noaa.gov/products/land/fire.html",
            "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/",
        ],
    ),
    "ibtracs": DatasetSpec(
        name="ibtracs",
        family="cyclone",
        access="public_direct",
        time_mode="latest",
        notes="NOAA/NCEI IBTrACS v04r01 CSV tracks and documentation.",
        provider_urls=[
            "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/",
        ],
    ),
    "hurdat2": DatasetSpec(
        name="hurdat2",
        family="cyclone",
        access="public_direct",
        time_mode="latest",
        notes="NOAA/NHC HURDAT2 best-track text archives and format documents.",
        provider_urls=["https://www.nhc.noaa.gov/data/hurdat/"],
    ),
    "usdm": DatasetSpec(
        name="usdm",
        family="drought",
        access="public_direct",
        time_mode="weekly_archive",
        default_years=(2000, 2026),
        notes="U.S. Drought Monitor weekly GIS shapefile archive.",
        provider_urls=[
            "https://droughtmonitor.unl.edu/Data.aspx",
            "https://droughtmonitor.unl.edu/DmData/GISData.aspx",
        ],
    ),
    "mtbs_perimeters": DatasetSpec(
        name="mtbs_perimeters",
        family="wildfire",
        access="arcgis_rest",
        time_mode="none",
        notes="MTBS national burned-area perimeter features from the USDA Forest Service ArcGIS service. Public client records provider instructions before any heavy chunked pull.",
        provider_urls=[
            "https://www.mtbs.gov/",
            "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer",
        ],
    ),
    "wfigs_current": DatasetSpec(
        name="wfigs_current",
        family="wildfire",
        access="arcgis_rest",
        time_mode="latest",
        notes="Current WFIGS interagency fire perimeter snapshot from the NIFC ArcGIS service.",
        provider_urls=["https://data-nifc.opendata.arcgis.com/"],
    ),
    "firms_area": DatasetSpec(
        name="firms_area",
        family="wildfire",
        access="auth_required",
        time_mode="range",
        aoi_mode="native",
        notes="NASA FIRMS area API. Requires a FIRMS MAP_KEY supplied by environment or extra option.",
        provider_urls=[
            "https://firms.modaps.eosdis.nasa.gov/api/",
            "https://firms.modaps.eosdis.nasa.gov/download/",
        ],
    ),
    "landfire_static": DatasetSpec(
        name="landfire_static",
        family="static",
        access="public_or_provider_direct",
        time_mode="none",
        notes="LANDFIRE static products. Direct product URLs can change; downloader includes known LF2024 CONUS URLs and falls back to provider instructions.",
        provider_urls=["https://landfire.gov/data"],
    ),
    "wrc_housing": DatasetSpec(
        name="wrc_housing",
        family="static",
        access="provider_direct",
        time_mode="none",
        notes="Wildfire Risk to Communities housing-unit density. Provider links may redirect; downloader records source instructions and optional URL override.",
        provider_urls=[
            "https://wildfirerisk.org/download/",
            "https://data-usfs.hub.arcgis.com/datasets/usfs::wildfire-risk-to-communities-housing-unit-density-image-service",
        ],
    ),
    "landscan": DatasetSpec(
        name="landscan",
        family="static",
        access="restricted_terms",
        time_mode="none",
        notes="LandScan requires provider-specific access and terms. Downloader intentionally writes instructions rather than bypassing access controls.",
        provider_urls=["https://landscan.ornl.gov/"],
    ),
    "merra2": DatasetSpec(
        name="merra2",
        family="weather",
        access="earthdata_required",
        time_mode="range",
        notes="NASA GES DISC MERRA-2 access requires Earthdata credentials. Downloader writes a request manifest/instructions unless credentials are configured externally.",
        provider_urls=[
            "https://disc.gsfc.nasa.gov/datasets?keywords=MERRA-2",
            "https://earthdata.nasa.gov/",
        ],
    ),
}
