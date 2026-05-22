# SPDX-PackageName = "traccc, a part of the ACTS project"
# SPDX-FileCopyrightText: CERN
# SPDX-License-Identifier: MPL-2.0

import subprocess
import pathlib
import logging
import os

log = logging.getLogger("traccc_benchmark")


def run_profile(
    build_dir: pathlib.Path, data_dir: str, events=1, ncu_wrapper=None
):
    profile_args = [
        "ncu",
        "--import-source",
        "no",
        "--section LaunchStats",
        "--section Occupancy",
        "--metrics gpu__time_duration.sum",
        "-f",
        "-o",
        build_dir / "profile",
        build_dir / "bin" / "traccc_throughput_st_cuda",
        "--input-directory=%s" % data_dir,
        "--digitization-file=geometries/odd/odd-digi-geometric-config.json",
        "--conditions-file=geometries/odd/odd-digi-geometric-config.json",
        "--detector-file=geometries/odd/odd-detray_geometry_detray.json",
        "--grid-file=geometries/odd/odd-detray_surface_grids_detray.json",
        "--material-file=geometries/odd/odd-detray_material_detray.json",
        "--input-events=%d" % events,
        "--cold-run-events=0",
        "--processed-events=%d" % events,
        "--track-candidates-range=5:100",
        "--seedfinder-vertex-range=-150:150",
        "--use-acts-geom-source=1",
        "--read-bfield-from-file",
        "--bfield-file=geometries/odd/odd-bfield.cvf",
    ]

    if ncu_wrapper is not None:
        profile_args = ncu_wrapper.split() + profile_args

    subprocess.run(
        profile_args,
        stdout=subprocess.DEVNULL,
        check=True,
    )
