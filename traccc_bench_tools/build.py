# SPDX-PackageName = "traccc, a part of the ACTS project"
# SPDX-FileCopyrightText: CERN
# SPDX-License-Identifier: MPL-2.0

import pathlib
import logging
import subprocess
import typing
import os


log = logging.getLogger("traccc_benchmark")


def configure(
    source_dir: pathlib.Path, build_dir: pathlib.Path, cc: str = None, root=False
):
    config_args = [
        "cmake",
        "-S",
        source_dir,
        "-B",
        build_dir,
        "-DTRACCC_BUILD_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTRACCC_USE_ROOT=OFF",
        "-DDETRAY_GENERATE_METADATA=itk_metadata",
        "-DTRACCC_USE_SPACK_LIBS=ON",
    ]

    if root:
        config_args.append("-DTRACCC_USE_ROOT=ON")
    else:
        config_args.append("-DTRACCC_USE_ROOT=OFF")

    if cc is not None:
        config_args.append("-DCMAKE_CUDA_ARCHITECTURES=%s" % cc)

    subprocess.run(
        config_args,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def build(build_dir: pathlib.Path, parallel: int = 1):
    subprocess.run(
        [
            "cmake",
            "--build",
            build_dir,
            "--",
            "-j",
            str(parallel),
            "traccc_throughput_st_cuda",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
