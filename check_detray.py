import github
import git
import subprocess
import os
import sys
import json
import pandas
import numpy
import tempfile
import pathlib
import logging
import time

log = logging.getLogger("detray_benchmark")

def render_throughput(t):
    if numpy.isnan(t):
        return "&mdash;"
    if t >= 1000000000.0:
        return "%.2f GHz" % (t / 1000000000.0)
    elif t >= 1000000.0:
        return "%.2f MHz" % (t / 1000000.0)
    elif t >= 1000.0:
        return "%.2f kHz" % (t / 1000.0)
    else:
        return "%.2f Hz" % (t)


def get_means(file, fc):
    mommap = {
        "tdwc": "[1, 100) GeV",
        "odd1cov": "1 GeV",
        "odd10cov": "10 GeV",
        "odd100cov": "100 GeV",
        "odd1to100cov": "[1, 100) GeV",
        "odd1nocov": "1 GeV",
        "odd10nocov": "10 GeV",
        "odd100nocov": "100 GeV",
        "odd1to100nocov": "[1, 100) GeV",
    }

    trans = {
        "DETRAY_BM__WIRE_CHAMBER_W_COV_TRANSPORT_262144_TRACKS": "Wire chamber (%s, cov. on)",
        "DETRAY_BM__WIRE_CHAMBER_262144_TRACKS": "Wire chamber (%s, cov. off)",
        "DETRAY_BM__TOY_DETECTOR_W_COV_TRANSPORT_262144_TRACKS": "Toy detector (%s, cov. on)",
        "DETRAY_BM__TOY_DETECTOR_262144_TRACKS": "Toy detector (%s, cov. off)",
        "DETRAY_BM__Cylindrical detector from DD4hep blueprint_W_COV_TRANSPORT_250000_TRACKS": "ODD (%s, cov. on)",
        "DETRAY_BM__Cylindrical detector from DD4hep blueprint_250000_TRACKS": "ODD (%s, cov. off)",
    }

    means = {}
    cvs = {}
    with open(file, "r") as f:
        cnt = json.load(f)
    for b in cnt["benchmarks"]:
        p1, p2 = b["name"].split("/", 1)
        nn  = trans[p1] % (mommap[fc])
        if b["name"].endswith("_cv"):
            assert(b["aggregate_unit"] == "percentage")
            cvs[nn] = b["TracksPropagated"] 
        elif b["name"].endswith("_mean"):
            assert(b["aggregate_unit"] == "time")
            assert(b["time_unit"] == "ns")
            means[nn] = b["TracksPropagated"]

    return {k: (means[k], cvs[k]) for k in means.keys()}


def build_and_benchmark(repo_dir, repo, commit):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = pathlib.Path(tmpdirname)

        log.info('Created temporary directory "%s"', str(tmppath))

        build_dir = tmppath / "build"
        log.info("Checking out commit %s", commit)
        repo.git.checkout(commit)

        log.info('Building detray into build directory "%s"', build_dir)
   

        log.info("Running configuration step")

        start_time = time.time()

        config_args = [
            "cmake",
            "-S",
            repo_dir,
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=86",
            "-DDETRAY_CUSTOM_SCALARTYPE=float",
            "-DDETRAY_BUILD_CUDA=ON",
            "-DDETRAY_BUILD_BENCHMARKS=ON",
            "-DDETRAY_BUILD_CLI_TOOLS=ON",
        ]

        subprocess.run(
            config_args,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        end_time = time.time()

        log.info(
            "Completed configuration step in %.1f seconds",
            end_time - start_time,
        )

        parallel = 16
        log.info("Running build step with %d thread(s)", parallel)

        start_time = time.time()

        subprocess.run(
            [
                "cmake",
                "--build",
                build_dir,
                "--",
                "-j",
                str(parallel),
                "detray_benchmark_cuda_propagation_array",
                "detray_propagation_benchmark_cuda_array",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        end_time = time.time()

        log.info("Completed build step in %.1f seconds", end_time - start_time)

        log.info("Running benchmark step")

        start_time = time.time()

        log.info("Running benchmark for the toy detector and wire chamber at [1, 100) GeV...")
        subprocess.run(
            [
                "gpu_lock",
                "A5000",
                build_dir / "bin" / "detray_benchmark_cuda_propagation_array",
                "--benchmark_filter=262144",
                "--benchmark_repetitions=5",
                "--benchmark_min_time=5s",
                "--benchmark_out=%s" % str(CACHE_DIR / ("%s_tdwc.json" % commit)),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        
        for cv in [True, False]:
            for pt in [1, 10, 100]:
                log.info("Running benchmark for the ODD at %d GeV %s covariance transport...", pt, "with" if cv else "without")
                args = [
                    "gpu_lock",
                    "A5000",
                    build_dir / "bin" / "detray_propagation_benchmark_cuda_array",
                    "--benchmark_filter=250000",
                    "--benchmark_repetitions=5",
                    "--benchmark_min_time=5s",
                    "--material_file=/software/modules/dt/traccc-data/10/geometries/odd/odd-detray_material_detray.json",
                    "--geometry_file=/software/modules/dt/traccc-data/10/geometries/odd/odd-detray_geometry_detray.json",
                    "--grid_file=/software/modules/dt/traccc-data/10/geometries/odd/odd-detray_surface_grids_detray.json",
                    "--sort_tracks",
                    "--randomize_charge",
                    "--pT_range", "%d" % pt, "%d" % pt,
                    "--benchmark_out=%s" % str(CACHE_DIR / ("%s_odd%d%s.json" % (commit, pt, "cov" if cv else "nocov"))),
                ]
                if cv:
                    args.append("--covariance_transport")
                subprocess.run(
                    args,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            
            log.info("Running benchmark for the ODD at [1, 100) GeV %s covariance transport...", "with" if cv else "without")
            args = [
                "gpu_lock",
                "A5000",
                build_dir / "bin" / "detray_propagation_benchmark_cuda_array",
                "--benchmark_filter=250000",
                "--benchmark_repetitions=5",
                "--benchmark_min_time=5s",
                "--material_file=/software/modules/dt/traccc-data/10/geometries/odd/odd-detray_material_detray.json",
                "--geometry_file=/software/modules/dt/traccc-data/10/geometries/odd/odd-detray_geometry_detray.json",
                "--grid_file=/software/modules/dt/traccc-data/10/geometries/odd/odd-detray_surface_grids_detray.json",
                "--sort_tracks",
                "--randomize_charge",
                "--pT_range", "1", "100",
                "--benchmark_out=%s" % str(CACHE_DIR / ("%s_odd1to100%s.json" % (commit, "cov" if cv else "nocov"))),
            ]
            if cv:
                args.append("--covariance_transport")
            subprocess.run(
                args,
                check=True,
                stdout=subprocess.DEVNULL,
            )

        end_time = time.time()

        log.info("Completed benchmark step in %.1f seconds", end_time - start_time)


CACHE_DIR = pathlib.Path("/mnt/ssd1/sswatman/detray-compute-cache")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    auth = github.Auth.Token(os.environ["GHTOKEN"])

    client = github.Github(auth=auth)

    repo = client.get_repo("acts-project/detray")

    pr_id = int(sys.argv[1])

    pr = repo.get_pull(pr_id)

    pr_sha = pr.head.sha
    
    repo_dir = "/mnt/ssd1/sswatman/benchdetray/detray_subject/"
    local_repo = git.Repo(repo_dir)
    
    if local_repo.is_dirty():
        e = "Repository is dirty; please clean it before use!"
        log.fatal(e)
        raise RuntimeError(e)

    if "DETRAY_BFIELD_FILE" not in os.environ:
        e = 'Environment variable "DETRAY_BFIELD_FILE" is not set; aborting!'
        log.fatal(e)
        raise RuntimeError(e)

    if "DETRAY_DATA_DIRECTORY" not in os.environ:
        e = 'Environment variable "DETRAY_DATA_DIRECTORY" is not set; aborting!'
        log.fatal(e)
        raise RuntimeError(e)

    remote = git.Remote(local_repo, pr.user.login)

    if remote.exists():
        remote.set_url(pr.head.repo.ssh_url)
    else:
        remote = local_repo.create_remote(pr.user.login, pr.head.repo.ssh_url)

    assert remote.exists()
    remote.fetch()

    local_repo.git.checkout("main")
    local_repo.git.pull()

    local_repo.git.checkout(pr.head.sha)

    mbl = local_repo.merge_base("main", pr.head.sha)
    assert len(mbl) == 1
    mb = mbl[0]
    mb_sha = mb.hexsha

    file_templates = ["odd1cov", "odd1nocov", "odd10cov", "odd10nocov", "odd100cov", "odd100nocov", "odd1to100cov", "odd1to100nocov", "tdwc"]

    for sha in [pr_sha, mb_sha]:
        all_files = [(CACHE_DIR / ("%s_%s.json" % (sha, tp))) for tp in file_templates]

        if all(f.is_file() for f in all_files):
            log.info("Cache files for commit %s exist; reusing", sha)
        else:
            log.info("Cache files for commit %s do not exist; running benchmark", sha)
            build_and_benchmark(repo_dir, local_repo, sha)

        assert(all(f.is_file() for f in all_files))


    text = (
        "# Performance summary\nHere is a summary of the performance effects of this PR:\n## Tabular\n"
    )
    
    before_perf = dict()
    after_perf = dict()

    for tp in file_templates:
        before_perf = dict(**before_perf, **get_means(CACHE_DIR / ("%s_%s.json" % (mb_sha, tp)), tp))
        after_perf = dict(**after_perf, **get_means(CACHE_DIR / ("%s_%s.json" % (pr_sha, tp)), tp))

    text += '<table><thead><tr><th rowspan="2">Benchmark</th><th align="center" colspan="2">{0}</th><th align="center" colspan="2">{1}</th><th rowspan="2">Delta</th></tr>'.format(str(mb)[:8], pr.head.sha[:8])
    text += '<tr><th align="right">Throughput</th><th align="right">CV</th><th align="right">Throughput</th><th align="right">CV</th></tr></thead>'.format(
        str(mb)[:8], pr.head.sha[:8]
    )
    text += "<tbody>"

    for i in before_perf.keys():
        thold = before_perf[i][0]
        thnew = after_perf[i][0]
        
        cvold = before_perf[i][1]
        cvnew = after_perf[i][1]

        text += (
            '<tr><td>%s</td><td align="right">%s</td><td align="right">%.2f%%</td><td align="right">%s</td><td align="right">%.2f%%</td><td align="right">%+.2f%%</td></tr>'
            % (i, render_throughput(thold), cvold * 100.0, render_throughput(thnew), cvnew* 100.0, ((thnew / thold) - 1) * 100.0)
        )

    text += "</table>\n\n"

    text += "> [!NOTE]\n> This is an automated message produced upon the explicit request of a human being."

    pr.create_issue_comment(text)

    # To close connections after use
    client.close()


if __name__ == "__main__":
    main()
