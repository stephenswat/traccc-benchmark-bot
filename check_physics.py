import github
import git
import subprocess
import os
import sys
import pandas
import shutil
import tempfile
import numpy
import json
import pathlib
import logging
import re
import matplotlib.pyplot
import traccc_bench_tools.build


log = logging.getLogger("traccc_bench_bot.check_physics")

CACHE_DIR = pathlib.Path("/mnt/ssd1/sswatman/traccc-physics-cache")

def build_run_convert(src_dir, local_repo, commit_hash, seeding_exec):
    log.info("Building repository for commit %s...", commit_hash)

    commit = local_repo.commit(commit_hash)
    
    local_repo.git.checkout(commit_hash)

    with tempfile.TemporaryDirectory() as tmpdirname:
        log.info("Building into temporary directory %s", tmpdirname)

        log.info("Running configuration step...")
        traccc_bench_tools.build.configure(src_dir, tmpdirname, commit, root=True)
        log.info("Running build step...")

        build_args = [
            "cmake",
            "--build",
            str(tmpdirname),
            "--",
            "-j",
            "48",
            "traccc_seeding_example_cuda",
        ]

        subprocess.run(
            build_args,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        log.info("Running seeding example...")
        
        result = subprocess.run(
            [pathlib.Path(tmpdirname) / "bin" / "traccc_seeding_example_cuda"] + seeding_exec, 
            check=True, cwd=tmpdirname,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        stdout = result.stdout.decode("utf-8")

        if match := re.search(
            r"- created \(cuda\)\s+(\d+) seeds", stdout
        ):
            num_seeds = int(match.group(1))
        else:
            raise ValueError("Seed count could not be parsed from stdout!")

        if match := re.search(
            r"- created \(cuda\)\s+(\d+) found tracks", stdout
        ):
            num_found_tracks = int(match.group(1))
        else:
            raise ValueError("Found track count could not be parsed from stdout!")

        log.info("Making new directory in cache...")
        try:
            os.mkdir(CACHE_DIR / commit_hash)
            log.info("Cache dir succesfully created")
        except FileExistsError:
            log.info("Cache dir already exists")

        with open(CACHE_DIR / commit_hash / "counts.json", 'w') as f:
            json.dump({"seeds": num_seeds, "found": num_found_tracks}, f)

        for f in ["performance_track_seeding.root", "performance_track_finding.root"]:
            shutil.copy(pathlib.Path(tmpdirname) / f, CACHE_DIR / commit_hash)

            subprocess.run(
                ["/mnt/ssd1/sswatman/teff_to_csv/teff_to_csv", str(CACHE_DIR/ commit_hash / f), str(CACHE_DIR / commit_hash)], check=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Initiating Github API client...")

    auth = github.Auth.Token(os.environ['GHTOKEN'])

    client = github.Github(auth=auth)

    log.info("Retrieving PR information...")
    repo = client.get_repo("acts-project/traccc")

    pr_id = int(sys.argv[1])

    pr = repo.get_pull(pr_id)

    log.info("Updating local repository")
    local_repo_path = "/mnt/ssd1/sswatman/benchtraccc/traccc_subject/"
    local_repo = git.Repo(local_repo_path)

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
    assert(len(mbl) == 1)
    mb = mbl[0]

    cwd = "/mnt/ssd1/sswatman/benchtraccc/traccc/extras/benchmark"

    csv_file_names = [
        ("seeding_trackeff_vs_pT", "eff"),
        ("seeding_trackeff_vs_eta", "eff"),
        ("seeding_trackeff_vs_phi", "eff"),
        ("finding_trackeff_vs_pT", "eff"),
        ("finding_trackeff_vs_eta", "eff"),
        ("finding_trackeff_vs_phi", "eff"),
        ("finding_nDuplicated_vs_eta", "prof"),
        ("finding_nFakeTracks_vs_eta", "prof"),
        ("ndf", "hist"),
        ("pval", "hist"),
        ("purity", "hist"),
        ("completeness", "hist"),
    ]

    pr_dir = CACHE_DIR / str(pr.head.sha)
    mb_dir = CACHE_DIR / str(mb.hexsha)

    seeding_example_args = [
        "--input-directory=/data/Acts/odd-simulations-20240506/geant4_ttbar_mu200",
        "--digitization-file=geometries/odd/odd-digi-geometric-config.json",
        "--detector-file=geometries/odd/odd-detray_geometry_detray.json",
        "--grid-file=geometries/odd/odd-detray_surface_grids_detray.json",
        "--material-file=geometries/odd/odd-detray_material_detray.json",
        "--use-detray-detector=on",
        "--input-events=10",
        "--use-acts-geom-source=on",
        "--check-performance",
        "--truth-finding-min-track-candidates=5",
        "--truth-finding-min-pt=1.0",
        "--truth-finding-min-z=-150",
        "--truth-finding-max-z=150",
        "--truth-finding-max-r=10",
        "--seed-matching-ratio=0.99",
        "--track-matching-ratio=0.5",
        "--track-candidates-range=5:100",
        "--seedfinder-vertex-range=-150:150"
    ]

    if pr_dir.is_dir() and all((pr_dir / (x + ".csv")).is_file() for (x, _) in csv_file_names):
        log.info("Cache dir for PR commit %s exists; reusing", pr.head.sha)
    else:
        log.info("Cache dir for PR commit %s does not exist; generating", pr.head.sha)
        build_run_convert(local_repo_path, local_repo, pr.head.sha, seeding_example_args)

    
    if mb_dir.is_dir() and all((mb_dir / (x + ".csv")).is_file() for (x, _) in csv_file_names):
        log.info("Cache dir for merge base %s exists; reusing", mb.hexsha)
    else:
        log.info("Cache dir for merge base %s does not exist; generating", mb.hexsha)
        build_run_convert(local_repo_path, local_repo, mb.hexsha, seeding_example_args)

    log.info("Making plots...")

    file_base = "%d_%s" % (pr_id, pr.head.sha)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for (plot, plot_type) in csv_file_names:
            prev_df = pandas.read_csv(CACHE_DIR / mb.hexsha / (plot + ".csv"))
            curr_df = pandas.read_csv(CACHE_DIR / pr.head.sha / (plot + ".csv"))

            px = 1 / matplotlib.pyplot.rcParams["figure.dpi"]

            fig = matplotlib.pyplot.figure(figsize=(806 * px, 450 * px))
            ax = fig.subplots()

            prev_df["center"] = (prev_df["bin_left"] + prev_df["bin_right"]) / 2
            prev_df["width"] = prev_df["center"] - prev_df["bin_left"]
            if "trackeff" in plot:
                prev_df = prev_df[prev_df["ntotal"] > 0]

            curr_df["center"] = (curr_df["bin_left"] + curr_df["bin_right"]) / 2
            curr_df["width"] = curr_df["center"] - curr_df["bin_left"]
            if "trackeff" in plot:
                curr_df = curr_df[curr_df["ntotal"] > 0]

            if plot_type == "eff":
                ax.set_ylabel("Efficiency")
            elif plot == "finding_nDuplicated_vs_eta":
                ax.set_ylabel("Duplicate rate")
            elif plot == "finding_nFakeTracks_vs_eta":
                ax.set_ylabel("Fake rate")
            else:
                ax.set_ylabel("Normalized entries")

            if "vs_eta" in plot:
                ax.set_xlabel("$\\eta$")
            elif "vs_phi" in plot:
                ax.set_xlabel("$\\phi$")
            elif "vs_pT" in plot:
                ax.set_xlabel("$p_T$ (GeV)")
            elif "pval" in plot:
                ax.set_xlabel("$p$")
            elif "ndf" in plot:
                ax.set_xlabel("NDF")
            elif "completeness" in plot:
                ax.set_xlabel("Completeness")
            elif "purity" in plot:
                ax.set_xlabel("Purity")

            base_color = 0 if "seeding" in plot else 2

            kwargs = {}

            if plot_type == "hist" or plot_type == "prof":
                kwargs["drawstyle"] = "steps-mid"
            else:
                kwargs["fmt"] = '.'

            if plot_type == "eff":
                xkey = "efficiency"
                prev_sf = 1
                curr_sf = 1
            elif plot_type == "prof":
                xkey = "value"
                prev_sf = 1
                curr_sf = 1
            elif plot_type == "hist":
                xkey = "ntotal"
                prev_sf = prev_df[xkey].sum()
                curr_sf = curr_df[xkey].sum()

            ax.errorbar(prev_df["center"], prev_df[xkey] / prev_sf, xerr=prev_df["width"], yerr=(prev_df["err_low"] / prev_sf, prev_df["err_high"] / prev_sf), capsize=3, label="Reference (%s)" % mb.hexsha[:8], color="C%d" % (base_color + 0), **kwargs)
            ax.errorbar(curr_df["center"], curr_df[xkey] / curr_sf, xerr=curr_df["width"], yerr=(curr_df["err_low"] / curr_sf, curr_df["err_high"] / curr_sf), capsize=2, label="Monitored (%s)" % pr.head.sha[:8], color="C%d" % (base_color + 1), **kwargs)

            ax.set_xlim(xmin=prev_df["bin_left"].min(), xmax=prev_df["bin_right"].max())

            ax.legend()
            fig.tight_layout()
            fig.savefig(pathlib.Path(tmpdirname) / ("%s_%s.png" % (file_base, plot)))

            subprocess.run(["eos", "root://eosuser.cern.ch", "cp", "file://%s/%s_%s.png" % (tmpdirname, file_base, plot), "/eos/user/s/sswatman/traccc_bench/physics/"], check=True)

    log.info("Submitting PR comment...")

    text = "# Physics performance summary\nHere is a summary of the physics performance effects of this PR. Command used:\n\n```\n%s\n```\n\n## Seeding performance\n" % ' '.join(["traccc_seeding_example_cuda"] + seeding_example_args)

    with open(CACHE_DIR / pr.head.sha / "counts.json") as f:
        pr_counts = json.load(f)

    with open(CACHE_DIR / mb.hexsha / "counts.json") as f:
        mb_counts = json.load(f)

    text += "Total number of seeds went from **%d** to **%d** **(%+.1f%%)**\n" % (mb_counts["seeds"], pr_counts["seeds"], 100. * ((pr_counts["seeds"] - mb_counts["seeds"]) / mb_counts["seeds"]))

    for plot in ["seeding_trackeff_vs_pT",
        "seeding_trackeff_vs_eta",
        "seeding_trackeff_vs_phi"]:
        text += "![](https://traccc-bench.web.cern.ch/physics/%s_%s.png?v=1)\n" % (file_base, plot)

    text += "\n## Track finding performance\n"
    
    text += "Total number of found tracks went from **%d** to **%d** **(%+.1f%%)**\n" % (mb_counts["found"], pr_counts["found"], 100. * ((pr_counts["found"] - mb_counts["found"]) / mb_counts["found"]))

    for plot in ["finding_trackeff_vs_pT",
        "finding_trackeff_vs_eta",
        "finding_trackeff_vs_phi", "finding_nDuplicated_vs_eta", "finding_nFakeTracks_vs_eta", "ndf", "pval", "purity", "completeness"]:
        text += "![](https://traccc-bench.web.cern.ch/physics/%s_%s.png?v=1)\n" % (file_base, plot)

    text += "\n> [!NOTE]\n> This is an automated message produced on the explicit request of a human being."

    pr.create_issue_comment(text)

    # To close connections after use
    client.close()


if __name__ == "__main__":
    main()
