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
import numpy
import scipy.stats
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
            [pathlib.Path(tmpdirname) / "bin" / "traccc_seeding_example_cuda"]
            + seeding_exec,
            check=True,
            cwd=tmpdirname,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        stdout = result.stdout.decode("utf-8")

        if match := re.search(r"- created \(cuda\)\s+(\d+) seeds", stdout):
            num_seeds = int(match.group(1))
        else:
            raise ValueError("Seed count could not be parsed from stdout!")

        if match := re.search(r"- created \(cuda\)\s+(\d+) found tracks", stdout):
            num_found_tracks = int(match.group(1))
        else:
            raise ValueError("Found track count could not be parsed from stdout!")

        if match := re.search(r"- created \(cuda\)\s+(\d+) fitted tracks", stdout):
            num_fitted_tracks = int(match.group(1))
        else:
            raise ValueError("Fitted track count could not be parsed from stdout!")

        log.info("Making new directory in cache...")
        try:
            os.mkdir(CACHE_DIR / commit_hash)
            log.info("Cache dir succesfully created")
        except FileExistsError:
            log.info("Cache dir already exists")

        with open(CACHE_DIR / commit_hash / "counts.json", "w") as f:
            json.dump(
                {
                    "seeds": num_seeds,
                    "found": num_found_tracks,
                    "fitted": num_fitted_tracks,
                },
                f,
            )

        for f in ["seeding", "finding", "postfit_finding", "fitting"]:
            fn = "performance_track_%s.root" % f
            shutil.copy(pathlib.Path(tmpdirname) / fn, CACHE_DIR / commit_hash)

            try:
                os.mkdir(CACHE_DIR / commit_hash / f)
            except FileExistsError:
                pass

            subprocess.run(
                [
                    "./teff_to_csv/teff_to_csv",
                    str(CACHE_DIR / commit_hash / fn),
                    str(CACHE_DIR / commit_hash / f),
                ],
                check=True,
            )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Initiating Github API client...")

    auth = github.Auth.Token(os.environ["GHTOKEN"])

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

    pr_sha = pr.head.sha

    local_repo.git.checkout(pr_sha)

    mbl = local_repo.merge_base("main", pr_sha)
    assert len(mbl) == 1
    mb = mbl[0]

    mb_sha = mb.hexsha

    cwd = "/mnt/ssd1/sswatman/benchtraccc/traccc/extras/benchmark"

    csv_file_names = [
        ("seeding", "seeding_trackeff_vs_pT", "eff"),
        ("seeding", "seeding_trackeff_vs_eta", "eff"),
        ("seeding", "seeding_trackeff_vs_phi", "eff"),
        ("finding", "finding_trackeff_vs_pT", "eff"),
        ("finding", "finding_trackeff_vs_eta", "eff"),
        ("finding", "finding_trackeff_vs_phi", "eff"),
        ("finding", "finding_nDuplicated_vs_eta", "prof"),
        ("finding", "finding_nFakeTracks_vs_eta", "prof"),
        ("finding", "ndf", "hist"),
        ("finding", "pval", "hist"),
        ("finding", "purity", "hist"),
        ("finding", "completeness", "hist"),
        ("postfit_finding", "finding_trackeff_vs_pT", "eff"),
        ("postfit_finding", "finding_trackeff_vs_eta", "eff"),
        ("postfit_finding", "finding_trackeff_vs_phi", "eff"),
        ("postfit_finding", "finding_nDuplicated_vs_eta", "prof"),
        ("postfit_finding", "finding_nFakeTracks_vs_eta", "prof"),
        ("fitting", "res_d0", "hist"),
        ("fitting", "res_z0", "hist"),
        ("fitting", "res_phi", "hist"),
        ("fitting", "res_qop", "hist"),
        ("fitting", "res_qopT", "hist"),
        ("fitting", "res_qopz", "hist"),
        ("fitting", "res_theta", "hist"),
        ("fitting", "pull_d0", "hist"),
        ("fitting", "pull_z0", "hist"),
        ("fitting", "pull_phi", "hist"),
        ("fitting", "pull_qop", "hist"),
        ("fitting", "pull_theta", "hist"),
        ("fitting", "ndf", "hist"),
        ("fitting", "pval", "hist"),
        ("postfit_finding", "purity", "hist"),
        ("postfit_finding", "completeness", "hist"),
    ]

    ratio_eff_plots = [
        (("seeding", "seeding_trackeff_vs_pT"), ("finding", "finding_trackeff_vs_pT")),
        (
            ("seeding", "seeding_trackeff_vs_eta"),
            ("finding", "finding_trackeff_vs_eta"),
        ),
        (
            ("seeding", "seeding_trackeff_vs_phi"),
            ("finding", "finding_trackeff_vs_phi"),
        ),
        (
            ("finding", "finding_trackeff_vs_pT"),
            ("postfit_finding", "finding_trackeff_vs_pT"),
        ),
        (
            ("finding", "finding_trackeff_vs_eta"),
            ("postfit_finding", "finding_trackeff_vs_eta"),
        ),
        (
            ("finding", "finding_trackeff_vs_phi"),
            ("postfit_finding", "finding_trackeff_vs_phi"),
        ),
    ]

    pr_dir = CACHE_DIR / str(pr_sha)
    mb_dir = CACHE_DIR / str(mb_sha)

    seeding_example_args = [
        "--input-directory=/data/Acts/odd-simulations-20240506/geant4_ttbar_mu200",
        "--digitization-file=geometries/odd/odd-digi-geometric-config.json",
        "--detector-file=geometries/odd/odd-detray_geometry_detray.json",
        "--grid-file=geometries/odd/odd-detray_surface_grids_detray.json",
        "--material-file=geometries/odd/odd-detray_material_detray.json",
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
        "--seedfinder-vertex-range=-150:150",
    ]

    if pr_dir.is_dir() and all(
        (pr_dir / u / (x + ".csv")).is_file() for (u, x, _) in csv_file_names
    ):
        log.info("Cache dir for PR commit %s exists; reusing", pr_sha)
    else:
        log.info("Cache dir for PR commit %s does not exist; generating", pr_sha)
        build_run_convert(local_repo_path, local_repo, pr_sha, seeding_example_args)

    if mb_dir.is_dir() and all(
        (mb_dir / u / (x + ".csv")).is_file() for (u, x, _) in csv_file_names
    ):
        log.info("Cache dir for merge base %s exists; reusing", mb_sha)
    else:
        log.info("Cache dir for merge base %s does not exist; generating", mb_sha)
        build_run_convert(local_repo_path, local_repo, mb_sha, seeding_example_args)

    log.info("Making plots...")

    file_base = "%d_%s" % (pr_id, pr_sha)

    px = 1 / matplotlib.pyplot.rcParams["figure.dpi"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        for data_cat, plot, plot_type in csv_file_names:
            prev_df = pandas.read_csv(CACHE_DIR / mb_sha / data_cat / (plot + ".csv"))
            curr_df = pandas.read_csv(CACHE_DIR / pr_sha / data_cat / (plot + ".csv"))

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

            plot_normal = False

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
            elif "res_d0" == plot:
                ax.set_xlabel("Residual $d_0$")
            elif "res_z0" == plot:
                ax.set_xlabel("Residual $z_0$")
            elif "res_phi" == plot:
                ax.set_xlabel("Residual $\\phi$")
            elif "res_qop" == plot:
                ax.set_xlabel("Residual $q/p$")
            elif "res_qopT" == plot:
                ax.set_xlabel("Residual $q/p_T$")
            elif "res_theta" == plot:
                ax.set_xlabel("Residual $\\theta$")
            elif "res_qopz" == plot:
                ax.set_xlabel("Residual $q/p_z$")
            elif "pull_d0" == plot:
                plot_normal = True
                ax.set_xlabel("Pull $d_0$")
            elif "pull_z0" == plot:
                plot_normal = True
                ax.set_xlabel("Pull $z_0$")
            elif "pull_phi" == plot:
                plot_normal = True
                ax.set_xlabel("Pull $\\phi$")
            elif "pull_qop" == plot:
                plot_normal = True
                ax.set_xlabel("Pull $q/p$")
            elif "pull_theta" == plot:
                plot_normal = True
                ax.set_xlabel("Pull $\\theta$")

            if data_cat == "seeding":
                base_color = 0
            elif data_cat == "finding":
                base_color = 2
            else:
                base_color = 4

            kwargs = {}
            prev_kwargs = {}
            curr_kwargs = {}

            if plot_type == "hist" or plot_type == "prof":
                kwargs["drawstyle"] = "steps-mid"
            else:
                kwargs["fmt"] = "."
                prev_kwargs["xerr"] = prev_df["width"]
                curr_kwargs["xerr"] = curr_df["width"]

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

            if "pull" in plot or "residual" in plot:
                prev_mean = numpy.average(prev_df["center"], weights=prev_df[xkey])
                curr_mean = numpy.average(curr_df["center"], weights=curr_df[xkey])
                prev_std = numpy.sqrt(
                    numpy.average(
                        (prev_df["center"] - prev_mean) ** 2, weights=prev_df[xkey]
                    )
                )
                curr_std = numpy.sqrt(
                    numpy.average(
                        (curr_df["center"] - curr_mean) ** 2, weights=curr_df[xkey]
                    )
                )

                prev_lbl = "Reference (%s); $\\mu = %.3f$, $\\sigma = %.3f$" % (
                    mb_sha[:8],
                    prev_mean,
                    prev_std,
                )
                curr_lbl = "Monitored (%s); $\\mu = %.3f$, $\\sigma = %.3f$" % (
                    pr_sha[:8],
                    curr_mean,
                    curr_std,
                )
            else:
                prev_lbl = "Reference (%s)" % mb_sha[:8]
                curr_lbl = "Monitored (%s)" % pr_sha[:8]

            if plot_normal:
                x = numpy.linspace(-5, 5, 200)
                ax.plot(x, scipy.stats.norm.pdf(x, 0, 1), label="Ideal", color="black")

            if plot_type == "hist":
                prev_sf *= (prev_df["bin_right"] - prev_df["bin_left"])[0]
                curr_sf *= (curr_df["bin_right"] - curr_df["bin_left"])[0]

            ax.errorbar(
                prev_df["center"],
                prev_df[xkey] / prev_sf,
                yerr=(prev_df["err_low"] / prev_sf, prev_df["err_high"] / prev_sf),
                capsize=3,
                label=prev_lbl,
                color="C%d" % (base_color + 0),
                **kwargs,
                **prev_kwargs,
            )
            ax.errorbar(
                curr_df["center"],
                curr_df[xkey] / curr_sf,
                yerr=(curr_df["err_low"] / curr_sf, curr_df["err_high"] / curr_sf),
                capsize=2,
                label=curr_lbl,
                color="C%d" % (base_color + 1),
                **kwargs,
                **curr_kwargs,
            )

            ax.set_xlim(xmin=prev_df["bin_left"].min(), xmax=prev_df["bin_right"].max())

            ax.legend()
            fig.tight_layout()
            if data_cat == "postfit_finding":
                n_data_cat = "fitting"
            else:
                n_data_cat = data_cat
            fig.savefig(
                pathlib.Path(tmpdirname)
                / ("%s_%s_%s.png" % (file_base, n_data_cat, plot))
            )

            subprocess.run(
                [
                    "eos",
                    "root://eosuser.cern.ch",
                    "cp",
                    "file://%s/%s_%s_%s.png"
                    % (tmpdirname, file_base, n_data_cat, plot),
                    "/eos/user/s/sswatman/traccc_bench/physics/",
                ],
                check=True,
            )

        for (from_data_cat, from_plot), (to_data_cat, to_plot) in ratio_eff_plots:
            prev_from_df = pandas.read_csv(
                CACHE_DIR / mb_sha / from_data_cat / (from_plot + ".csv")
            )
            curr_from_df = pandas.read_csv(
                CACHE_DIR / pr_sha / from_data_cat / (from_plot + ".csv")
            )
            prev_to_df = pandas.read_csv(
                CACHE_DIR / mb_sha / to_data_cat / (to_plot + ".csv")
            )
            curr_to_df = pandas.read_csv(
                CACHE_DIR / pr_sha / to_data_cat / (to_plot + ".csv")
            )

            fig = matplotlib.pyplot.figure(figsize=(806 * px, 450 * px))
            ax = fig.subplots()

            prev_from_df["center"] = (
                prev_from_df["bin_left"] + prev_from_df["bin_right"]
            ) / 2
            prev_from_df["width"] = prev_from_df["center"] - prev_from_df["bin_left"]
            prev_mask = prev_from_df["ntotal"] > 0
            prev_ratio = (
                prev_to_df[prev_mask]["efficiency"]
                / prev_from_df[prev_mask]["efficiency"]
            )

            prev_yerr_low = prev_ratio * numpy.sqrt(
                (
                    prev_from_df[prev_mask]["err_low"]
                    / prev_from_df[prev_mask]["efficiency"]
                )
                ** 2
                + (
                    prev_to_df[prev_mask]["err_low"]
                    / prev_to_df[prev_mask]["efficiency"]
                )
                ** 2
            )
            prev_yerr_high = prev_ratio * numpy.sqrt(
                (
                    prev_from_df[prev_mask]["err_high"]
                    / prev_from_df[prev_mask]["efficiency"]
                )
                ** 2
                + (
                    prev_to_df[prev_mask]["err_high"]
                    / prev_to_df[prev_mask]["efficiency"]
                )
                ** 2
            )

            curr_from_df["center"] = (
                curr_from_df["bin_left"] + curr_from_df["bin_right"]
            ) / 2
            curr_from_df["width"] = curr_from_df["center"] - curr_from_df["bin_left"]
            curr_mask = curr_from_df["ntotal"] > 0
            curr_ratio = (
                curr_to_df[curr_mask]["efficiency"]
                / curr_from_df[curr_mask]["efficiency"]
            )

            curr_yerr_low = curr_ratio * numpy.sqrt(
                (
                    curr_from_df[curr_mask]["err_low"]
                    / curr_from_df[curr_mask]["efficiency"]
                )
                ** 2
                + (
                    curr_to_df[curr_mask]["err_low"]
                    / curr_to_df[curr_mask]["efficiency"]
                )
                ** 2
            )
            curr_yerr_high = curr_ratio * numpy.sqrt(
                (
                    curr_from_df[curr_mask]["err_high"]
                    / curr_from_df[curr_mask]["efficiency"]
                )
                ** 2
                + (
                    curr_to_df[curr_mask]["err_high"]
                    / curr_to_df[curr_mask]["efficiency"]
                )
                ** 2
            )

            ax.set_ylabel("Relative efficiency")

            if "vs_eta" in from_plot:
                ax.set_xlabel("$\\eta$")
            elif "vs_phi" in from_plot:
                ax.set_xlabel("$\\phi$")
            elif "vs_pT" in from_plot:
                ax.set_xlabel("$p_T$ (GeV)")

            if from_data_cat == "seeding":
                base_color = 6
            elif from_data_cat == "finding":
                base_color = 8
            else:
                base_color = 10

            prev_lbl = "Reference (%s)" % mb_sha[:8]
            curr_lbl = "Monitored (%s)" % pr_sha[:8]

            ax.errorbar(
                prev_from_df[prev_mask]["center"],
                prev_ratio,
                yerr=(prev_yerr_low, prev_yerr_high),
                capsize=3,
                label=prev_lbl,
                color="C%d" % (base_color + 0),
                fmt=".",
                xerr=prev_from_df[prev_mask]["width"],
            )
            ax.errorbar(
                curr_from_df[curr_mask]["center"],
                curr_ratio,
                yerr=(curr_yerr_low, curr_yerr_high),
                capsize=2,
                label=curr_lbl,
                color="C%d" % (base_color + 1),
                fmt=".",
                xerr=curr_from_df[curr_mask]["width"],
            )

            ax.set_xlim(
                xmin=prev_from_df[prev_mask]["bin_left"].min(),
                xmax=prev_from_df[prev_mask]["bin_right"].max(),
            )

            ax.legend()
            fig.tight_layout()

            vs_name = "%svs%s" % (from_data_cat, to_data_cat)

            fig.savefig(
                pathlib.Path(tmpdirname)
                / ("%s_%s_%s.png" % (file_base, vs_name, from_plot))
            )

            subprocess.run(
                [
                    "eos",
                    "root://eosuser.cern.ch",
                    "cp",
                    "file://%s/%s_%s_%s.png"
                    % (tmpdirname, file_base, vs_name, from_plot),
                    "/eos/user/s/sswatman/traccc_bench/physics/",
                ],
                check=True,
            )

    log.info("Submitting PR comment...")

    text = (
        "# Physics performance summary\nHere is a summary of the physics performance effects of this PR. Command used:\n\n```\n%s\n```\n\n## Seeding performance\n"
        % " ".join(["traccc_seeding_example_cuda"] + seeding_example_args)
    )

    with open(CACHE_DIR / pr_sha / "counts.json") as f:
        pr_counts = json.load(f)

    with open(CACHE_DIR / mb_sha / "counts.json") as f:
        mb_counts = json.load(f)

    version = 17

    text += (
        "Total number of seeds went from **%d** to **%d** **(%+.1f%%)**\n<details><summary>Seeding plots</summary>\n\n"
        % (
            mb_counts["seeds"],
            pr_counts["seeds"],
            100.0 * ((pr_counts["seeds"] - mb_counts["seeds"]) / mb_counts["seeds"]),
        )
    )

    for u, plot, _ in csv_file_names:
        if u != "seeding":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_seeding_%s.png?v=%d)\n"
            % (file_base, plot, version)
        )

    text += "\n</details>\n\n## Track finding performance\n"

    text += (
        "Total number of found tracks went from **%d** to **%d** **(%+.1f%%)**\n<details><summary>Finding plots</summary>\n\n"
        % (
            mb_counts["found"],
            pr_counts["found"],
            100.0 * ((pr_counts["found"] - mb_counts["found"]) / mb_counts["found"]),
        )
    )

    for u, plot, _ in csv_file_names:
        if u != "finding":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_finding_%s.png?v=%d)\n"
            % (file_base, plot, version)
        )

    text += "\n</details>\n\n## Track fitting performance\n"

    # text += "Total number of fitted tracks went from **%d** to **%d** **(%+.1f%%)**\n<details><summary>Fitting plots</summary>\n\n" % (mb_counts["fitted"], pr_counts["fitted"], 100. * ((pr_counts["fitted"] - mb_counts["fitted"]) / mb_counts["fitted"]))
    text += "<details><summary>Fitting plots</summary>\n\n"

    for u, plot, _ in csv_file_names:
        if u == "postfit_finding":
            u = "fitting"
        if u != "fitting":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_fitting_%s.png?v=%d)\n"
            % (file_base, plot, version)
        )

    text += "\n</details>\n\n## Seeding to track finding relative performance\n"
    text += "<details><summary>Seeding to track finding plots</summary>\n\n"

    for (from_data_cat, from_plot), (to_data_cat, to_plot) in ratio_eff_plots:
        if from_data_cat != "seeding":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_seedingvsfinding_%s.png?v=%d)\n"
            % (file_base, from_plot, version)
        )

    text += "\n</details>\n\n## Track finding to track fitting relative performance\n"
    text += "<details><summary>Track finding to track fitting plots</summary>\n\n"

    for (from_data_cat, from_plot), (to_data_cat, to_plot) in ratio_eff_plots:
        if from_data_cat != "finding":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_findingvspostfit_finding_%s.png?v=%d)\n"
            % (file_base, from_plot, version)
        )

    text += "\n</details>\n\n> [!NOTE]\n> This is an automated message produced on the explicit request of a human being."

    pr.create_issue_comment(text)

    # To close connections after use
    client.close()


if __name__ == "__main__":
    main()
