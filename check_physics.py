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
import traccc_physics_plots.plot
import traccc_physics_plots.run


log = logging.getLogger("traccc_bench_bot.check_physics")


CACHE_DIR = pathlib.Path("/mnt/ssd1/sswatman/traccc-physics-cache")


def build(tmpdirname, src_dir, local_repo, commit_hash):
    log.info("Building repository for commit %s...", commit_hash)

    local_repo.git.checkout(commit_hash)

    log.info("Building into temporary directory %s", tmpdirname)

    log.info("Running configuration step...")
    traccc_bench_tools.build.configure(src_dir, tmpdirname, root=True)
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

    result = subprocess.run(
        build_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        log.error("stdout:")
        log.error(result.stdout)
        log.error("stderr:")
        log.error(result.stderr)
        raise RuntimeError("Subprocess failed")


def get_branch_head_and_merge_base(repo, local_repo, pr_id):
    log.info("Retrieving PR information...")

    pr = repo.get_pull(pr_id)

    log.info("Updating local repository")

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

    return pr_sha, mb_sha


def create_github_comment(pr_sha, mb_sha, file_base, seeding_example_args):
    text = (
        "# Physics performance summary\nHere is a summary of the physics performance effects of this PR. Command used:\n\n```\n%s\n```\n\n## Seeding performance\n"
        % " ".join(["traccc_seeding_example_cuda"] + seeding_example_args)
    )

    with open(CACHE_DIR / pr_sha / "counts.json") as f:
        pr_counts = json.load(f)

    with open(CACHE_DIR / mb_sha / "counts.json") as f:
        mb_counts = json.load(f)

    version = 22

    text += (
        "Total number of seeds went from **%d** to **%d** **(%+.1f%%)**\n<details><summary>Seeding plots</summary>\n\n"
        % (
            mb_counts["seeds"],
            pr_counts["seeds"],
            100.0 * ((pr_counts["seeds"] - mb_counts["seeds"]) / mb_counts["seeds"]),
        )
    )

    for u, plot, _ in traccc_physics_plots.plot.PLOT_NAMES:
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

    for u, plot, _ in traccc_physics_plots.plot.PLOT_NAMES:
        if u != "finding":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_finding_%s.png?v=%d)\n"
            % (file_base, plot, version)
        )

    text += "\n</details>\n\n## Track fitting performance\n"

    text += "<details><summary>Fitting plots</summary>\n\n"

    for u, plot, _ in traccc_physics_plots.plot.PLOT_NAMES:
        if u != "fitting":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_fitting_%s.png?v=%d)\n"
            % (file_base, plot, version)
        )

    text += "\n</details>\n\n## Seeding to track finding relative performance\n"
    text += "<details><summary>Seeding to track finding plots</summary>\n\n"

    for (from_data_cat, from_plot), (
        to_data_cat,
        to_plot,
    ) in traccc_physics_plots.plot.RATIO_PLOTS:
        if from_data_cat != "seeding":
            continue
        text += (
            "![](https://traccc-bench.web.cern.ch/physics/%s_seedingvsfinding_%s.png?v=%d)\n"
            % (file_base, from_plot, version)
        )

    text += "\n</details>\n\n> [!NOTE]\n> This is an automated message produced on the explicit request of a human being."

    return text


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Initiating Github API client...")

    auth = github.Auth.Token(os.environ["GHTOKEN"])
    root_to_csv = os.environ["ROOT_TO_CSV"]

    client = github.Github(auth=auth)

    repo = client.get_repo("acts-project/traccc")

    pr_id = int(sys.argv[1])
    pr = repo.get_pull(pr_id)

    local_repo_path = "/mnt/ssd1/sswatman/benchtraccc/traccc_subject/"
    local_repo = git.Repo(local_repo_path)

    pr_sha, mb_sha = get_branch_head_and_merge_base(repo, local_repo, pr_id)

    for hash in [pr_sha, mb_sha]:
        dir = CACHE_DIR / hash

        if dir.is_dir() and all(
            (dir / u / (x + ".csv")).is_file()
            for (u, x, _) in traccc_physics_plots.plot.PLOT_NAMES
        ):
            log.info("Cache dir for PR commit %s exists; reusing", hash)
        else:
            log.info("Making new directory in cache...")
            try:
                os.mkdir(dir)
                log.info("Cache dir succesfully created")
            except FileExistsError:
                log.info("Cache dir already exists")

            log.info("Cache dir for PR commit %s does not exist; generating", hash)
            with tempfile.TemporaryDirectory() as tmpdirname:
                build(tmpdirname, local_repo_path, local_repo, hash)
                traccc_physics_plots.run.run_convert_seeding_example(
                    pathlib.Path(tmpdirname) / "bin" / "traccc_seeding_example_cuda",
                    traccc_physics_plots.run.SEEDING_EXAMPLE_ARGS,
                    tmpdirname,
                    dir,
                    root_to_csv_exec=root_to_csv,
                )

    log.info("Making plots...")

    PX = 1 / matplotlib.pyplot.rcParams["figure.dpi"]

    FIG_KWARGS = {
        "figsize": (806 * PX, 450 * PX),
    }

    file_base = "%d_%s" % (pr_id, pr_sha)

    with tempfile.TemporaryDirectory() as tmpdirname:
        plot_candidate_names = {
            "ref": (CACHE_DIR / mb_sha, ("Reference (%s)" % mb_sha[:8])),
            "mon": (CACHE_DIR / pr_sha, ("Monitored (%s)" % pr_sha[:8])),
        }

        files_to_copy = traccc_physics_plots.plot.make_plots(
            plot_candidate_names, tmpdirname, fig_kwargs=FIG_KWARGS, file_base=file_base
        )

        log.info("Uploading files to EOS...")

        subprocess.run(
            [
                "eos",
                "root://eosuser.cern.ch",
                "cp",
                *["file://%s" % f for f in files_to_copy],
                "/eos/user/s/sswatman/traccc_bench/physics/",
            ],
            check=True,
        )

    log.info("Submitting PR comment...")

    text = create_github_comment(
        pr_sha, mb_sha, file_base, traccc_physics_plots.run.SEEDING_EXAMPLE_ARGS
    )
    pr.create_issue_comment(text)

    # To close connections after use
    client.close()


if __name__ == "__main__":
    main()
