import github
import git
import subprocess
import os
import sys
import pandas
import numpy


def render_time(t):
    if numpy.isnan(t):
        return "&mdash;"
    if t >= 1.0:
        return "%.2f s" % t
    elif t >= 0.001:
        return "%.2f ms" % (t * 1000)
    elif t >= 0.000001:
        return "%.2f μs" % (t * 1000000)
    elif t >= 0.000000001:
        return "%.2f ns" % (t * 1000000000)


def main():
    auth = github.Auth.Token(os.environ['GHTOKEN'])

    client = github.Github(auth=auth)

    repo = client.get_repo("acts-project/traccc")

    pr_id = int(sys.argv[1])

    pr = repo.get_pull(pr_id)

    local_repo = git.Repo("/mnt/ssd1/sswatman/benchtraccc/traccc_subject/")

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

    file_base = "%d_%s" % (pr_id, pr.head.sha)

    subprocess.run(["python", "benchmark.py", "--ncu-wrapper", "gpu_lock A5000", "--cc=86", "-j=96", "-e=25", "--num-sm=64", "--num-threads-per-sm=1536", "--from=%s~" % (str(mb.hexsha)), "--to=%s" % (pr.head.sha), "/mnt/ssd1/sswatman/benchtraccc/traccc_subject", "%s.csv" % (file_base), "/data/Acts/odd-simulations-20240509/geant4_ttbar_mu200"], check=True, cwd=cwd)


    subprocess.run(["python", "plot.py", "%s.csv" % (file_base), "%s.png" % (file_base), "--threshold=0.0005"], check=True, cwd=cwd)

    subprocess.run(["eos", "root://eosuser.cern.ch", "cp", "file://%s/%s.png" % (cwd, file_base), "/eos/user/s/sswatman/traccc_bench"], check=True)

    text = "# Performance summary\nHere is a summary of the performance effects of this PR:\n## Graphical\n![](https://traccc-bench.web.cern.ch/%s.png)\n## Tabular\n" % (file_base)

    df = pandas.read_csv(cwd + "/%s.csv" % (file_base))

    df_before = df[df["commit"] == str(mb)]
    df_after = df[df["commit"] == pr.head.sha]

    kernels = []

    df_merge = pandas.merge(df_before, df_after, how="outer", on="kernel", suffixes=("_before", "_after"))

    for i in df_merge.iloc:
        kernels.append((i["kernel"], i["rec_throughput_before"], i["rec_throughput_after"], i["latency_before"], i["latency_after"]))

    kernels.sort(key=lambda x: x[2], reverse=True)

    text += "<table><thead><tr><th rowspan=\"2\">Kernel</th><th align=\"center\" colspan=\"3\">Reciprocal Throughput</th><th align=\"center\" colspan=\"2\">Parallelism</th></tr>"
    text += "<tr><th align=\"center\">{0}</th><th align=\"center\">{1}</th><th align=\"right\">Delta</th><th align=\"center\">{0}</th><th align=\"center\">{1}</th></tr></thead>".format(str(mb)[:8], pr.head.sha[:8])
    text += "<tbody>"

    bigreg = False

    total_before = 0.0
    total_after = 0.0
    total_lat_before = 0.0
    total_lat_after = 0.0

    for (k, b, a, lb, la) in kernels:
        if numpy.isnan(b) or numpy.isnan(a):
            delta = "&mdash;"
        else:
            rf = (100. * (a - b) / b)
            if rf >= 10.:
                bigreg = True
            delta = "%.1f%%" % (100. * (a - b) / b)
        if not numpy.isnan(b):
            total_before += b
        if not numpy.isnan(a):
            total_after += a
        if not numpy.isnan(lb):
            total_lat_before += lb
        if not numpy.isnan(la):
            total_lat_after += la
        pb = lb / b
        pa = la / a
        text += "<tr><td><code class=\"notranslate\">%s</code></td><td align=\"right\">%s</td><td align=\"right\">%s</td><td align=\"right\">%s</td><td align=\"right\">%.2f</td><td align=\"right\">%.2f</td></tr>" % (k, render_time(b), render_time(a), delta, pb, pa)
    
    text += "<tr><td><strong>Total</strong></td><td align=\"right\"><strong>%s</strong></td><td align=\"right\"><strong>%s</strong></td><td align=\"right\"><strong>%.1f%%</strong></td><td align=\"right\"><strong>%.2f</strong></td><td align=\"right\"><strong>%.2f</strong></td></tr>" % (render_time(total_before), render_time(total_after), (100. * (total_after - total_before) / total_before), total_lat_before / total_before, total_lat_after / total_after)
    text += "</table>\n\n"

    text += "> [!IMPORTANT]\n> All metrics in this report are given as reciprocal throughput, _not_ as wallclock runtime.\n\n"

    if bigreg:
        text += "> [!WARNING]\n> At least one kernel incurred a significant performance regression.\n\n"

    text += "> [!NOTE]\n> This is an automated message produced upon the explicit request of a human being."

    pr.create_issue_comment(text)

    # To close connections after use
    client.close()


if __name__ == "__main__":
    main()
