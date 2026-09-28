# Publishing this portfolio

Suggested repository name: `traffic-analytics-libcity`.

Publish the `Bigscity-LibCity-master` directory as the repository root. Keep the existing upstream license and attribution. The original data and notebooks are located in a sibling directory and are not needed for the synthetic demo. Do not add that sibling directory to this repository.

The upstream source was provided without Git metadata. This portfolio does not claim to be an authenticated GitHub fork or identify a verified upstream commit. A GitHub repository has not been created or pushed by this task.

Before a first push, review `git status --short` and the staged diff. Include the source, tests, documentation and benchmark reports. The `.gitignore` excludes generated outputs, environments and caches. GitHub Actions is configured, but its hosted execution is unverified until a push.

For reproducibility, the original Seq2Seq file can be recovered for comparison by reversing `docs/portfolio/seq2seq_changes.patch`. The before report was recorded prior to applying those changes. Do not reverse the patch in the working project merely to run the normal demo.
