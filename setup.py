from setuptools import setup
import string
import os.path as op
from setuptools_scm import get_version


def local_version(version):
    """
    Patch in a version that can be uploaded to test PyPI
    """
    scm_version = get_version()
    if "dev" in scm_version:
        gh_in_int = []
        for char in version.node:
            if char.isdigit():
                gh_in_int.append(str(char))
            else:
                gh_in_int.append(str(string.ascii_letters.find(char)))
        return "".join(gh_in_int)
    else:
        return ""


opts = dict(
    use_scm_version={
        "root": ".", "relative_to": __file__,
        "write_to": op.join("ndslib", "version.py"),
        "local_scheme": local_version},
    package_data={'ndslib': [op.join('templates', '*')]}
    )


if __name__ == '__main__':
    setup(**opts)
