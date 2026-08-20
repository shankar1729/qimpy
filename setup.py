import setuptools
import sys
import os


def get_version_cmdclass():
    import versioneer

    return versioneer.get_version(), versioneer.get_cmdclass()


sys.path.append(os.path.dirname(__file__))  # needed for versioneer
version, cmdclass = get_version_cmdclass()
setuptools.setup(version=version, cmdclass=cmdclass)
