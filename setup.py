import setuptools
from torch.utils import cpp_extension
import sys
import os


def get_version_cmdclass():
    import versioneer
    return versioneer.get_version(), versioneer.get_cmdclass()


sys.path.append(os.path.dirname(__file__))  # needed for versioneer
version, cmdclass = get_version_cmdclass()
cmdclass['build_ext'] = cpp_extension.BuildExtension

ext_modules = [
    cpp_extension.CppExtension(
        'qimpy.utils._bufferview',
        ['src/qimpy/utils/_bufferview.cpp'])]

setuptools.setup(
    version=version,
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
