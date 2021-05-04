import setuptools
import versioneer
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CppExtension(
        'qimpy.utils._bufferview',
        ['src/qimpy/utils/_bufferview.cpp'])]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = cpp_extension.BuildExtension

setuptools.setup(
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
