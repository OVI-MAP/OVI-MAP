import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DVERSION_INFO={}'.format(self.distribution.get_version()),
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                cmake_toolchain_file = os.environ.get('CMAKE_TOOLCHAIN_FILE')
                # print(f'-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}')
                cmake_args += ['-DCMAKE_TOOLCHAIN_FILE='+cmake_toolchain_file]
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                    cmake_args += ['-DVCPKG_TARGET_TRIPLET=x64-windows']
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(['cmake', ext.sourcedir] + cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='depth_segmentation_py',
    version='1.0.0',
    author='Yang Miao',
    author_email='yamiao@student.ethz.ch',
    description='Pybind version for depth_segmentation',
    long_description='',
    ext_modules=[CMakeExtension('depth_segmentation_py')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    setup_requires=['numpy'],
    install_requires=['numpy'],
)


## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

# from setuptools import setup
# from catkin_pkg.python_setup import generate_distutils_setup

# # fetch values from package.xml
# setup_args = generate_distutils_setup(
#     packages=['depth_segmentation_py'],
#     package_dir={'': 'src'})

# setup(**setup_args)