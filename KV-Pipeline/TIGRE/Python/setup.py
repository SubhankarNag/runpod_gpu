import copy
import glob
import os
from os.path import join as pjoin
import re
import subprocess
import sys
import shutil
import re

from Cython.Distutils import build_ext
import numpy
from setuptools import setup, find_packages, Extension


# Macros for compilation
define_macros=[("IS_FOR_PYTIGRE", None)]
no_pinned=False
if len(sys.argv)>2:
    if "--no_pinned_memory" in sys.argv[2:] :
        no_pinned=True
        sys.argv.pop(sys.argv.index("--no_pinned_memory"))
 
if no_pinned:
    define_macros.append(("NO_PINNED_MEMORY",None))     

IS_WINDOWS = sys.platform == "win32"


# Code from https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py

CC_COMPATIBILITY_TABLE = [
    # gencode, code, support_begin, suport_end
    (20, 20,    1,  1.0),
    (30, 30,    1, 11.0), 
    (35, 35,    1, 12.0),
    (37, 37,    1, 12.0), 
    (50, 50,  6.5, 999), 
    (52, 52,  6.5, 999 ), 
    (60, 60,  8.0, 999 ), 
    (61, 61,  8.0, 999 ), 
    (70, 70,  9.0, 999 ), 
    (75, 75, 10.0, 999 ), # From CUDA 10
    (80, 80, 11.0, 999 ), # From CUDA 11.0
    (86, 86, 11.1, 999 ), # From CUDA 11.1
    (87, 87, 11.5, 999 ), 
    (90, 90, 11.8, 999 ), # From CUDA 12
]

COMPUTE_CAPABILITY_ARGS = [
    "-gencode=arch=compute_70,code=compute_70", # allows forward compiling
    "--ptxas-options=-v",
    "-c",
    "--default-stream=per-thread",
]


def get_cuda_version(cuda_home):
    """Locate the CUDA version."""
    # Try version.json (CUDA >= 12.x)
    version_json = os.path.join(cuda_home, "version.json")
    if os.path.isfile(version_json):
        try:
            with open(version_json, "r") as f:
                data = json.load(f)
            # Most Spack/NVIDIA layouts:
            if "cuda" in data and "version" in data["cuda"]:
                return data["cuda"]["version"][:4]
        except Exception:
            pass

    # Try version.txt (older CUDA)
    version_txt = os.path.join(cuda_home, "version.txt")
    if os.path.isfile(version_txt):
        try:
            with open(version_txt) as f:
                version_str = f.readline().strip()
                return version_str.split(" ")[2][:4]
        except Exception:
            pass

    # Try nvcc (if available)
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.isfile(nvcc_path):
        try:
            version_str = subprocess.check_output([nvcc_path, "--version"]).decode()
            idx = version_str.find("release")
            return version_str[idx + 8 : idx + 12]
        except Exception:
            pass

    try:
        out = subprocess.check_output(["nvcc", "--version"]).decode()
        import re
        match = re.search(r"release (\d+\.\d+)", out)
        if match:
            return match.group(1)
    except:
        pass

    raise RuntimeError("Cannot read CUDA version (version.json, version.txt, or nvcc)")


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'include' and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME or CUDA_PATH env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        try:
            which = "where" if IS_WINDOWS else "which"
            nvcc = subprocess.check_output([which, "nvcc"]).decode().rstrip("\r\n")
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except subprocess.CalledProcessError:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    # If there any errors, try to hardcode the path
    # cuda_home = ""
    version = get_cuda_version(cuda_home)
    cudaconfig = {
        "home": cuda_home,
        "include": pjoin(cuda_home, "include"),
        "lib64": pjoin(cuda_home, pjoin("lib", "x64") if IS_WINDOWS else "lib64"),
    }
    if not all([os.path.exists(v) for v in cudaconfig.values()]):
        raise EnvironmentError(
            "The CUDA  path could not be located in $PATH, $CUDA_HOME or $CUDA_PATH. "
            "Either add it to your path, or set $CUDA_HOME or $CUDA_PATH."
        )
    print(cudaconfig)
    return cudaconfig, version


def _is_cuda_file(path):
    return os.path.splitext(path)[1] in [".cu", ".cuh"]


CUDA, CUDA_VERSION = locate_cuda()

cuda_version = 12.1
try:
    cuda_version = float(CUDA_VERSION)
except ValueError:
    cuda_list = re.findall(r'\d+', CUDA_VERSION)
    cuda_version = float( str(cuda_list[0] + '.' + cuda_list[1]))

# Insert CUDA arguments depedning on the version
for item in CC_COMPATIBILITY_TABLE:
    support_begin = item[2]
    support_end   = item[3]
    if cuda_version < support_begin:
        continue
    if cuda_version >= support_end:
        continue
    str_arg = f"-gencode=arch=compute_{item[0]},code=sm_{item[1]}"
    COMPUTE_CAPABILITY_ARGS.insert(0, str_arg)

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    NUMPY_INCLUDE = numpy.get_include()
except AttributeError:
    NUMPY_INCLUDE = numpy.get_numpy_include()


COMMON_MSVC_FLAGS = ["/MD", "/wd4819", "/EHsc"]


COMMON_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
]


def _join_cuda_home(*paths):
    return os.path.join(CUDA["home"], *paths)


class BuildExtension(build_ext):
    """
    A custom :mod:`Cython.Distutils` build extension .

    This :class:`Cython.Distutils.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++11``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``nvcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Returns an alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """

        def init_with_options(*args, **kwargs):
            kwargs = kwargs.copy()
            kwargs.update(options)
            return cls(*args, **kwargs)

        return init_with_options

    def __init__(self, *args, **kwargs):
        build_ext.__init__(self, *args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

    def build_extensions(self):
        # Register .cu and .cuh as valid source extensions.
        self.compiler.src_extensions += [".cu", ".cuh"]
        # Save the original _compile method for later.
        if self.compiler.compiler_type == "msvc":
            self.compiler._cpp_extensions += [".cu", ".cuh"]
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    print("2 - src : " + src)
                    nvcc = _join_cuda_home("bin", "nvcc")
                    print("4 - " + nvcc)
                    if not isinstance(nvcc, list):
                        nvcc = [nvcc]
                    self.compiler.set_executable("compiler_so", nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags["nvcc"]
                    cflags = (
                        COMMON_NVCC_FLAGS
                        + ["--compiler-options", "'-fPIC'"]
                        + cflags
                        + COMPUTE_CAPABILITY_ARGS
                    )
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]
                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith("-std=") for flag in cflags):
                    cflags.append("-std=c++11")

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def win_wrap_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):

            cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd, cflags):
                # Using regex to match src, obj and include files
                src_regex = re.compile("/T(p|c)(.*)")
                src_list = [m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m]

                obj_regex = re.compile("/Fo(.*)")
                obj_list = [m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m]

                include_regex = re.compile(r"((\-|\/)I.*)")
                include_list = [
                    m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        print("3 - src : " + src)
                        nvcc = _join_cuda_home("bin", "nvcc")
                        print("4 - " + nvcc)
                        if isinstance(cflags, dict):
                            cflags = cflags["nvcc"]
                        elif not isinstance(cflags, list):
                            cflags = []

                        cflags = COMMON_NVCC_FLAGS + cflags + COMPUTE_CAPABILITY_ARGS
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ["-Xcompiler", flag] + cflags
                        for macro in macros:
                            if len(macro) == 2:
                                if macro[1] == None:
                                    cflags += ["--define-macro", macro[0]]
                                else:
                                    cflags += ["--define-macro", "{}={}".format(macro[0], macro[1])]
                            elif len(macro) == 1:
                                cflags += ["--undefine-macro", macro[0]]

                        cmd = [nvcc, "-c", src, "-o", obj] + include_list + cflags
                    elif isinstance(cflags, dict):
                        cflags = COMMON_MSVC_FLAGS  # + self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(cflags, list):
                        cflags = COMMON_MSVC_FLAGS + cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = lambda cmd: spawn(cmd, cflags)
                return original_compile(
                    sources,
                    output_dir,
                    macros,
                    include_dirs,
                    debug,
                    extra_preargs,
                    extra_postargs,
                    depends,
                )
            finally:
                self.compiler.spawn = original_spawn

        # Monkey-patch the _compile method.
        if self.compiler.compiler_type == "msvc":
            self.compiler.compile = win_wrap_compile
        else:
            self.compiler._compile = unix_wrap_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu. On Python 2, there is no such ABI name.
        # The final extension, .so, would be .lib/.dll on Windows of course.
        ext_filename = build_ext.get_ext_filename(self, ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix and sys.version_info >= (3, 0):
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split(".")
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = ".".join(without_abi)
        return ext_filename


def include_headers(filename_list, sdist=False):
    """add hpp and h files to list if sdist is called"""
    if not sdist:
        return filename_list

    c_extensions = [".cu", ".c", ".C", ".cc", ".cpp", ".cxx", ".c++"]
    header_list = []
    for filename in filename_list:
        header = list(os.path.splitext(filename))
        if header[1] in c_extensions:
            header[1] = ".hpp"
            header_list.append("".join(header))

    filename_list += ["../CUDA/types_TIGRE.hpp", "../CUDA/errors.hpp"]
    return filename_list + header_list

Ax_ext = Extension(
    "_Ax",
    sources=include_headers(
        [
            "../CUDA/projection.cpp",
            "../CUDA/TIGRE_common.cpp",
            "../CUDA/Siddon_projection.cu",
            "../CUDA/Siddon_projection_parallel.cu",
            "../CUDA/ray_interpolated_projection.cu",
            "../CUDA/ray_interpolated_projection_parallel.cu",
            "../CUDA/GpuIds.cpp",
            "tigre/utilities/cuda_interface/_Ax.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    define_macros=define_macros,
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


Atb_ext = Extension(
    "_Atb",
    sources=include_headers(
        [
            "../CUDA/TIGRE_common.cpp",
            "../CUDA/voxel_backprojection.cu",
            "../CUDA/voxel_backprojection2.cu",
            "../CUDA/voxel_backprojection_parallel.cu",
            "../CUDA/GpuIds.cpp",
            "../CUDA/gpuUtils.cu",
            "tigre/utilities/cuda_interface/_Atb.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    define_macros=define_macros,
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


tvdenoising_ext = Extension(
    "_tvdenoising",
    sources=include_headers(
        [
            "../CUDA/TIGRE_common.cpp",
            "../CUDA/tvdenoising.cu",
            "../CUDA/GpuIds.cpp",
            "../CUDA/gpuUtils.cu",
            "tigre/utilities/cuda_interface/_tvdenoising.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    define_macros=define_macros,
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


minTV_ext = Extension(
    "_minTV",
    sources=include_headers(
        [
            "../CUDA/TIGRE_common.cpp",
            "../CUDA/POCS_TV.cu",
            "../CUDA/GpuIds.cpp",
            "../CUDA/gpuUtils.cu",
            "tigre/utilities/cuda_interface/_minTV.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    define_macros=define_macros,
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


AwminTV_ext = Extension(
    "_AwminTV",
    sources=include_headers(
        [
            "../CUDA/TIGRE_common.cpp",
            "../CUDA/POCS_TV2.cu",
            "../CUDA/GpuIds.cpp",
            "../CUDA/gpuUtils.cu",
            "tigre/utilities/cuda_interface/_AwminTV.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    define_macros=define_macros,
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


gpuUtils_ext = Extension(
    "_gpuUtils",
    sources=include_headers(
        [
            "../CUDA/gpuUtils.cu",
            "tigre/utilities/cuda_interface/_gpuUtils.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


RandomNumberGenerator_ext = Extension(
    "_RandomNumberGenerator",
    sources=include_headers(
        [
            "../CUDA/TIGRE_common.cpp",
            "../CUDA/RandomNumberGenerator.cu",
            "../CUDA/GpuIds.cpp",
            "../CUDA/gpuUtils.cu",
            "tigre/utilities/cuda_interface/_randomNumberGenerator.pyx",
        ],
        sdist=sys.argv[1] == "sdist",
    ),
    define_macros=define_macros,
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]] if not IS_WINDOWS else None,
    include_dirs=[NUMPY_INCLUDE, CUDA["include"], "../CUDA/"],
)


setup(
    name="pytigre",
    version="2.4.0",
    author="Ander Biguri, Reuben Lindroos, Sam Loescher",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[Ax_ext, Atb_ext, tvdenoising_ext, minTV_ext, AwminTV_ext, gpuUtils_ext, RandomNumberGenerator_ext],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["Cython", "matplotlib", "numpy", "scipy", "tqdm"],
    license_files=("LICENSE",),
    license="BSD 3-Clause",
    # since the package has c code, the egg cannot be zipped
    zip_safe=False,
)