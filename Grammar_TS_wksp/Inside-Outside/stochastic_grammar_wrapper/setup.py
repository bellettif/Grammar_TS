'''
Created on 5 nov. 2013

compile with command line 
python setup.py build_ext --inplace

@author: francois belletti
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# For numpy array support
import numpy as np

sourcefiles = ["stochastic_grammar_wrapper.pyx",
			   "../Inside-Outside_string_lk/stochastic_rule.cpp"]
main_I = "/usr/local/include"
main_L = ["-L/usr/local/lib"]
#boost_include = "/usr/local/Cellar/boost/1.54.0/include"
#boost_l_flags = ["-lboost_system", "-lboost_filesystem", "-lboost_timer"]
#opencv_l_flags = ["-lopencv_highgui", "-lopencv_core", "-lopencv_imgproc",
#				  "-lopencv_objdetect", "-lopencv_calib3D", "-lsqlite3"]
c11_args = ["-std=c++11", "-stdlib=libc++"]

setup(
	cmdclass = {"build_ext" : build_ext},
	ext_modules = [Extension("SCFG_c",
			sourcefiles,
			include_dirs = [".",
							np.get_include(),
							main_I,
							"../Inside-Outside_string_lk"],
			language = "c++",
			extra_compile_args= c11_args + ["-O3"],
            extra_link_args=(main_L ) #+ opencv_l_flags + boost_l_flags)
            )]
)