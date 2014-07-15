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

import os
current_dir = os.getcwd()
os.system('rm ' + current_dir + "/*.so")

import platform
if platform.system() == 'Linux':
	c11_args = ["-std=c++11"]
elif platform.system() == 'Darwin':
	c11_args = ["-std=c++11", "-stdlib=libc++"]

sourcefiles = ["k_seq_wrapper.pyx", "k_seq.cpp"]
main_I = "/usr/local/include"
main_L = ["-L/usr/local/lib"]

setup(
	cmdclass = {"build_ext" : build_ext},
	ext_modules = [Extension("k_sequitur_c",
			sourcefiles,
			include_dirs = [".", 
							np.get_include(),
							main_I,
							"../../../k_sequitur"],
			language = "c++",
			extra_compile_args= c11_args + ["-O3"],
            extra_link_args=(main_L )
            )]
)