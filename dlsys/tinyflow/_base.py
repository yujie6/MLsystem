# coding: utf-8
# pylint: disable=invalid-name
""" ctypes library of dlsys and helper functions """
from __future__ import absolute_import

import os
import ctypes
import numpy as np

float32 = np.float32

# lib = np.ctypeslib.load_library("./src/main.so", ".")
#
#
# ##################
# # Helper Methods #
# ##################
#
# def check_call(ret):
#     """Check the return value of C API call
#     This function will crash when error occurs.
#     Wrap every API call with this function
#     Parameters
#     ----------
#     ret : int
#         return value from API calls
#     """
#     assert (ret == 0)
#
#
# def c_array(ctype, values):
#     """Create ctypes array from a python array
#     Parameters
#     ----------
#     ctype : ctypes data type
#         data type of the array we want to convert to
#     values : tuple or list
#         data content
#     Returns
#     -------
#     out : ctypes array
#         Created ctypes array
#     """
#     return (ctype * len(values))(*values)
#
#
# ########################################################
#
# def cast_to_ndarray(arr):
#     if isinstance(arr, np.ndarray):
#         return arr
#     if not isinstance(arr, list):
#         arr = [arr]
#     return np.array(arr)