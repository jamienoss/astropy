# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .convolution_test_classes import ConvolveFunc, OneDTests, TwoDTests, ThreeDTests, MiscellaneousTests

import itertools

def pytest_generate_tests(metafunc): # This should go somewhere more generic
    argnames, argvalues = metafunc.cls.parameterize(metafunc)
    #print(argnames)
    #print(argvalues)
    #print()
    metafunc.parametrize(argnames, argvalues)

VALID_DTYPES = ['>f4', '<f4', '>f8', '<f8']

#VALID_DTYPES = []
#for dtype_array in ['>f4', '<f4', '>f8', '<f8']:
#    for dtype_kernel in ['>f4', '<f4', '>f8', '<f8']:
#        VALID_DTYPES.append((dtype_array, dtype_kernel))

class Convolve(ConvolveFunc):
    parameter_space = {
        "dtype_array" : VALID_DTYPES,
        "dtype_kernel" : VALID_DTYPES,
        "boundary" : [None, 'fill', 'wrap', 'extend'],
        "nan_treatment" : ['interpolate', 'fill'],
        "normalize_kernel" : [True, False],
        "preserve_nan" : [True, False],
        "ndims" : [1, 2, 3]
        }
    
    def convolveFunc(self, *args, **kargs):
        return self.convolve(*args, **kargs)

class TestConvolve1D(OneDTests, Convolve):
    pass

class TestConvolve2D(TwoDTests, Convolve):
    pass

class TestConvolve3D(ThreeDTests, Convolve):
    pass

class TestMiscellaneous(MiscellaneousTests, Convolve):
    pass
