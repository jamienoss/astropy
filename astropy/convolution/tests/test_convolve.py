# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .convolution_test_classes import ConvolveFunc, OneDTests, TwoDTests, ThreeDTests, MiscellaneousTests

def pytest_generate_tests(metafunc): # This should go somewhere more generic
    argnames, argvalues = metafunc.cls.parameterize(metafunc)
    metafunc.parametrize(argnames, argvalues)


class Convolve(ConvolveFunc):
    parameter_space = {
        "dtype_array" : ('>f4', '<f4', '>f8', '<f8'),
        "dtype_kernel" : ('>f4', '<f4', '>f8', '<f8'),
        "boundary" : (None, 'fill', 'wrap', 'extend'),
        "nan_treatment" : ('interpolate', 'fill'),
        "normalize_kernel" : (True, False),
        "preserve_nan" : (True, False),
        "ndims" : (1, 2, 3)
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
