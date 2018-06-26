# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .convolution_test_classes import ConvolveFunc, OneDTests, TwoDTests, ThreeDTests, MiscellaneousTests


class Convolve(ConvolveFunc):
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
