import math
import unittest

from utils import fixedRateToTick, fixedRateToSqrtPrice, getSqrtRatioAtTick, sqrtPriceToFixedRate, getAmount0Delta, getAmount1Delta, notional_to_liquidity

class TestUtils(unittest.TestCase):

    def test_fixed_rate_to_tick(self):
        self.assertAlmostEqual(fixedRateToTick(1), 0, delta=1e-6)

        self.assertAlmostEqual(fixedRateToTick(2), -6931.81, delta=1e-2)
        self.assertAlmostEqual(fixedRateToTick(3), -10986.67, delta=1e-2)
        self.assertAlmostEqual(fixedRateToTick(1000), -69081, delta=1e-2)

        self.assertAlmostEqual(fixedRateToTick(1/2), 6931.81, delta=1e-2)
        self.assertAlmostEqual(fixedRateToTick(1/3), 10986.67, delta=1e-2)
        self.assertAlmostEqual(fixedRateToTick(1/1000), 69081, delta=1e-2)

    
    def test_fixed_rate_to_sqrt_price(self):
        self.assertAlmostEqual(fixedRateToSqrtPrice(1), 1, delta=1e-6)

        self.assertAlmostEqual(fixedRateToSqrtPrice(2), math.sqrt(1/2), delta=1e-6)
        self.assertAlmostEqual(fixedRateToSqrtPrice(3), math.sqrt(1/3), delta=1e-6)
        self.assertAlmostEqual(fixedRateToSqrtPrice(1000), math.sqrt(1/1000), delta=1e-6)

        self.assertAlmostEqual(fixedRateToSqrtPrice(1/2), math.sqrt(2), delta=1e-6)
        self.assertAlmostEqual(fixedRateToSqrtPrice(1/3), math.sqrt(3), delta=1e-6)
        self.assertAlmostEqual(fixedRateToSqrtPrice(1/1000), math.sqrt(1000), delta=1e-6)


    def test_get_sqrt_ratio_at_tick(self):
        self.assertAlmostEqual(getSqrtRatioAtTick(0), 1, delta=1e-6)

        self.assertAlmostEqual(getSqrtRatioAtTick(-6931.81), math.sqrt(1/2), delta=1e-2)
        self.assertAlmostEqual(getSqrtRatioAtTick(-10986.67), math.sqrt(1/3), delta=1e-2)
        self.assertAlmostEqual(getSqrtRatioAtTick(-69081), math.sqrt(1/1000), delta=1e-2)

        self.assertAlmostEqual(getSqrtRatioAtTick(6931.81), math.sqrt(2), delta=1e-2)
        self.assertAlmostEqual(getSqrtRatioAtTick(10986.67), math.sqrt(3), delta=1e-2)
        self.assertAlmostEqual(getSqrtRatioAtTick(69081), math.sqrt(1000), delta=1e-2)


    def test_sqrt_price_to_fixed_rate(self):
        fixed_rates = [1, 2, 3, 1000, 1/2, 1/3, 1/1000]

        for fr in fixed_rates:
            self.assertAlmostEqual(sqrtPriceToFixedRate(fixedRateToSqrtPrice(fr)), fr)


    def test_get_amount_0_delta(self):
        amount0 = getAmount0Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(0.5), 0)
        self.assertAlmostEqual(amount0, 0)

        amount0 = getAmount0Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(1), 1000)
        self.assertAlmostEqual(amount0, 0)

        amount0 = getAmount0Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(1 / 1.21), 1)
        self.assertAlmostEqual(amount0, 0.090909090909090910, delta=1e-6)

        amount0 = getAmount0Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(1.21), 1)
        self.assertAlmostEqual(amount0, 0.1, delta=1e-6)

        amount0 = getAmount0Delta(fixedRateToSqrtPrice(4), fixedRateToSqrtPrice(1.21), 1)
        self.assertAlmostEqual(amount0, 0.9, delta=1e-6)

        amount0 = getAmount0Delta(fixedRateToSqrtPrice(1.21), fixedRateToSqrtPrice(4), 1)
        self.assertAlmostEqual(amount0, 0.9, delta=1e-6)


    def test_get_amount_1_delta(self):
        amount1 = getAmount1Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(0.5), 0)
        self.assertAlmostEqual(amount1, 0)

        amount1 = getAmount1Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(1), 1000)
        self.assertAlmostEqual(amount1, 0)

        amount1 = getAmount1Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(1.21), 1)
        self.assertAlmostEqual(amount1, 0.090909090909090910, delta=1e-6)

        amount1 = getAmount1Delta(fixedRateToSqrtPrice(1), fixedRateToSqrtPrice(1 / 1.21), 1)
        self.assertAlmostEqual(amount1, 0.1, delta=1e-6)

        amount1 = getAmount1Delta(fixedRateToSqrtPrice(1 / 4), fixedRateToSqrtPrice(1 / 1.21), 1)
        self.assertAlmostEqual(amount1, 0.9, delta=1e-6)

        amount1 = getAmount1Delta(fixedRateToSqrtPrice(1 / 1.21), fixedRateToSqrtPrice(1 / 4), 1)
        self.assertAlmostEqual(amount1, 0.9, delta=1e-6)

    
    def test_notional_to_liquidity(self):
        inputs = [(1, 2, 1000), (1.1, 2, 100), (1/2, 1, 10), (1/2, 1/1.1, 1)]


        for low_fix, high_fix, notional in inputs:
            liquidity = notional_to_liquidity(notional, fixedRateToTick(high_fix), fixedRateToTick(low_fix))
            amount1 = getAmount1Delta(fixedRateToSqrtPrice(high_fix), fixedRateToSqrtPrice(low_fix), liquidity)
            self.assertAlmostEqual(notional, amount1)
            

if __name__ == '__main__':
    unittest.main()






