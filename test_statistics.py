from unittest import TestCase
from statistics import variance, stdev, average
from math import sqrt

class StatisticsTest(TestCase):

    def test_average(self):
        """Test average function with various cases."""
        self.assertEqual(average([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(average([10.0, 20.0, 30.0]), 20.0)
        self.assertEqual(average([1e10, 1e10]), 1e10)
        self.assertEqual(average([1e-10, 1e-10]), 1e-10)
        self.assertAlmostEqual(average([1e10, 1e-10]), 5e9)
        self.assertEqual(average([5]), 5.0)
        self.assertEqual(average([-1, -2, -3]), -2.0)
        self.assertAlmostEqual(average([0.5, 1.5, 2.5]), 1.5)
        with self.assertRaises(ValueError):
            average([])

    def test_variance_typical_values(self):
        """Test variance with typical values."""
        self.assertEqual(0.0, variance([10.0, 10.0, 10.0, 10.0, 10.0]))
        self.assertEqual(2.0, variance([1, 2, 3, 4, 5]))
        self.assertEqual(8.0, variance([10, 2, 8, 4, 6]))

    def test_variance_non_integers(self):
        """Test variance with decimal values."""
        self.assertAlmostEqual(4.0, variance([0.1, 4.1]))
        self.assertAlmostEqual(8.0, variance([0.1, 4.1, 4.1, 8.1]))

        # Corrected test cases
        self.assertAlmostEqual(2.5e19, variance([1e10, 2e10]))
        self.assertAlmostEqual(1e-20, variance([1e-10, 1e-10]))
        self.assertAlmostEqual(2.5e19, variance([1e10, 1e-10]))

    def test_stdev(self):
        """Test standard deviation calculations."""
        self.assertEqual(0.0, stdev([10.0]))
        self.assertEqual(2.0, stdev([1, 5]))
        self.assertAlmostEqual(sqrt(0.5), stdev([0, 0.5, 1, 1.5, 2]))

        # Corrected test cases
        self.assertAlmostEqual(sqrt(2.5e19), stdev([1e10, 2e10]))
        self.assertAlmostEqual(sqrt(1e-20), stdev([1e-10, 1e-10]))
        self.assertAlmostEqual(sqrt(2.5e19), stdev([1e10, 1e-10]))

    def test_variance_empty_list(self):
        """Ensure variance handles empty lists correctly."""
        with self.assertRaises(ValueError):
            variance([])

    def test_edge_case_zero_variance(self):
        """Ensure variance of identical numbers is zero."""
        self.assertEqual(0.0, variance([1, 1, 1, 1, 1]))
        self.assertEqual(0.0, variance([3.14, 3.14, 3.14, 3.14]))

    def test_edge_case_single_element_stdev(self):
        """Ensure standard deviation of a single value is zero."""
        self.assertEqual(0.0, stdev([0]))
        self.assertEqual(0.0, stdev([1.0]))

    def test_large_range_variance(self):
        """Test variance with a large range of numbers."""
        self.assertAlmostEqual(2.499999995e+17, variance([1, 1000000000]))

    def test_negative_numbers_variance(self):
        """Ensure variance handles negative numbers correctly."""
        self.assertAlmostEqual(0.6666666666666666, variance([-1, -2, -3]))

    def test_large_range_stdev(self):
        """Test standard deviation with a large range of numbers."""
        self.assertAlmostEqual(sqrt(2.499999995e+17), stdev([1, 1000000000]))

    def test_zero_variance_single_number(self):
        """Ensure variance for a single number is zero."""
        self.assertEqual(0.0, variance([0]))
        self.assertEqual(0.0, variance([1.0]))

    def test_large_data_set(self):
        """Test variance and stdev with a very large data set."""
        data = list(range(1, 10001))
        calculated_variance = variance(data)
        expected_variance = (1/10000) * sum((x - average(data))**2 for x in data)
        self.assertAlmostEqual(calculated_variance, expected_variance)
        self.assertAlmostEqual(stdev(data), sqrt(expected_variance))

    def test_edge_case_small_data_set(self):
        """Test variance and stdev with the smallest non-trivial data sets."""
        self.assertAlmostEqual(variance([1, 2]), 0.25)
        self.assertAlmostEqual(stdev([1, 2]), sqrt(0.25))

    def test_large_range_combined_variance_stdev(self):
        """Test combined large range for variance and stdev."""
        data = list(range(-1000000000, 1000000000, int(1e6)))
        avg = average(data)
        calculated_variance = variance(data)
        expected_variance = sum((x - avg) ** 2 for x in data) / len(data)
        self.assertAlmostEqual(calculated_variance, expected_variance)
        self.assertAlmostEqual(stdev(data), sqrt(expected_variance))

