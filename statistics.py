from math import sqrt


def average(data):
    """Return the average of a list of numeric values in data."""
    if len(data)==0:
        raise ValueError("List must contain at least one value")
    return sum(data)/len(data)


def variance(data):
    """the population variance of a list of numbers in data.

    The variance is the sum of squared differences between data values
    and their mean, divided by the number of items in the list.
    This is different from the Python library function statistics.variance
    which returns the sample variance, where the sum is divided by (n-1).

    Example: variance([1,5]) is ((1-3)**2 + (5-3)**2)/2 = 4.

    :param data: list of numbers for which variance will be computed. 
           Must contain at least one element.
    :returns: population variance of values in data list.
    :raises ValueError: if the data parameter is empty.

    >>> variance([1])
    0.0
    >>> variance([1, 1, 1, 1])
    0.0
    >>> variance([1, 2])
    0.25
    >>> variance([1000000, 1000004])
    4.0
    """
    # ugly code.
    n=len(data)
    if n==0:
        raise ValueError("List must contain at least one value")
    avg = average(data)
    return sum( [(x-avg)**2 for x in data] )/n


def stdev(data):
    """the standard deviation of a list of values"""
    return sqrt(variance(data))

