import math


def calculate_sqrt(x):
    return math.sqrt(x)


def calculate_interval(p, n):
    """Calculates the 95% confidence interval for a proportion.

    Args:
      p: The observed proportion in the sample.
      n: The sample size.

    Returns:
      A tuple containing the lower and upper bounds of the confidence interval.
    """

    z_alpha_by_2 = 1.96
    se = math.sqrt(p * (1 - p) / n)
    lower_bound = p - z_alpha_by_2 * se
    upper_bound = p + z_alpha_by_2 * se

    return lower_bound, upper_bound


def main():
    p = 0.7
    n = 12

    print("The square root of 0.7 * 0.3 / 12 is:",
          calculate_sqrt(0.7 * 0.3 / 12))

    lower_bound, upper_bound = calculate_interval(p, n)
    print("The 95% confidence interval for the proportion is:",
          lower_bound, "< p >", upper_bound)


if __name__ == "__main__":
    main()
