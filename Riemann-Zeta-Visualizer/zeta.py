import matplotlib.pyplot as plt

def generate_range(start, end):
    """
    GENERATE A LIST OF CONSECUTIVE INTEGERS FROM THE STARTING VALUE TO THE LAST ENDING VALUE
    ARGUMENTS/PARAMETERS :
        start(int) : Starting number of the range
        end(int) : Ending number of the range(included)
    RETURNS:
        list : List of integers from start to end(including end)
    """
    return list(range(start, end + 1))


def generate_primes(start, end):
    """
    GENERATE A LIST OF PRIME NUMBERS WITHIN A GIVEN RANGE
    ARGUMENTS/PARAMETERS :
        start(int) : Starting number of the range
        end(int) : Ending number of the range(included)
    RETURNS :
        list : List of Prime numbers in the range
    """
    prime = []
    for num in range(start, end + 1):
        if num < 2:
            continue
        isprime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                isprime = False
                break
        if isprime:
            prime.append(num)
    return prime


def count_primes(x_range_values, primes):
    count = 0
    prime_counts = []
    for value in x_range_values:
        if value in primes:
            count = count + 1
        prime_counts.append(count)
    return prime_counts


def main():
    print("This is part of Riemann-Zeta Function")
    print("This generates the graph of prime number counting step function")
    print("The function ζ(x) counts the number of primes less than or equal to x")

    start_range = int(input("Enter Starting Value : "))
    end_range = int(input("Enter End Value : "))

    if start_range > end_range:
        print("Error: Starting value must be less than or equal to ending value")
        exit()

    x_values = generate_range(start_range, end_range)
    primes = generate_primes(start_range, end_range)
    prime_count = count_primes(x_values, primes)

    plt.step(
        x_values,
        prime_count,
        where="post",
        linewidth=2,
        label="ζ(x): Prime Counting Function"
    )
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Number of primes ≤ X")
    plt.title("Prime Counting Step Function ζ(x)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
