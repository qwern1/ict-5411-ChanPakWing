from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import random


def computation_on_numpy_arrays() -> None:
	"""Complete the five computation exercises."""
	print(" Computation on NumPy Arrays")

	# Part 1: create and print a 2D array.
	two_dimensional_array = np.array([[1, 2, 3], [4, 5, 6]])
	print("Part 1 2D array:")
	print(two_dimensional_array)

	first_array = np.array([10, 20, 30])
	second_array = np.array([1, 2, 3])
	print("\nPart 2 - addition:")
	print(first_array + second_array)
	print("Part 2 - subtraction:")
	print(first_array - second_array)

	numeric_array = np.array([1, 2, 3, 4, 5])
	print("\nPart 3 - product:")
	print(np.prod(numeric_array))
	print("Part 3 - differences:")
	print(np.diff(numeric_array))

	function_array = np.array([1, 2, 4, 8], dtype=float)
	print("\nPart 4 - natural logarithm:")
	print(np.log(function_array))
	print("Part 4 - base-2 logarithm:")
	print(np.log2(function_array))
	print("Part 4 - sine:")
	print(np.sin(function_array))

	print("\nPart 5 - random number from 0 to 1000:")
	print(random.randint(0, 1001))
	shuffled_array = np.array([5, 2, 1, 3, 0])
	random.shuffle(shuffled_array)
	print("First shuffle:")
	print(shuffled_array)
	random.shuffle(shuffled_array)
	print("Second shuffle:")
	print(shuffled_array)

def indexing_and_sorting() -> None:
	"""This short are two and it is about indexing."""
	array_to_sort = np.array([5, 2, 1, 3, 0])
	sorted_array = np.sort(array_to_sort)
	print("\n Indexing and Sorting NumPy Arrays")
	print("Sorted array:")
	print(sorted_array)
	print("Last two elements:")
	print(sorted_array[-2:])


def probability_distribution(output_path: Path) -> None:
	"""Plot a binomial distribution with probability p = 0.3."""
	probability = 0.3
	trials = 10
	outcomes = random.binomial(n=trials, p=probability, size=1000)

	counts = np.bincount(outcomes, minlength=trials + 1)
	probabilities = counts / outcomes.size
	values = np.arange(trials + 1)

	figure, axis = plt.subplots(figsize=(8, 4.5))
	axis.bar(values, probabilities, color="steelblue", edgecolor="black")
	axis.set_title("Binomial Distribution (p = 0.3)")
	axis.set_xlabel("Number of successes")
	axis.set_ylabel("Observed probability")
	axis.set_xticks(values)
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)

	print("\nProbability Distribution in NumPy")
	print(f"Binomial distribution plot saved to: {output_path}")


def structured_data() -> None:
	"""Demonstrate structured arrays with named fields."""
	print("\n Structured Data using NumPy")

	data = [
		("Alice", "F", 26),
		("Bob", "M", 19),
		("Cathy", "F", 21),
		("Doug", "M", 24),
	]
	data_type = [
		("name", "U10"),
		("gender", "U1"),
		("age", "i4"),
	]
	people = np.array(data, dtype=data_type)

	print("Structured array:")
	print(people)
	print("Names:")
	print(people["name"])
	print("Ages:")
	print(people["age"])
	print("People aged 21 or older:")
	print(people[people["age"] >= 21])
	print("Sorted by age:")
	print(np.sort(people, order="age"))


def main() -> None:
	"""Run all workshop exercises."""
	computation_on_numpy_arrays()
	indexing_and_sorting()
	probability_distribution(Path(__file__).with_name("binomial_distribution.png"))
	structured_data()


if __name__ == "__main__":
	main()
