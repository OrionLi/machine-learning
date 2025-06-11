import numpy as np

# Create a list of 10 numbers
numbers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Original list: {numbers_list}")

# Convert the list to a NumPy array
numbers_array = np.array(numbers_list)
print(f"NumPy array: {numbers_array}")

# Slice the array to extract the first 5 elements
first_five = numbers_array[:5]
print(f"First 5 elements: {first_five}")

# Slice the array to extract the last 3 elements
last_three = numbers_array[-3:]
print(f"Last 3 elements: {last_three}")

# Calculate and print the mean, maximum, and minimum of the original array
array_mean = np.mean(numbers_array)
array_max = np.max(numbers_array)
array_min = np.min(numbers_array)

print(f"Mean of the array: {array_mean}")
print(f"Maximum of the array: {array_max}")
print(f"Minimum of the array: {array_min}")
