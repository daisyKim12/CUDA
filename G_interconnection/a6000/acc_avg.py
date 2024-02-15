from collections import defaultdict

# Function to read the file and accumulate values for each index
def accumulate_values(file_path):
    index_values = defaultdict(list)

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            index, value = map(float, line.strip().split(','))
            index_values[int(index)].append(value)

    # Calculate the sum of values for each index
    accumulated_values = {}
    for index, values in index_values.items():
        accumulated_values[index] = sum(values)

    return accumulated_values

# Example usage
file_path = 'avg.txt'
accumulated_values = accumulate_values(file_path)

# Output the accumulated values
for index, value in sorted(accumulated_values.items()):
    # print(f'Index {index}: {value}')
    print(f'{index}, {value}')