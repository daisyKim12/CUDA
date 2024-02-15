import sys

# Function to read the file and accumulate values
def accumulate_values(file_path):
    accumulated_value = 0.0

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            value = float(line.strip())
            accumulated_value += value

    return accumulated_value

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    total_accumulated_value = accumulate_values(file_path)

    # Output the accumulated value
    print(f'Total accumulated value: {total_accumulated_value}')

