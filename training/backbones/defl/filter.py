import numpy as np

def create_base_filters():
    """
    Create base filters to capture horizontal, vertical, and diagonal features.
    
    Returns:
    - A dictionary containing base filters in horizontal, vertical, and diagonal directions.
    """
    # Horizontal base filters
    horizontal_filter_1 = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    horizontal_filter_2 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    
    # Vertical base filters
    vertical_filter_1 = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    vertical_filter_2 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    
    # Diagonal base filters
    diagonal_filter_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    diagonal_filter_2 = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
    diagonal_filter_3 = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
    diagonal_filter_4 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    # Collect all filters in a dictionary for easy access
    base_filters = {
        "horizontal_1": horizontal_filter_1,
        "horizontal_2": horizontal_filter_2,
        "vertical_1": vertical_filter_1,
        "vertical_2": vertical_filter_2,
        "diagonal_1": diagonal_filter_1,
        "diagonal_2": diagonal_filter_2,
        "diagonal_3": diagonal_filter_3,
        "diagonal_4": diagonal_filter_4
    }
    
    return base_filters

# Generate base filters
base_filters = create_base_filters()

# Print out filters
# for name, filter_matrix in base_filters.items():
#     print(f"{name}:\n{filter_matrix}\n")

from scipy.signal import convolve2d
from itertools import combinations

def create_composite_filters(base_filters):
    """
    Create composite filters from base filters by combining up to 7 base filters using 2D convolution.
    
    Parameters:
    - base_filters: A dictionary containing base filters.
    
    Returns:
    - A list of composite filters.
    """
    # Convert base filters to a list for combinations
    base_filter_list = list(base_filters.values())
    
    # Store composite filters
    composite_filters = []
    
    # Generate combinations (up to 7 filters)
    for n in range(1, 8):  # n is the number of filters in the combination, from 1 to 7
        for combo in combinations(base_filter_list, n):
            # Initialize composite filter as the first base filter
            composite_filter = combo[0]
            # Convolve sequentially with each filter in the combination
            for f in combo[1:]:
                composite_filter = convolve2d(composite_filter, f, mode='same')
            # Add to the composite filter list
            composite_filters.append(composite_filter)
    
    return composite_filters

# Generate composite filters
composite_filters = create_composite_filters(base_filters)
print("composite_filters size:", len(composite_filters))