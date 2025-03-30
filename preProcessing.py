#### ACCUMULATED PRE-PROCESSING STUFF

import pandas as pd
import re
from word2number import w2n

from sys import set_coroutine_origin_tracking_depth

from thefuzz import fuzz, process
import jellyfish
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

def main():
    ## Testing pandas
    # Create a simple DataFrame
    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
    print(df)


    ## Testing re
    # Test regex pattern to find numbers in a string
    text = "My phone number is 123-456-7890"
    match = re.findall(r'\d+', text)
    print(match)  # Expected: ['123', '456', '7890']


    ## Testing w2n
    # Convert number words to digits
    print(w2n.word_to_num("twenty five"))  # Expected: 25
    print(w2n.word_to_num("three hundred and forty two"))  # Expected: 342


    ## Testing set_coroutine_origin_tracking_depth
    # Test setting coroutine tracking depth
    set_coroutine_origin_tracking_depth(2)
    print("Coroutine tracking depth set successfully.")


    ## Testing fuzz, process (from thefuzz)
    # Compare two similar strings
    print(fuzz.ratio("hello world", "hello world!"))  # Expected: Close to 95-100

    # Find the closest match from a list
    choices = ["apple", "banana", "grape"]
    print(process.extractOne("appl", choices))  # Expected: ('apple', high score)


    ## Testing jellyfish
    # Compute Levenshtein distance
    print(jellyfish.levenshtein_distance("kitten", "sitting"))  # Expected: 3

    # Compute Soundex encoding
    print(jellyfish.soundex("Robert"))  # Expected: 'R163'
    print(jellyfish.soundex("Rupert"))  # Expected: 'R163' (same as Robert)

    ## Testing connected_components
    # Example adjacency matrix
    graph = csr_matrix([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    n_components, labels = connected_components(graph)
    print(f"Number of components: {n_components}")  # Expected: 1
    print(f"Component labels: {labels}")  # Expected: [0 0 0] (all connected)


    ## Testing squareform
    # Example distance matrix (condensed form)
    condensed = [1, 2, 3, 4, 5, 6]
    square = squareform(condensed)

    print("Squareform Distance Matrix:")
    print(square)


    ## Testing plt
    # Create a simple line plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y, label="Sine Wave")
    plt.legend()
    plt.show()