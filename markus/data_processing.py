import pandas as pd
import re
import numpy as np
import statistics
from ast import literal_eval


############## we got helper functions at home ################

def text2int(textnum, numwords={}):
    """

    :param textnum:
    :param numwords:
    :return:
    """
    textnum = textnum.lower()
    textnum = re.sub(r'-', ' ', textnum)
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current


def connected_components(csgraph, directed):
    """
    Finds connected components in a graph using Depth-First Search (DFS).

    Args:
        adj_matrix: 2D NumPy array or Pandas DataFrame representing adjacency matrix

    Returns:
        n_components: int (number of connected components)
        labels: 1D array of component labels
    """
    adj_matrix = csgraph

    if isinstance(adj_matrix, pd.DataFrame):
        adj_matrix = adj_matrix.values

    n = len(adj_matrix)
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=int)
    current_label = 0

    def dfs(node):
        stack = [node]
        while stack:
            v = stack.pop()
            if not visited[v]:
                visited[v] = True
                labels[v] = current_label
                # Add unvisited neighbors to stack
                neighbors = np.where(adj_matrix[v] > 0)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)

    for i in range(n):
        if not visited[i]:
            dfs(i)
            current_label += 1

    return current_label, labels

def clean_text(text):
    """Removes non-alphanumeric characters and normalizes spaces."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize spaces (replace multiple spaces with a single space)
    return cleaned

def jaro_similarity(s1, s2):
    """Computes the Jaro similarity between two strings."""
    len_s1, len_s2 = len(s1), len(s2)

    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    match_distance = (max(len_s1, len_s2) // 2) - 1
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    matches = 0
    transpositions = 0

    # Count matches
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)

        for j in range(start, end):
            if s2_matches[j]:  # Already matched
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    s1_mapped = [s1[i] for i in range(len_s1) if s1_matches[i]]
    s2_mapped = [s2[i] for i in range(len_s2) if s2_matches[i]]

    for i in range(len(s1_mapped)):
        if s1_mapped[i] != s2_mapped[i]:
            transpositions += 1

    transpositions //= 2

    return (matches / len_s1 + matches / len_s2 + (matches - transpositions) / matches) / 3


def jaro_winkler_similarity(s1, s2, p=0.1):
    """Computes Jaro-Winkler similarity between two strings."""
    jaro_sim = jaro_similarity(s1, s2)

    # Find the common prefix length (up to 4 characters)
    prefix_length = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break

    return jaro_sim + (prefix_length * p * (1 - jaro_sim))


def compute_similarity_matrix(responses, similarity_func, threshold):
    """Computes a similarity matrix based on the given similarity function."""
    # preparing to use jaro winkler similarity test
    n = len(responses)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_func(responses[i], responses[j])
            if similarity >= threshold:
                similarity_matrix[i, j] = 1
                similarity_matrix[j, i] = 1

    return similarity_matrix

def simple_fuzzy_match(s1, s2):
    """A simple fuzzy matching function similar to `fuzz.partial_ratio`."""
    s1, s2 = clean_text(s1).lower(), clean_text(s2).lower()
    if s1 in s2 or s2 in s1:
        return 100  # Exact substring match

    common_chars = sum((c in s2) for c in s1)
    return int((2 * common_chars / (len(s1) + len(s2))) * 100)

def extract_best_match(query, choices, threshold=85):
    """Finds the best match for a query string within a list of choices."""
    best_match, best_score = None, 0
    for choice in choices:
        score = simple_fuzzy_match(query, choice)
        if score > best_score:
            best_match, best_score = choice, score
    return (best_match, best_score) if best_score >= threshold else (None, 0)


###########################

######## Q2 parsing ##########
# Define a dictionary to map month abbreviations to numbers
month_map = {
    'Jan': '1',
    'Feb': '2',
    'Mar': '3',
    'Apr': '4',
    'May': '5',
    'Jun': '6',
    'Jul': '7',
    'Aug': '8',
    'Sep': '9',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

# Function to convert date format
def convert_date_format(date_str):
    if isinstance(date_str, str) and '-' in date_str:
        try:
            day, month_abbr = date_str.split('-')
            month_num = month_map.get(month_abbr, month_abbr)
            return f"{int(month_num)}-{int(day)}"
        except ValueError:
           return date_str
    return date_str



# Function to take the average if there's a range(either with '-' or 'to')
def average_of_range(text):
  if isinstance(text, str):
    range_match = re.search(r'(\d+)[\s-]+(\d+)', text)
    range_match2 = re.search(r'(\d+)\s* to \s*(\d+)', text)

    if range_match is not None:
      num1, num2 = int(range_match.group(1)), int(range_match.group(2))
      return int((num1 + num2) / 2 )# Take the average of the range
    elif range_match2 is not None:
      num1, num2 = int(range_match2.group(1)), int(range_match2.group(2))
      return int((num1 + num2) / 2)
    else:
      return text


# Function to extract numerical values in data entry
def extract_number(text):
  # check if entry is string
  if isinstance(text, str):
    # search for any digits in the text
    # note, this is called after all previous functions, which should indicate high likelihood
    # that what's left is the answer value with only v/ small margin of error
    num = re.search(r'(\d+)', text)

    # if found, return as int
    if num is not None:
      return int(num.group(1))

    # if no match, return original entry
    else:
      return text
  # if not, return original entry
  else:
    return text


def convert_wrds(text):
    if isinstance(text, str):
        # Regex to find hyphenated numbers (e.g., "twenty-two", "one-hundred-three")
        hyphenated_pattern = r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)(?:-\s?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion))+\b'

        # Regex to find single (non-hyphenated) number words
        single_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b'

        # Look for a hyphenated number first
        hyphenated_match = re.search(hyphenated_pattern, text, re.IGNORECASE)
        if hyphenated_match:
            try:
                return text2int(hyphenated_match.group(0))  # Convert full hyphenated number
            except ValueError:
                return hyphenated_match.group(0)  # Return as text if conversion fails

        # If no hyphenated number, look for a single non-hyphenated number
        single_match = re.search(single_pattern, text, re.IGNORECASE)
        if single_match:
            try:
                return text2int(single_match.group(1))  # Convert first non-hyphenated number
            except ValueError:
                return single_match.group(1)  # Return as text if conversion fails

    return text  # Return original text if no numbers found




# Function to count the number of ingredients if responses are comma-separated or use '*'
def count_ingredients(text):
    if isinstance(text, str):  # Ensure it's a string
        if ',' in text:  # Count comma-separated items
            return len([item.strip() for item in text.split(',') if item.strip()])
        elif '* ' in text:  # Count bullet-pointed items
            return len([item.strip() for item in text.split('* ') if item.strip()])
    return text  # If it's a single ingredient or doesn't match conditions, return 1




# Function to identify likely 'outliers' (ie. stories/anecdotes, 'i don't knows', etc.)
def outlier_rest(text):
  # check if entry is string
  if isinstance(text, str):
    # search for any text characters
    is_match = re.search(r'(\s+)', text)
    # if found, return 'None' as a string to identify an outlier in the data
    if is_match is not None:
      return 'None'

    # if not, return original entry
    else:
      return text
  # if not, return original
  else:
    return text

def replace_non_numeric(value):
    try:
        float(value)  # Attempt to convert to a number
        return value  # If successful, keep the value
    except (ValueError, TypeError):  # Catch both value and type errors
        return 'none'  # If conversion fails, replace with 'none'

def q2_processing(df):

    df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(convert_date_format)
    df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(average_of_range)
    df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(extract_number)

    df['Q2: How many ingredients would you expect this food item to contain?'] = df[
        'Q2: How many ingredients would you expect this food item to contain?'
    ].apply(convert_wrds)

    df['Q2: How many ingredients would you expect this food item to contain?'] = df[
        'Q2: How many ingredients would you expect this food item to contain?'
    ].apply(count_ingredients)

    df['Q2: How many ingredients would you expect this food item to contain?'] = df[
        'Q2: How many ingredients would you expect this food item to contain?'
    ].apply(outlier_rest)



    q2='Q2: How many ingredients would you expect this food item to contain?'
    # Apply the function only to 'Column1'
    df[q2] = df[q2].apply(replace_non_numeric)

    return df



######## Q4 ########


# Function to take the average if there's a range(either with '-' or 'to')
def q4_range(text):
  # make sure data entry is a string
  if isinstance(text, str):

    # case where there's a range and no dollar sign (eg. 2-5)
    no_sign = re.search(r'(\d+)\s*(?:-|to|up to)\s*(\d+)', text)

    # case where there's a range w/ a dollar sign at the front (eg. 5)
    sign_front = re.search(r'(\d+)\s*(?:-|to|up to)\s*$(\d+)', text)

    # case where there's a range w/ a dollar sign at the back (eg. 2)
    sign_back = re.search(r'(\d+)$\s*(?:-|to|up to)\s*(\d+)', text)

    # if one is not None, ie a match was found, extract #s and take average
    if no_sign is not None:
      num1, num2 = int(no_sign.group(1)), int(no_sign.group(2))
      return (num1 + num2) / 2

    elif sign_front is not None:
      num1, num2 = int(sign_front.group(1)), int(sign_front.group(2))
      return (num1 + num2) / 2

    elif sign_back is not None:
      num1, num2 = int(sign_back.group(1)), int(sign_back.group(2))
      return (num1 + num2) / 2

    # if no match found, return original data entry
    else:
      return text
  # if not string, return original entry
  else:
    return text

# Function to find float money values and return as floats
def floats(text):
  # check if data entry is a string
  if isinstance(text, str):
    # find values in full 'monetary' form(ie. 12.00 instead of 12, or 12.99)
    float_vars = re.search(r'(\d+\.\d{2})', text)

    # if not None, ie match found, return the value in float format
    if float_vars is not None:
      return float(float_vars.group(1))

    # if no match found, return original entry
    else:
      return text
  # if not string, return original entry
  else:
    return text

# Function to find values with written 'money' indications (ie. 12 dollars)
def dollars(text):
  # check if entry is a string
  if isinstance(text, str):
    # search for variations of 'money-indicative' words that have a number directly preceeding it
    dollar_vars = re.search(r'(\d+)\s*(dollars|dollar|dollor|bucks|buck)', text, re.IGNORECASE)
    currencies = re.search(r'(\d+)\s*(CAD|USD|Canadian)', text, re.IGNORECASE)

    # if match for either found, return float value of that number
    if dollar_vars is not None:
      return float(dollar_vars.group(1))
    elif currencies is not None:
      return float(currencies.group(1))

    # if no match found, return original
    else:
      return text
  # if not, return original entry
  else:
    return text

# Function to find values directly preceding or following a '$' symbol
def dollar_sign(text):
  # make sure entry is a string
  if isinstance(text, str):

    # search for numbers either before or after '$'
    # note, this is called after q4_range to not misselect 5 as just 4
    sign_first = re.search(r'$(\d+)', text)
    sign_after = re.search(r'(\d+)\s*$', text)

    # if found, return value as float
    if sign_first is not None:
      return float(sign_first.group(1))
    elif sign_after is not None:
      return float(sign_after.group(1))

    # if no match found, return original entry
    else:
      return text
  # if not, return original entry
  else:
    return text

# Function to extract numerical values in data entry
def extract_number(text):
  # check if entry is string
  if isinstance(text, str):
    # search for any digits in the text
    # note, this is called after all previous functions, which should indicate high likelihood
    # that what's left is the answer value with only v/ small margin of error
    num = re.search(r'(\d+)', text)

    # if found, return as float
    if num is not None:
      return float(num.group(1))

    # if no match, return original entry
    else:
      return text
  # if not, return original entry
  else:
    return text

# Function to find written numbercal values and convert into proper form

def convert_wrds(text):
    if isinstance(text, str):
        # Regex to find hyphenated numbers (e.g., "twenty-two", "one-hundred-three")
        hyphenated_pattern = r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)(?:-\s?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion))+\b'

        # Regex to find single (non-hyphenated) number words
        single_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b'

        # Look for a hyphenated number first
        hyphenated_match = re.search(hyphenated_pattern, text, re.IGNORECASE)
        if hyphenated_match:
            try:
                return text2int(hyphenated_match.group(0))  # Convert full hyphenated number
            except ValueError:
                return hyphenated_match.group(0)  # Return as text if conversion fails

        # If no hyphenated number, look for a single non-hyphenated number
        single_match = re.search(single_pattern, text, re.IGNORECASE)
        if single_match:
            try:
                return text2int(single_match.group(1))  # Convert first non-hyphenated number
            except ValueError:
                return single_match.group(1)  # Return as text if conversion fails

    return text  # Return original text if no numbers found


# Function to identify likely 'outliers' (ie. stories/anecdotes, 'i don't knows', etc.)
def outlier_rest(text):
  # check if entry is string
  if isinstance(text, str):
    # search for any text characters
    is_match = re.search(r'(\s+)', text)
    # if found, return 'None' as a string to identify an outlier in the data
    if is_match is not None:
      return 'None'

    # if not, return original entry
    else:
      return text
  # if not, return original
  else:
    return text


def q4_processing(df):
    # Apply function to the Q2 column
    df['Q2: How many ingredients would you expect this food item to contain?'] = df[
        'Q2: How many ingredients would you expect this food item to contain?'
    ].apply(convert_wrds)

    # Applying all functions to the Q4 column
    # Important note! ORDER MATTERS! Specific order to identify cases, and not *mis*-identify cases with only very small margin of error

    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].fillna('none')


    # Apply the date function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(convert_date_format)

    # Apply the q4_range function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(q4_range)

    # Apply the floats function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(floats)

    # Apply the dollars function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(dollars)

    # Apply the dollar_sign function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(dollar_sign)

    # Apply the extract number function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(extract_number)

    # Apply the convert words function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(convert_wrds)

    # Apply the outlier_rest function to the Q4 column
    df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(outlier_rest)
    return df





######### Q6 ###########

#### define a function to isolate a set of entries that are greater than 2
def divide_into_short_long(df, col):
  df = df.copy()
  df['num_words'] = df[col].str.split().str.len()  # counting number words in each entry
  df['original_index'] = df.index  # Store the original index

  small_entries_df = df[df['num_words'] <= 2].drop(columns=['num_words'])
  large_entries_df = df[df['num_words'] > 2].drop(columns=['num_words'])
  return small_entries_df, large_entries_df


def q6_processing(df):

    #### define a function to isolate a set of entries that are greater than 2
    def divide_into_short_long(df, col):
        df = df.copy()
        df['num_words'] = df[col].str.split().str.len()  # counting number words in each entry
        df['original_index'] = df.index  # Store the original index

        small_entries_df = df[df['num_words'] <= 2].drop(columns=['num_words'])
        large_entries_df = df[df['num_words'] > 2].drop(columns=['num_words'])
        return small_entries_df, large_entries_df

    def connected_components(csgraph, directed):
        """
        Finds connected components in a graph using Depth-First Search (DFS).

        Args:
            adj_matrix: 2D NumPy array or Pandas DataFrame representing adjacency matrix

        Returns:
            n_components: int (number of connected components)
            labels: 1D array of component labels
        """
        adj_matrix = csgraph

        if isinstance(adj_matrix, pd.DataFrame):
            adj_matrix = adj_matrix.values

        n = len(adj_matrix)
        visited = np.zeros(n, dtype=bool)
        labels = np.zeros(n, dtype=int)
        current_label = 0

        def dfs(node):
            stack = [node]
            while stack:
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    labels[v] = current_label
                    # Add unvisited neighbors to stack
                    neighbors = np.where(adj_matrix[v] > 0)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            stack.append(neighbor)

        for i in range(n):
            if not visited[i]:
                dfs(i)
                current_label += 1

        return current_label, labels



    def clean_text(text):
        """Removes non-alphanumeric characters and normalizes spaces."""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize spaces (replace multiple spaces with a single space)
        return cleaned

    def jaro_similarity(s1, s2):
        """Computes the Jaro similarity between two strings."""
        len_s1, len_s2 = len(s1), len(s2)

        if len_s1 == 0 or len_s2 == 0:
            return 0.0

        match_distance = (max(len_s1, len_s2) // 2) - 1
        s1_matches = [False] * len_s1
        s2_matches = [False] * len_s2

        matches = 0
        transpositions = 0

        # Count matches
        for i in range(len_s1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len_s2)

            for j in range(start, end):
                if s2_matches[j]:  # Already matched
                    continue
                if s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        s1_mapped = [s1[i] for i in range(len_s1) if s1_matches[i]]
        s2_mapped = [s2[i] for i in range(len_s2) if s2_matches[i]]

        for i in range(len(s1_mapped)):
            if s1_mapped[i] != s2_mapped[i]:
                transpositions += 1

        transpositions //= 2

        return (matches / len_s1 + matches / len_s2 + (matches - transpositions) / matches) / 3


    def jaro_winkler_similarity(s1, s2, p=0.1):
        """Computes Jaro-Winkler similarity between two strings."""
        jaro_sim = jaro_similarity(s1, s2)

        # Find the common prefix length (up to 4 characters)
        prefix_length = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix_length += 1
            else:
                break

        calculated_similarity = jaro_sim + (prefix_length * p * (1 - jaro_sim))
        return min(max(calculated_similarity, 0.0), 1.0)  # Ensure value is between 0-1



    def compute_similarity_matrix(responses, similarity_func, threshold):
        """Computes a similarity matrix based on the given similarity function."""
        # preparing to use jaro winkler similarity test
        n = len(responses)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_func(responses[i], responses[j])
                if similarity >= threshold:
                    similarity_matrix[i, j] = 1
                    similarity_matrix[j, i] = 1

        return similarity_matrix

    def simple_fuzzy_match(s1, s2):
        """A simple fuzzy matching function similar to `fuzz.partial_ratio`."""
        s1, s2 = clean_text(s1).lower(), clean_text(s2).lower()
        if s1 in s2 or s2 in s1:
            return 100  # Exact substring match

        common_chars = sum((c in s2) for c in s1)
        return int((2 * common_chars / (len(s1) + len(s2))) * 100)

    def extract_best_match(query, choices, threshold=85):
        """Finds the best match for a query string within a list of choices."""
        best_match, best_score = None, 0
        for choice in choices:
            score = simple_fuzzy_match(query, choice)
            if score > best_score:
                best_match, best_score = choice, score
        return (best_match, best_score) if best_score >= threshold else (None, 0)



    def load_training_clusters(cluster_path):
        """Load clusters with empty indices lists"""
        train_cluster_df = pd.read_csv(cluster_path)

        # Initialize empty indices if column missing or empty
        if 'Indices' not in train_cluster_df.columns:
            train_cluster_df['Indices'] = [[] for _ in range(len(train_cluster_df))]

        train_clusters = {}
        for _, row in train_cluster_df.iterrows():
            train_clusters[row['Cluster Id']] = {
                'Count': row['Count'],
                'Cluster Words': set(row['Cluster Words'].split(', ')),
                'Indices': literal_eval(str(row['Indices'])) if pd.notna(row['Indices']) else []
            }
        return train_clusters

    def update_clusters_with_test_data(test_entries, train_clusters):
        """Properly matches test data to best existing clusters"""
        # Preprocess test data
        test_responses = test_entries[['drinks', 'original_index']].copy()
        test_responses['drinks'] = test_responses['drinks'].astype(str).apply(clean_text)

        # Convert train clusters to comparable format
        train_cluster_words = {
            cluster_id: {
                'words': data['Cluster Words'],
                'count': data['Count'],
                'indices': data['Indices']
            }
            for cluster_id, data in train_clusters.items()
        }

        # Process each test response
        for _, row in test_responses.iterrows():
            test_word = row['drinks']
            test_index = row['original_index']

            best_match = None
            best_score = 0

            # Compare against all existing clusters
            for cluster_id, cluster_data in train_cluster_words.items():
                # Compare against all words in this cluster
                for cluster_word in cluster_data['words']:
                    # Use both similarity measures
                    jaro_score = jaro_winkler_similarity(test_word, cluster_word)
                    fuzzy_score = simple_fuzzy_match(test_word, cluster_word)/100  # Normalize to 0-1

                    # Combined score (you can adjust weights)
                    combined_score = 0.7*jaro_score + 0.3*fuzzy_score

                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = cluster_id

            # Apply threshold (same as training)
            if best_score >= 0.85:  # Your combined threshold
                train_cluster_words[best_match]['count'] += 1
                train_cluster_words[best_match]['indices'].append(test_index)
                train_cluster_words[best_match]['words'].add(test_word)
            else:
                # Create new cluster
                train_cluster_words[test_word] = {
                    'words': {test_word},
                    'count': 1,
                    'indices': [test_index]
                }

        # Convert back to original format
        updated_clusters = {
            cluster_id: {
                'Cluster Words': data['words'],
                'Count': data['count'],
                'Indices': data['indices']
            }
            for cluster_id, data in train_cluster_words.items()
        }

        return updated_clusters

    def clusters_to_dataframe(updated_clusters):
        """Convert to DataFrame with proper formatting"""
        cluster_data = []
        for cluster_id, data in updated_clusters.items():
            cluster_data.append([
                cluster_id,
                data['Count'],
                ', '.join(sorted(data['Cluster Words'])),
                data['Indices']
            ])

        cluster_df = pd.DataFrame(
            cluster_data,
            columns=['Cluster Id', 'Count', 'Cluster Words', 'Indices']
        )

        # Clean indices
        cluster_df['Indices'] = cluster_df['Indices'].apply(
            lambda x: [int(i) for i in x] if isinstance(x, list) else []
        )

        return cluster_df.sort_values('Count', ascending=False)


    # ===== Usage =====
    # During TRAINING:
    # cluster_df = ... (your existing training code)
    # save_training_clusters(cluster_df, 'training_clusters.csv')

    # Usage
    df_test_entries = df #pd.read_csv('/content/drive/MyDrive/311/cleaned_data_combined_modified 60 ONLY.csv')  # Load your test data
    q6 = df_test_entries[['Q6: What drink would you pair with this food item?']] # q6 is a dataframe
    q6 = q6.rename(columns={'Q6: What drink would you pair with this food item?': 'drinks'})  # making it easier to reference this column
    q6['drinks'] = q6['drinks'].str.lower() # q6 is one column = Pandas Series, also lowercasing all entries
    # print(q6['drinks'])

    #### divide based on median, so we divide the yapping rants vs actual titles (assuming its likely chances)
    word_lengths = q6['drinks'].str.split().explode().str.len()   # this is a series (same order as q6['drinks']) of num words in each entry
    #print(word_lengths)



    small_entries, large_entries = divide_into_short_long(q6, 'drinks')
    train_clusters = load_training_clusters('drink_clusters.csv')
    test_entries = small_entries

    #print(test_entries)
    updated_clusters = update_clusters_with_test_data(test_entries, train_clusters)

    # For processing (keep indices):
    processing_df = clusters_to_dataframe(updated_clusters)

    # For final export (no indices):
    final_df = clusters_to_dataframe(updated_clusters)



    # print(large_entries)


    #######

    refined_clusters = {}
    for cluster_id, data in updated_clusters.items():
        # Create list of (word, index) tuples for each cluster
        words_indices = []
        for word in data['Cluster Words']:
            # Pair each word with all indices in this cluster
            words_indices.extend([(word, idx) for idx in data['Indices']])
        refined_clusters[cluster_id] = words_indices


    #### create a list of tuples: a label and its frequency
    label_freqs = []
    for cluster_id, group in refined_clusters.items():
        # print(group, "\n\n")  # for testing
        freq = len(group)
        if len(label_freqs) == 0:
            label_freqs.append((cluster_id, freq))
        else:
            # compare to see where (based on frequency) this label lands in the list
            inserted = False
            for i in range(len(label_freqs)):
                if freq > label_freqs[i][1]:
                    label_freqs.insert(i, (cluster_id, freq))
                    inserted = True
                    break
            if not inserted:
                label_freqs.append((cluster_id, freq))
    #print(label_freqs)

    #### go through the large_entries and pick out which of the labels appears first
    # once you figure out which label appears first, replace the entry with that label

    labels = [pair[0] for pair in label_freqs]
    #print("before\n")
    #print(large_entries)  # to see the before and after, you have to re-run everything (otherwise this is referencing the new one)

    for index, row in large_entries.iterrows():
        entry = clean_text(row['drinks'])
        words = entry.split()
        # iterate through labels, finds the first one that appears in words
        matched = False
        for label in labels:
            if label in words:
                # easily found a match in existing labels
                large_entries.loc[index, 'drinks'] = label  # loc is from Pandas and allows us to directly modify specific rows/cols in df
                matched = True
                break

        if (not matched):
            # none of the labels appeared in this entry -- manually assign to "none"
            large_entries.loc[index, 'drinks'] = 'none'


    #print(large_entries)


    #### add large_entries data into the clusters
    for i, row in large_entries.iterrows():
        drink_label = row['drinks']
        index = row['original_index']

        # Skip if 'none'
        if drink_label == 'none':
            continue

        # Find matching cluster
        for cluster_id in refined_clusters:
            # Check if label matches any word in this cluster
            if any(drink_label == word for word, _ in refined_clusters[cluster_id]):
                refined_clusters[cluster_id].append((drink_label, index))
                break
        else:
            # Create new cluster if no match found
            refined_clusters[drink_label] = [(drink_label, index)]

    # Convert to final DataFrame
    cluster_data = []
    for cluster_id, words_indices in refined_clusters.items():
        words = list(set(word for word, _ in words_indices))  # Unique words
        indices = [idx for _, idx in words_indices]
        count = len(words_indices)  # Total entries (including duplicates)

        cluster_data.append([
            cluster_id,
            count,
            ", ".join(sorted(words)),
            indices
        ])

    # Create and save final DataFrame
    final_cluster_df = pd.DataFrame(
        cluster_data,
        columns=['Cluster Id', 'Count', 'Cluster Words', 'Indices']
    )
    final_cluster_df = final_cluster_df.sort_values('Count', ascending=False)
    final_cluster_df['Indices'] = final_cluster_df['Indices'].apply(lambda x: [int(i) for i in x])

    #print("\nFinal clusters:")
    #print(final_cluster_df.head())


    #### manually combining some clusters (ex. coke and cola) to further clean the labels
    if "coke" in refined_clusters and "cola" in refined_clusters:
        refined_clusters["coke"].extend(refined_clusters["cola"])  # Merge "cola" into "coke"
        del refined_clusters["cola"]  # Remove the "cola" cluster

    if "ice tea" in refined_clusters and "nestea" in refined_clusters:
        refined_clusters["ice tea"].extend(refined_clusters["nestea"])
        del refined_clusters["nestea"]

    if "boba" in refined_clusters and "bubble tea" in refined_clusters:
        refined_clusters["boba"].extend(refined_clusters["bubble tea"])
        del refined_clusters["bubble tea"]

    # if we didn't find a "sake" cluster_id, create one now -- trying to avoid grouping all of sake into water
    if "sake" not in refined_clusters:
        refined_clusters["sake"] = []

    # find the prospective labels that have "sake" in them
    sake_merge_targets = [cluster_id for cluster_id in refined_clusters if "sake" in cluster_id and cluster_id != "sake"]

    for cluster_id in sake_merge_targets:
        refined_clusters["sake"].extend(refined_clusters[cluster_id])  # Merge "cluster_id into "water"
        del refined_clusters[cluster_id]  # Remove the cluster_id cluster

    # if we didn't find a "water" cluster_id, create one now -- we know there are lots of responses that have "water" but maybe not one that has a singular label
    if "water" not in refined_clusters:
        refined_clusters["water"] = []

    # find the prospective labels that have "water" in them
    water_merge_targets = [cluster_id for cluster_id in refined_clusters if "water" in cluster_id and cluster_id != "water"]

    for cluster_id in water_merge_targets:
        refined_clusters["water"].extend(refined_clusters[cluster_id])  # Merge "cluster_id into "water"
        del refined_clusters[cluster_id]  # Remove the cluster_id cluster

    # Convert refined clusters to DataFrame
    cluster_data = []
    for cluster_id, words_indices in refined_clusters.items():
        words = [word for word, idx in words_indices]
        indices = [idx for word, idx in words_indices]
        count = len(words)

        cluster_data.append([cluster_id, count, ", ".join(words), indices])

    cluster_df = pd.DataFrame(cluster_data, columns=['Cluster Id', 'Count', 'Cluster Words', 'Indices'])
    cluster_df = cluster_df.sort_values(by='Count', ascending=False).reset_index(drop=True) #  use sort_values to reorder the clusters by count
    cluster_df["Indices"] = cluster_df["Indices"].apply(lambda x: [int(i) for i in x])  # convert np.uint64 to int -- otherwise is weird in csv file

    # Print results
    #print(cluster_df)

    question = 'Q6: What drink would you pair with this food item?'
    # Iterate through each row in all_res_mapping
    for index, row in cluster_df.iterrows():
        ids = row['Indices']  # Get the list of IDs for this group
        label = row['Cluster Id']  # Get the label for this group

        # Iterate through each ID in the list
        for id in ids:
            # match the actual df row index with id
            match_index = df_test_entries.index[df_test_entries.index == id]

            # If a match is found, replace the 'question' column with the label
            if not match_index.empty:
                df_test_entries.loc[match_index, question] = label

    df_test_entries = df_test_entries.fillna('none')
    return df_test_entries

#q5
# from trainig
# saved clusters based on unique values / added values in them for both and drop the indices

# get the clusters from saved
# re run clusters for new test data, ensure to saved the indices for test data
# do fuzzy mathc first: where check if words are subsets of eachother and do a rpeprpocess where u remove the outlier
#               words like 'i think of' 'this blabla' 'idk'
# only those with indices in test data will u add as the label back to the og df


def q5_processing(df):
    # print(df)


    # Constants for text processing
    NONE_PHRASES = {
        'none', 'idk', 'no movie', 'not sure', 'nothing',
        'i dont', 'i cant', 'na', 'no', "i don't know"
    }

    FILLER_PHRASES = [
        'i think of', 'comes to mind', 'i thought of',
        'when thinking of', 'to be honest', 'i might think of',
        'probably some', 'nothing comes to mind', 'think about the movie', 'think about',
        'took place', ''
    ]

    def clean_movie_text(text):
        """Enhanced cleaning with optimized phrase removal"""
        text = str(text).lower().strip()

        # Quick check for empty/none responses
        if not text or any(phrase in text for phrase in NONE_PHRASES):
            return 'none'

        # Remove filler phrases more efficiently
        for phrase in FILLER_PHRASES:
            text = text.replace(phrase, '')

        # Advanced cleaning with single regex pass
        text = re.sub(r'[^a-zA-Z0-9\s]|\b\w{1,2}\b', '', text)  # Remove short words
        text = re.sub(r'\s+', ' ', text).strip()

        return text if text else 'none'

    def divide_movie_responses(df, text_col='Q5: What movie do you think of when thinking of this food item?', threshold=7):
        """Split responses into DataFrames for short and long responses"""
        # Create a working copy with just the columns we need
        q5 = df[['id', text_col]].copy()

        # Clean the text and calculate word counts
        q5['cleaned'] = q5[text_col].apply(clean_movie_text)
        q5['word_count'] = q5['cleaned'].str.split().str.len()

        # Split based on threshold
        mask = q5['word_count'] < threshold
        short = q5[mask].copy()
        long = q5[~mask].copy()

        # Prepare DataFrames with consistent column names
        short_df = pd.DataFrame({
            'movies': short['cleaned'],
            'original_index': short['id']
        }).reset_index(drop=True)

        long_df = pd.DataFrame({
            'movies': long['cleaned'],
            'original_index': long['id']
        }).reset_index(drop=True)

        return short_df, long_df




    def load_training_clusters(cluster_path):
        """Optimized cluster loading with error handling"""
        df = pd.read_csv(cluster_path)

        # Convert indices safely
        df['Indices'] = df['Indices'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )

        # Handle float/NaN values in Cluster Words
        def safe_split(words):
            if pd.isna(words):
                return set()
            if isinstance(words, float):
                return set()
            return set(str(words).split(', '))

        clusters = {}
        for _, row in df.iterrows():
            clusters[row['Cluster Id']] = {
                'Count': int(row['Count']),
                'Cluster Words': safe_split(row['Cluster Words']),
                'Indices': [int(i) for i in row['Indices']]
            }
        return clusters

    def simple_fuzzy_match(s1, s2):
        """Safe fuzzy matching that always returns a numeric score"""
        s1 = clean_movie_text(s1) if s1 else ''
        s2 = clean_movie_text(s2) if s2 else ''

        if not s1 or not s2:
            return 0

        if s1 in s2 or s2 in s1:
            return 100

        try:
            common_chars = sum((c in s2) for c in s1)
            return int((2 * common_chars / (len(s1) + len(s2))) * 100)
        except ZeroDivisionError:
            return 0

    def process_movie_clusters(test_entries, train_clusters):
        """Optimized cluster processing with safe scoring"""
        test_data = test_entries[['movies', 'original_index']].copy()
        test_data['clean_movie'] = test_data['movies'].apply(clean_movie_text)

        # Precompute cluster word lists for faster access
        cluster_info = {
            cid: {
                'words': list(data['Cluster Words']),
                'count': data['Count'],
                'indices': data['Indices']
            }
            for cid, data in train_clusters.items()
        }

        updated = train_clusters.copy()

        for _, row in test_data.iterrows():
            text, idx = row['clean_movie'], row['original_index']
            if text == 'none':
                continue

            best_match, best_score = None, 0

            # Optimized comparison loop
            for cid, cinfo in cluster_info.items():
                for word in cinfo['words']:
                    # Early termination if perfect match
                    if text == word:
                        best_match, best_score = cid, 1.0
                        break

                    # Safe similarity calculations
                    js = jaro_winkler_similarity(text, word) or 0.0
                    fs = (simple_fuzzy_match(text, word) or 0)/100
                    score = 0.7*js + 0.3*fs

                    if score > best_score:
                        best_score, best_match = score, cid

                if best_score == 1.0:  # Early exit
                    break

            # Apply matching logic
            if best_score >= 0.85 and best_match:
                updated[best_match]['Count'] += 1
                updated[best_match]['Indices'].append(idx)
                updated[best_match]['Cluster Words'].add(text)
            else:
                updated.setdefault(text, {
                    'Count': 1,
                    'Cluster Words': {text},
                    'Indices': [idx]
                })

        return updated


    def process_long_entries(long_entries, clusters):
        """Handle long entries with simplified matching"""
        long_entries['clean_movie'] = long_entries['movies'].apply(clean_movie_text)

        for _, row in long_entries.iterrows():
            text, idx = row['clean_movie'], row['original_index']
            if text == 'none':
                continue

            matched = False
            for cid in clusters:
                if text in clusters[cid]['Cluster Words']:
                    clusters[cid]['Count'] += 1
                    clusters[cid]['Indices'].append(idx)
                    matched = True
                    break

            if not matched:
                clusters[text] = {
                    'Count': 1,
                    'Cluster Words': {text},
                    'Indices': [idx]
                }

        return clusters

    def save_clusters(clusters):
        """Optimized cluster saving"""
        records = []
        for cid, data in clusters.items():
            records.append({
                'Cluster Id': cid,
                'Count': data['Count'],
                'Cluster Words': ', '.join(sorted(data['Cluster Words'])),
                'Indices': data['Indices']
            })

        df = pd.DataFrame(records)
        df['Indices'] = df['Indices'].apply(lambda x: str(x))
        df.sort_values('Count', ascending=False, inplace=True)
        return df

    # Main execution flow

    question = 'Q5: What movie do you think of when thinking of this food item?'

    """End-to-end processing pipeline"""
    # Load and prepare data
    train_clusters = load_training_clusters('movie_clusters.csv')
    test_data = df # pd.read_csv(test_path).fillna({question: 'none'})
    test_data[question] = test_data[question].astype(str).str.lower()

    # Process responses
    short, long = divide_movie_responses(test_data)
    clusters = process_movie_clusters(short, train_clusters)
    clusters = process_long_entries(long, clusters)

    # Save results
    df = save_clusters(clusters)

    model_df=test_data

    def parse_indices(indices_str):
        """Convert string representation of list to list of integers"""
        if pd.isna(indices_str) or indices_str == '[]':
            return []
        try:
            # Remove brackets and split by commas
            return [int(id_str.strip()) for id_str in indices_str.strip('[]').split(',')]
        except:
            return []

    # Iterate through each row in all_res_mapping
    for index, row in df.iterrows():
        ids = parse_indices(row['Indices'])
        label = row['Cluster Id']  # Get the label for this group

        # Iterate through each ID in the list
        for id in ids:
            # Find the row in qq where the 'id' column matches the current ID
            match_index = model_df[model_df['id'] == id].index

            # If a match is found, replace the 'question' column with the label
            if not match_index.empty:
                model_df.loc[match_index, question] = label

    model_df = model_df.fillna('none')
    return model_df

def preprocess_data(df):
    df = q2_processing(df)
    df = q4_processing(df)
    df = q6_processing(df)
    df = q5_processing(df)

    return df

