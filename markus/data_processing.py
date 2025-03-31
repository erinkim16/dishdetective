import pandas as pd
import re
import numpy as np
import statistics 

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
    except ValueError:
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

    #### isolate the column with the data relating to drinks (Q6)
    q6 = df[['Q6: What drink would you pair with this food item?']] # q6 is a dataframe
    q6 = q6.rename(columns={'Q6: What drink would you pair with this food item?': 'drinks'})  # making it easier to reference this column
    q6['drinks'] = q6['drinks'].str.lower() # q6 is one column = Pandas Series, also lowercasing all entries
    # print(q6['drinks'])

    #### divide based on median, so we divide the yapping rants vs actual titles (assuming its likely chances)
    word_lengths = q6['drinks'].str.split().explode().str.len()   # this is a series (same order as q6['drinks']) of num words in each entry
    print(word_lengths)
    # Compute the median word length across the entire column
    median_word_length = word_lengths.median()  # this is a float
    print('median = ', median_word_length)


    small_entries, large_entries = divide_into_short_long(q6, 'drinks')





def q5_processing(df):
   print(df)

def preprocess_data(df):    
    df = q2_processing(df)
    df = q4_processing(df)
    df = q6_processing(df)
    df = q5_processing(df)

    return df
