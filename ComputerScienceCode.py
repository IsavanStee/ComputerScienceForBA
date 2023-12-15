import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
import random
import itertools
from itertools import combinations
import math
import psutil
import cProfile
import pstats

random_seed = 1
random.seed(random_seed)

# Specify the path to JSON file
json_file_path = '/Downloads/ComputerScienceAssignment.json'

with open("TVs-all-merged.json", 'r') as file:
    data = json.load(file)

# The JSON data is converted to a DataFrame
df = pd.DataFrame([(model_id, model['title']) for model_id, model_list in data.items() for model in model_list],
                  columns=['model_id', 'title'])
# Add a new column 'brand' to the DataFrame
df['brand'] = df['model_id'].apply(lambda model_id: data[model_id][0]['featuresMap'].get('Brand', None))


# This method is used to clean the title
def clean_title(title):
    # Define variations of 'inch'
    inch_patterns = ['Inch', 'inches', '"', '-inch', ' inch', 'inch']

    # Define variations of 'hertz'
    hertz_patterns = ['Hertz', 'hertz', 'Hz', 'HZ', ' hz', '-hz', 'hz']

    # Replace inch patterns with 'inch'
    for pattern in inch_patterns:
        title = re.sub(pattern, 'inch', title)

    # Replace hertz patterns with 'hz'
    for pattern in hertz_patterns:
        title = re.sub(pattern, 'hz', title)

    # Replace upper-case characters with lower-case characters
    title = title.lower()

    # Remove spaces and non-alphanumeric tokens in front of the units
    title = re.sub(r'\W+', ' ', title)

    return title


# Apply the cleaning function to the 'title' column
df['cleaned_title'] = df['title'].apply(clean_title)

# This method returns a list with model words of 1 TV title
def extract_model_words(cleaned_title):
    regex = '[a-zA-Z]+[0-9]+[a-zA-Z0-9]*|[a-zA-Z0-9]*[0-9]+[a-zA-Z]+|[0-9]+[.][0-9]+[a-zA-Z]*'

    mw = re.findall(regex, cleaned_title)
    model_words = list(set(mw))  # Use a set to remove duplicates
    return model_words

# This method returns a set with all model words of all 1624 TVS
def extract_all_model_words(dataframe):
    all_model_words = set()

    # Loop through each row in the DataFrame
    for index, row in dataframe.iterrows():
        # Extract model words from the title
        model_title = row['cleaned_title']
        model_words = extract_model_words(model_title)

        # Update the set of distinct model words
        all_model_words.update(model_words)

    return all_model_words


def add_model_words_column(dataframe):
    # Initialize an empty list to store the sets of model words
    model_words_list = []

    # Loop through each row in the DataFrame
    for index, row in dataframe.iterrows():
        tv_title = row['cleaned_title']
        model_words_title = extract_model_words(tv_title)
        model_words_list.append(set(model_words_title))

    # Add a new column 'model_words' to the DataFrame
    dataframe['model_words_title'] = model_words_list

# This method creates a binary vector that indicates whether each model word is present or not in
# cleaned title of the product
def obtain_binary_vector(product, all_model_words):
    title = product['cleaned_title']
    binary_vector = {mw: 1 if mw in extract_model_words(title) else 0 for mw in all_model_words}
    return binary_vector

# This method produces the binary matrix based on the binary vectors obtained before
def create_binary_matrix(dataframe, all_model_words):
    binary_matrix = []

    # Loop through each row in the DataFrame
    for index, row in dataframe.iterrows():
        binary_vector = obtain_binary_vector(row, all_model_words)
        binary_matrix.append(list(binary_vector.values()))
    transposed_matrix = np.array(binary_matrix).T
    return transposed_matrix


# ###################################################### MINHASHING ###################################################

#This method performs min-hashing on the binary matrix to create the signature matrix, where each row of the signature matrix corresponds to a permutation, and each column corresponds to a product.
def minhash(binary_matrix, num_permutations):
    num_products = binary_matrix.shape[1]
    num_model_words = binary_matrix.shape[0]

    np.random.seed(1)
    # Initialize the signature matrix
    signature_matrix = np.zeros((num_permutations, num_products), dtype=int)

    # Generate random permutations
    permutations = [np.random.permutation(num_model_words) for _ in range(num_permutations)]

    # Perform minhashing for each permutation
    for i in range(num_permutations):
        permutation = permutations[i]

        for j in range(num_products):
            # Find the index of the first 1 in the permuted binary vector
            index_first_one = np.argmax(binary_matrix[permutation, j] == 1)
            signature_matrix[i, j] = index_first_one

    return signature_matrix


# ###################################################### LSH ###########################################################

# This method performs Locality-Sensitive Hashing (LSH) on the signature matrix to identify
# candidate pairs of identical products.
def locality_sensitive_hashing(signature_matrix, bands, rows):
    num_products = signature_matrix.shape[1]

    # Initialize a hash table to store candidate pairs
    hash_table = defaultdict(list)

    # Divide the signature matrix into bands
    for band in range(bands):
        band_start = band * rows
        band_end = (band + 1) * rows

        # Hash each product in the band to a bucket
        for product_id in range(num_products):
            hashed_value = hash(tuple(signature_matrix[band_start:band_end, product_id]))
            hash_table[hashed_value].append(product_id)

    # Identify candidate pairs from the hash table
    candidate_pairs = set()
    for bucket, products_in_bucket in hash_table.items():
        if len(products_in_bucket) > 1:
            # If there are at least two products in the same bucket, consider them as candidate pairs
            candidate_pairs.update([(p1, p2) for p1 in products_in_bucket for p2 in products_in_bucket if p1 < p2])

    return list(candidate_pairs)


# ##################################################Classification############################################

# This method determines the Jaccard similarity between two sets.
def jaccard_similarity(set_a, set_b):
    if len(set_a.union(set_b)) == 0 or len(set_a.intersection(set_b)) == 0:
        jaccard_sim = 0
    else:
        jaccard_sim = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    return jaccard_sim

# This method performs classification to identify high similarity pairs among the candidate pairs, where high similarity pairs represent pairs with Jaccard similarity above the specified threshold.
def classification_to_find_duplicates(dataframe, candidate_pairs, num_tvs, jaccard_threshold=0.8):
    n_lsh_duplicates = 2 * len(candidate_pairs)
    sim_matrix = np.zeros((num_tvs, num_tvs,))  # similarity matrix

    # Populate similarity matrix
    for i, j in candidate_pairs:
        set_a = dataframe.loc[i, 'model_words_title']  # set of mw of 1 element of pair
        set_b = dataframe.loc[j, 'model_words_title']  # set of mw of the other element of pair
        similarity = jaccard_similarity(set_a, set_b)
        sim_matrix[i, j] = similarity
        sim_matrix[j, i] = similarity  # Since similarity is symmetric

    # Find high similarity pairs
    high_similarity_pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):  # Avoid duplicates and the diagonal
            similarity = sim_matrix[i, j]

            # Check if similarity is above the threshold
            if similarity > jaccard_threshold:
                high_similarity_pairs.append((i, j))

    return high_similarity_pairs

# This method computes performance measures for duplicate detection based on candidate pairs and high similarity pairs.
def obtain_performance_metrics(candidate_pairs, high_similarity_pairs, dataframe):
    # candidates obtained from lsh+classification
    n_true_positives = 0
    n_false_positives = 0

    n_true_duplicates = 0

    complete_duplicates = 0  # total number of true duplicates
    unique_model_ids = dataframe['model_id'].unique()

    for model_ID in unique_model_ids:
        indices_duplicates = np.where(dataframe['model_id'] == model_ID)[0]
        duplicates_combinations = len(list(combinations(indices_duplicates, 2)))
        n_true_duplicates += duplicates_combinations

    n_comparisons = len(candidate_pairs)

    # candidates obtained from Locality-Sensitive Hashing
    lsh_TP = 0
    lsh_FP = 0

    for k, l in candidate_pairs:  # Pairs obtained from LSH only (what if we would have a perfect classifier)
        model_id_k = dataframe.at[k, 'model_id']
        model_id_l = dataframe.at[l, 'model_id']
        if model_id_l == model_id_k:
            lsh_TP += 1
        else:
            lsh_FP += 1

    for i, j in high_similarity_pairs:  # Pairs obtained from classification with Locality-Sensitive Hashing as preselection
        model_id_i = dataframe.at[i, 'model_id']
        model_id_j = dataframe.at[j, 'model_id']
        if model_id_i == model_id_j:
            n_true_positives += 1
        else:
            n_false_positives += 1

    n_false_negatives = (n_true_duplicates - n_true_positives)

    precision = n_true_positives / (n_false_positives + n_true_positives)
    recall = n_true_positives / n_true_duplicates

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * (precision * recall)) / (precision + recall)

    pairQuality = lsh_TP / n_comparisons
    pairCompleteness = lsh_TP / n_true_duplicates

    if (pairQuality + pairCompleteness) == 0:
        f1_star = 0
    else:
        f1_star = (2 * pairQuality * pairCompleteness) / (pairQuality + pairCompleteness)

    metrics = [precision, recall, f1, f1_star, pairQuality, pairCompleteness]
    return metrics


all_pc_measures = []
all_pq_measures = []
all_fraction_of_comparisons = []
all_f1_measures = []
all_f1star_measures = []

# This method performs bootstrapping to find optimal parameters for Locality-Sensitive Hashing and evaluates performance metrics on the test set.
def bootstrapping(df):
    n_bootstraps = 6

    for bootstrap in range(n_bootstraps):
        f1_measures = []  # for each bootstrap
        f1star_measures = []
        pq_measures = []
        pc_measures = []
        fraction_of_comparisons = []
        t_thresholds = []
        num_tvs = len(df)

        # Create a bootstrap sample for training (63% of the original data)
        train_indices = np.random.choice(range(num_tvs), size=int(0.63 * num_tvs), replace=True)
        train_df = df.iloc[train_indices].reset_index(drop=True)

        num_tvs_train = len(train_df)
        # Extract model words from the training data
        add_model_words_column(train_df)
        all_model_words_train = extract_all_model_words(train_df)
        num_permutations_train = 510
        binary_matrix_train = create_binary_matrix(train_df, all_model_words_train)
        signature_matrix_train = minhash(binary_matrix_train, num_permutations_train)

        possible_b_values = [1, 2, 3, 5, 6, 10, 15, 17, 30, 34, 51, 85, 102, 170, 255, 510]
        possible_r_values = [510, 255, 170, 102, 85, 51, 34, 30, 17, 15, 10, 6, 5, 3, 2, 1]

        best_f1 = -1
        optimal_t = 0
        optimal_n_bands = 0
        optimal_n_rows = 0
        total_possible_comparisons = math.comb(len(train_df), 2)

        # Try different values of b, r, and t
        for b_candidate, r_candidate in zip(possible_b_values, possible_r_values):

            # Perform LSH with the candidate threshold, bands, and rows on the training set
            candidate_pairs_train = locality_sensitive_hashing(signature_matrix_train, b_candidate, r_candidate)
            high_similarity_pairs_train = classification_to_find_duplicates(train_df, candidate_pairs_train, num_tvs_train)
            available_memory = psutil.virtual_memory().available
            print(available_memory)
            performance_metrics = obtain_performance_metrics(candidate_pairs_train, high_similarity_pairs_train, train_df)
            comparison_fraction = len(candidate_pairs_train) / total_possible_comparisons
            f1 = performance_metrics[2]
            f1_star = performance_metrics[3]
            pairQuality = performance_metrics[4]
            pairCompleteness = performance_metrics[5]
            t_treshold = (1 / b_candidate) ** (1 / r_candidate)

            print(f" For {b_candidate} bands and {r_candidate} rows (TRAINING):")
            print(f"PairQuality: {pairQuality}")
            print(f"PairComplete: {pairCompleteness}")
            print(f"f1: {f1}")
            print(f"f1_star: {f1_star}")
            print(f"t_threshold: {t_treshold}")
            print()

            # Update optimal values if a better F1-measure is found
            if f1 > best_f1:
                best_f1 = f1
                optimal_n_bands = b_candidate
                optimal_n_rows = r_candidate
                optimal_t = (1 / optimal_n_bands) ** (1 / optimal_n_rows)

        print(f"Optimal Parameters for Bootstrap {bootstrap + 1} (Training):")
        print(f"  Threshold (t): {optimal_t}")
        print(f"  Bands: {optimal_n_bands}")
        print(f"  Rows: {optimal_n_rows}")
        print(f"  best f1: {best_f1}")

        # Perform LSH with the optimal threshold, bands, and rows on the test set
        # Create a test sample (37% of the original data)
        test_indices = np.setdiff1d(range(num_tvs), train_indices)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        num_tvs_test = len(test_df)
        add_model_words_column(test_df)
        all_model_words_test = extract_all_model_words(test_df)
        num_permutations_test = 300
        binary_matrix_test = create_binary_matrix(test_df, all_model_words_test)
        signature_matrix_test = minhash(binary_matrix_test, num_permutations_test)

        possible_b_values_test = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100, 150, 300]
        possible_r_values_test = [300, 150, 100, 75, 60, 50, 30, 25, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1]
        total_possible_comparisons_test = math.comb(len(test_df), 2)

        for b_candidate_test, r_candidate_test in zip(possible_b_values_test, possible_r_values_test):
            candidate_pairs_test = locality_sensitive_hashing(signature_matrix_test, b_candidate_test, r_candidate_test)
            high_similarity_pairs_test = classification_to_find_duplicates(test_df, candidate_pairs_test, num_tvs_test)
            performance_metrics_test = obtain_performance_metrics(candidate_pairs_test, high_similarity_pairs_test, test_df)

            f1_test = performance_metrics_test[2]
            f1_star_test = performance_metrics_test[3]
            pairQuality_test = performance_metrics_test[4]
            pairCompleteness_test = performance_metrics_test[5]
            t_treshold_test = (1 / b_candidate) ** (1 / r_candidate)
            comparison_fraction = len(candidate_pairs_test) / total_possible_comparisons_test

            # Save results for current b and r combination
            f1_measures.append(f1_test)
            f1star_measures.append(f1_star_test)
            pc_measures.append(pairCompleteness_test)
            pq_measures.append(pairQuality_test)
            t_thresholds.append(t_treshold_test)
            fraction_of_comparisons.append(comparison_fraction)

        # Save results for the current bootstrap
        all_pq_measures.append(pq_measures)
        all_pc_measures.append(pc_measures)
        all_f1_measures.append(f1_measures)
        all_f1star_measures.append(f1star_measures)
        all_fraction_of_comparisons.append(fraction_of_comparisons)


# Call the bootstrapping function with your DataFrame

# Call your existing bootstrapping function or any other function you want to profile
bootstrapping(df)

print(all_pq_measures)
print(all_fraction_of_comparisons)
print(all_f1_measures)
print(all_f1star_measures)
print(all_pc_measures)

# Calculate averages over all bootstraps
avg_pq_measures = np.mean(all_pq_measures, axis=0)
avg_fraction_of_comparisons = np.mean(all_fraction_of_comparisons, axis=0)
avg_pc_measures = np.mean(all_pc_measures, axis=0)
avg_f1_measures = np.mean(all_f1_measures, axis=0)
avg_f1star_measures = np.mean(all_f1star_measures, axis=0)

# Create a 2x2 grid of subplots
plt.subplot(2, 2, 1)
plt.plot(avg_fraction_of_comparisons, avg_pq_measures, label='Average PQ')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality')
plt.title('PQ vs Fraction of Comparisons')

plt.subplot(2, 2, 2)
plt.plot(avg_fraction_of_comparisons, avg_pc_measures, label='Average PC')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness')
plt.title('PC vs Fraction of Comparisons')

plt.subplot(2, 2, 3)
plt.plot(avg_fraction_of_comparisons, avg_f1_measures, label='F1')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1')
plt.title('F1 vs Fraction of Comparisons')

plt.subplot(2, 2, 4)
plt.plot(avg_fraction_of_comparisons, avg_f1star_measures, label='F1*')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1*')
plt.title('F1* vs Fraction of Comparisons')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()