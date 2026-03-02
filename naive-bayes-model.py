# Waruni Sawindi Liyanapathirana

import re
from collections import Counter
import random
import math
from collections import defaultdict

STOPWORDS = {"the", "a", "an", "is", "and", "to", "of", "in", "that", "it", "for", "on", "with", "as", "was", "but", "be", "at", "by", "this", "from", "or", "are", "so", "if", "not", "its"}

# Preprocessing
def preprocess_text(text):
    """
    This function cleans and preprocesses the review string.

    Steps:
    1. Converts text to lowercase.
    2. Removes punctuation and numerical characters.
    3. Tokenises text using whitespace.
    4. Removes stopwords.

    Parameters:
        text (str): Raw review text

    Returns:
        cleaned_tokens (list): List of cleaned word tokens

    Assumptions:
        Input text is in English
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in STOPWORDS]
    return cleaned_tokens

# Loading dataset
def load_dataset(file_path):
    """
    This function loads and preprocesses the dataset from a file.

    The dataset contains review text and a sentiment label at the end of each line.
    The file is parsed manually to handle non standard separators.

    Parameters:
        file_path (str): Path to the dataset file

    Returns:
        tuple: (reviews, labels)
            reviews (list): List of tokenised reviews
            labels (list): List of sentiment labels (0 or 1)
    """
    reviews = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

         # Skip header line
        for line in lines[1:]:
            # Split line from the right - last value is label
            parts = line.strip().rsplit(maxsplit=1)

            if len(parts) != 2:
                continue

            review_text = parts[0]
            label = parts[1]

            reviews.append(preprocess_text(review_text))
            labels.append(int(label))

    return reviews, labels

# split dataset - 80% => training, 20% => testing
def train_test_split(reviews, labels, test_size=0.2):
    """
    This function splits the dataset into training and testing sets.

    Parameters:
        reviews (list): Tokenised reviews
        labels (list): Corresponding sentiment labels
        test_size (float): Proportion of data used for testing

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    data = list(zip(reviews, labels))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return list(X_train), list(X_test), list(y_train), list(y_test)

# Feature extraction
def build_vocabulary(reviews):
    """
    This function builds a vocabulary from tokenised reviews.

    Parameters:
        reviews (list): Tokenised reviews.

    Returns:
        list: Sorted list of unique words.
    """
    vocabulary = set()
    for review in reviews:
        for word in review:
            vocabulary.add(word)
    return sorted(list(vocabulary))

def vectorize_reviews(reviews, vocabulary):
    """
    This converts reviews into Bag-of-Words vectors.

    Parameters:
        reviews (list): Tokenised reviews.
        vocabulary (list): Vocabulary list.

    Returns:
        list: Bag-of-Words vectors.
    """
    vectors = []
    for review in reviews:
        word_counts = Counter(review)
        vector = [word_counts.get(word, 0) for word in vocabulary]
        vectors.append(vector)
    return vectors

# Naive Bayes training
def train_naive_bayes(reviews, labels, vocabulary):
    """
    This trains a Naive Bayes classifier using Laplace smoothing and log-likelihoods.

    Parameters:
        reviews (list): Training reviews.
        labels (list): Training labels.
        vocabulary (list): Vocabulary that built from training data.

    Returns:
        tuple:
            priors (dict): Log prior probabilities.
            likelihoods (dict): Log likelihoods for each word.
    """
    class_counts = defaultdict(int)
    word_counts = {
        0: defaultdict(int),
        1: defaultdict(int)
    }
    total_words = {0: 0, 1: 0}

    # Count words per class
    for review, label in zip(reviews, labels):
        class_counts[label] += 1
        for word in review:
            word_counts[label][word] += 1
            total_words[label] += 1

    total_reviews = len(labels)

    # Prior probabilities
    priors = {
        0: math.log(class_counts[0] / total_reviews),
        1: math.log(class_counts[1] / total_reviews)
    }

    # Likelihoods with Laplace smoothing
    likelihoods = {0: {}, 1: {}}
    vocab_size = len(vocabulary)

    for label in [0, 1]:
        for word in vocabulary:
            count = word_counts[label][word]
            likelihoods[label][word] = math.log(
                (count + 1) / (total_words[label] + vocab_size)
            )

    return priors, likelihoods

# Prediction Function
def predict(review, priors, likelihoods):
    """
    This function predicts the sentiment label of a review.

    Parameters:
        review (list): Tokenised review.
        priors (dict): Log prior probabilities.
        likelihoods (dict): Log likelihoods.

    Returns:
        int: Predicted sentiment label (0 or 1).
    """
    scores = {}

    for label in [0, 1]:
        score = priors[label]
        for word in review:
            if word in likelihoods[label]:
                score += likelihoods[label][word]
        scores[label] = score

    return max(scores, key = scores.get)

# Model evaluation
def evaluate_model(X_test, y_test, priors, likelihoods):
    """
    This evaluates the classifier using standard performance metrics.

    Returns:
        tuple:
            accuracy, precision, recall, f1, TP, FP, TN, FN
    """
    tp = fp = tn = fn = 0

    for review, true_label in zip(X_test, y_test):
        pred = predict(review, priors, likelihoods)

        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 1 and true_label == 0:
            fp += 1
        elif pred == 0 and true_label == 0:
            tn += 1
        elif pred == 0 and true_label == 1:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return accuracy, precision, recall, f1, tp, fp, tn, fn

# Main execution
if __name__ == "__main__":
    reviews, labels = load_dataset("24135861_Restaurant_Reviews.tsv")
    print("Sample processed review:", reviews[0])
    print("Label:", labels[0])

    # Feature extraction
    vocab = build_vocabulary(reviews)
    vectors = vectorize_reviews(reviews, vocab)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels)

    # Train Naive Bayes
    priors, likelihoods = train_naive_bayes(X_train, y_train, vocab)

    # Test on one review
    prediction = predict(X_test[0], priors, likelihoods)
    print("Actual label:", y_test[0])
    print("Predicted label:", prediction)

    # Model evaluation
    accuracy, precision, recall, f1, tp, fp, tn, fn = evaluate_model(X_test, y_test, priors, likelihoods)

    print("\nModel Evaluation Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print("\nConfusion Matrix:")
    print("TP:", tp, "FP:", fp)
    print("FN:", fn, "TN:", tn)