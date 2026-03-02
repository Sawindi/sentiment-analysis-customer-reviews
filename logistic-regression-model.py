# Waruni Sawindi Liyanapathirana

from collections import Counter
import random
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

STOPWORDS = {"the", "a", "an", "is", "and", "to", "of", "in", "that", "it", "for", "on", "with", "as", "was", "but", "be", "at", "by", "this", "from", "or", "are", "so", "if", "not", "its"}

# Preprocessing
def preprocess_text(text):
    """
    This function cleans and preprocesses a review string.

    Steps:
        1. Converts text to lowercase.
        2. Removes punctuation and numerical characters.
        3. Tokenises text using whitespace.
        4. Removes stopwords.

    Parameters:
        text (str): Raw review text.

    Returns:
        cleaned_tokens (list): List of cleaned word tokens.

    Assumptions:
        Input text is in English.
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
            # Split line from the right - last value is the label
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

def to_matrix(vectors):
    """
    this function converts Bag-of-Words vectors to numeric format.

    Parameters:
        vectors (list): Bag-of-Words vectors.

    Returns:
        list: Numeric matrix suitable for Logistic Regression.
    """
    return [list(map(float, v)) for v in vectors]

# Main execution
if __name__ == "__main__":

    # Load dataset
    reviews, labels = load_dataset("24135861_Restaurant_Reviews.tsv")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels)

    # Build vocabulary from training data only
    vocab = build_vocabulary(X_train)

    # Vectorize reviews
    X_train_vec = vectorize_reviews(X_train, vocab)
    X_test_vec = vectorize_reviews(X_test, vocab)

    # Convert to numeric matrix
    X_train_mat = to_matrix(X_train_vec)
    X_test_mat = to_matrix(X_test_vec)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_mat, y_train)

    # Predict
    predictions = model.predict(X_test_mat)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Logistic Regression Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)