import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from split_train_test import split_data
import gensim.downloader as api
import string
from gen_description import generate_breed_description
from gen_comparison import compare_breeds
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


word2vec_model = api.load("glove-wiki-gigaword-100")

input_file = "cat_personality.xlsx"
output_train_file = "train_data.xlsx"
output_test_file = "test_data.xlsx"

split_data(input_file, output_train_file, output_test_file)

train_data = pd.read_excel(output_train_file)
test_data = pd.read_excel(output_test_file)

numeric_columns = ['Time Spent', 'Shy', 'Calm', 'Fearful', 'Intelligent','Affectionate', 'Friendly',
                   'Independent', 'Dominant', 'Aggressive', 'Predictable', 'Distracted', 'Vocal', 'Hair', 'Pointy Ears', 'Pattern',
                    'Gray coat','Limp Body',   'Size']

scaler = StandardScaler()
train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])

breed_mapping = {breed: idx for idx, breed in enumerate(train_data['Breed'].unique())}
train_data['Breed'] = train_data['Breed'].map(breed_mapping)
test_data['Breed'] = test_data['Breed'].map(breed_mapping)

X_train = train_data[numeric_columns].values
y_train = train_data['Breed'].values
X_test = test_data[numeric_columns].values
y_test = test_data['Breed'].values


def one_hot_encoding(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = one_hot_encoding(y_train, len(breed_mapping))
y_test_one_hot = one_hot_encoding(y_test, len(breed_mapping))

input_size = len(numeric_columns)
hidden_size = 1000
output_size = len(breed_mapping)
learning_rate = 0.01
epochs = 300
dropout_rate = 0.2
lambda_reg = 0.01

np.random.seed(32)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
b2 = np.zeros((1, output_size))


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def apply_dropout(A, rate):
    mask = (np.random.rand(*A.shape) < (1 - rate)).astype(float)
    return A * mask

def forward_propagation(X, rate=0, is_training=True):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    if is_training:
        A1_dropout = apply_dropout(A1, rate)
    else:
        A1_dropout = A1
    Z2 = np.dot(A1_dropout, W2) + b2
    A2 = softmax(Z2)
    return A1, A2


def backward_propagation(X, A1, A2, y, lambda_reg):
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def count_misclassifications(predictions, true_labels):
    return np.sum(predictions != true_labels)

def create_batches(X, y, batch_size):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    num_batches = len(X) // batch_size
    for i in range(num_batches):
        X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
        y_batch = y_shuffled[i * batch_size:(i + 1) * batch_size]
        yield X_batch, y_batch

    if len(X) % batch_size != 0:
        X_batch = X_shuffled[num_batches * batch_size:]
        y_batch = y_shuffled[num_batches * batch_size:]
        yield X_batch, y_batch

A1_test, A2_test = forward_propagation(X_test, dropout_rate)
predictions_before = np.argmax(A2_test, axis=1)
misclassifications_before = count_misclassifications(predictions_before, y_test)
accuracy_before = accuracy_score(y_test, predictions_before)

print(f"Accuracy înainte de antrenament: {accuracy_before * 100:.2f}%")
print(f"Număr greșeli înainte de antrenament: {misclassifications_before}")

losses = []
accuracies = []

batch_size = 32
for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in create_batches(X_train, y_train_one_hot, batch_size):
        A1, A2 = forward_propagation(X_batch, dropout_rate)

        dW1, db1, dW2, db2 = backward_propagation(X_batch, A1, A2, y_batch, lambda_reg)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        batch_loss = -np.mean(np.sum(y_batch * np.log(A2), axis=1)) + lambda_reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        epoch_loss += batch_loss

    losses.append(epoch_loss / len(X_train))

    A1_test, A2_test = forward_propagation(X_test, dropout_rate)
    predictions_after = np.argmax(A2_test, axis=1)
    accuracy_after = accuracy_score(y_test, predictions_after)
    accuracies.append(accuracy_after)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(X_train)}")

A1_test, A2_test = forward_propagation(X_test, dropout_rate)
predictions_after = np.argmax(A2_test, axis=1)

accuracy_after = accuracy_score(y_test, predictions_after)

print(f"\nAccuracy după antrenament: {accuracy_after * 100:.2f}%")


attribute_keywords = {
    'Time Spent': ['time', 'spent', 'duration', 'hours', 'loyal'],
    'Shy': ['shy', 'timid', 'reserved'],
    'Calm': ['calm', 'relaxed', 'peaceful', 'lazy'],
    'Fearful': ['fearful', 'scared', 'anxious', 'nervous'],
    'Intelligent': ['intelligent', 'smart', 'clever'],
    'Affectionate': ['affectionate', 'loving', 'caring'],
    'Friendly': ['friendly', 'sociable', 'approachable'],
    'Independent': ['independent', 'self-reliant', 'autonomous'],
    'Dominant': ['dominant', 'assertive', 'bossy'],
    'Aggressive': ['aggressive', 'hostile', 'violent'],
    'Predictable': ['predictable', 'consistent', 'reliable'],
    'Distracted': ['distracted', 'unfocused', 'inattentive'],
    'Vocal': ['vocal', 'talkative', 'noisy', 'meows', 'meow'],
    'Hair': ['hair','fur', 'furry', 'hairy', 'coat', 'length', 'fluffy', 'hairless'],
    'Pointy Ears': ['pointy ears', 'sharp ears', 'pointed'],
    'Pattern': ['pattern', 'design', 'markings'],
    'Gray coat': ['gray', 'gray fur', 'gray coat', 'gray hair'],
    'Limp Body': ['limp', 'soft', 'relaxed'],
    'Size': ['size', 'body']
}

positive_emph_dictionary = {
    'very': 2,
    'really': 2,
    'extremely': 2,
    'huge': 2,
    'fluffy': 2,
    'big': 1,
    'a little bit': 1,
    'large': 1,
    'easily': 2
}

negative_emph_dictionary = {
    'not': 4,
    'leopard': 5,
    'spotted': 5,
    'small': 4,
    'hairless': 4,
    'no': 4
}

initial_attribute_vector = {
    'Time Spent': 0,
    'Shy': 0,
    'Calm': 0,
    'Fearful': 0,
    'Intelligent': 0,
    'Affectionate': 0,
    'Friendly': 0,
    'Independent': 0,
    'Dominant': 0,
    'Aggressive': 0,
    'Predictable': 0,
    'Distracted': 0,
    'Vocal': 0,
    'Hair': 1,
    'Pointy Ears': 0,
    'Pattern': 0,
    'Gray coat': 0,
    'Limp Body': 0,
    'Size': 1
}

def clean_word(word):
    return word.strip(string.punctuation)


def find_attribute(word, attribute_keywords):
    for idx, (attribute, keywords) in enumerate(attribute_keywords.items()):
        if word in keywords:
            if attribute == 'Limp Body':
                return idx, 1
            if attribute == 'Pointy Ears':
                return idx, 1
            if attribute == 'Gray coat':
                return idx, 1
            if attribute == 'Hair':
                return idx, 0
            return idx, 3

    return None, None



def process_description(description, attribute_keywords, positive_emph_dictionary, negative_emph_dictionary, initial_attribute_vector):
    tokens = description.lower().split()
    attribute_vector = initial_attribute_vector.copy()
    processed_positions = set()
    token_count = len(tokens)

    i = 0
    while i < token_count:
        if i in processed_positions:
            i += 1
            continue

        word = tokens[i]

        if word in positive_emph_dictionary or word in negative_emph_dictionary:

            if i + 1 < token_count and (i + 1) not in processed_positions:
                next_word = tokens[i + 1]
                idx, score = find_attribute(next_word, attribute_keywords)
                if idx is not None:
                    attribute_name = list(attribute_keywords.keys())[idx]
                    if word in positive_emph_dictionary:
                        score += positive_emph_dictionary[word]
                    elif word in negative_emph_dictionary:
                        score -= negative_emph_dictionary[word]
                    attribute_vector[attribute_name] += score
                    processed_positions.add(i + 1)

            processed_positions.add(i)
            i += 2
            continue

        idx, score = find_attribute(word, attribute_keywords)
        if idx is not None:
            attribute_name = list(attribute_keywords.keys())[idx]
            attribute_vector[attribute_name] += score
            processed_positions.add(i)

        i += 1
    for attribute_name in attribute_vector:
        if attribute_vector[attribute_name] < 0:
            attribute_vector[attribute_name] = 0
        else:
            if attribute_vector[attribute_name] > 5:
                attribute_vector[attribute_name] = 5

    print("Updated attribute vector:", attribute_vector)
    return attribute_vector

inverse_breed_mapping = {v: k for k, v in breed_mapping.items()}


def predict_breed(user_input_vector, rate=0, is_training=False):
    _, A2 = forward_propagation(user_input_vector, rate, is_training)
    predicted_index = np.argmax(A2, axis=1)[0]
    predicted_breed = inverse_breed_mapping[predicted_index]
    return predicted_breed, A2

def call_guess(df):
    description = input("Introduceți descrierea pisicii (ex: 'My cat is friendly and intelligent'): ")

    attribute_vector = process_description(description, attribute_keywords, positive_emph_dictionary,
                                           negative_emph_dictionary,
                                           initial_attribute_vector)
    attribute_vector_normalized = scaler.transform([list(attribute_vector.values())])
    predicted_breed, _ = predict_breed(attribute_vector_normalized)

    if predicted_breed == "BEN":
        predicted_breed = "Bengal"
    elif predicted_breed == "BIRM":
        predicted_breed = "Birman"
    elif predicted_breed == "BRI":
        predicted_breed = "British Shorthair"
    elif predicted_breed == "CHA":
        predicted_breed = "Chartreux"
    elif predicted_breed == "EUR":
        predicted_breed = "European Shorthair"
    elif predicted_breed == "MCO":
        predicted_breed = "Maine Coon"
    elif predicted_breed == "PER":
        predicted_breed = "Persian"
    elif predicted_breed == "RAG":
        predicted_breed = "Ragdoll"
    elif predicted_breed == "SAV":
        predicted_breed = "Savannah"
    elif predicted_breed == "SPH":
        predicted_breed = "Sphynx"
    elif predicted_breed == "SIA":
        predicted_breed = "Siamese"
    elif predicted_breed == "TUV":
        predicted_breed = "Turkish Angora"

    print(f"Rasa prezisă a pisicii este: {predicted_breed}")


def main():
    file_path = "cat_personality.xlsx"
    df = pd.read_excel(file_path)

    while True:
        print("\nChoose an option:")
        print("1. Guess the breed based on description")
        print("2. Get a description of a breed")
        print("3. Compare two breeds")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            call_guess(df)
        elif choice == "2":
            breed_name = input("Enter the full name of the breed: ").strip()
            result = generate_breed_description(df, breed_name)
            print(result)
        elif choice == "3":
            breed1 = input("Enter the first breed name: ").strip()
            breed2 = input("Enter the second breed name: ").strip()
            result = compare_breeds(df, breed1, breed2)
            print(result)
        elif choice == "4":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

