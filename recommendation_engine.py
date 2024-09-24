import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import normalize
import data_processing
import requests
import pickle

# This relied heavily on information from https://scikit-learn.org/stable/

data_processing.process_data()

pickle_path = "./data/processed_features.pkl"
book_dataframe = pd.read_csv("./data/popular-books-enhanced.csv", dtype={
    "Title": str, "Author": str, "Score": float, "Ratings": float,
    "Shelvings": float, "Published": float, "Description": str,
    "Image": str, "Categories": str, "Language": str
})

with open('./data/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('./data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./data/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open(pickle_path, "rb") as f:
    data = pickle.load(f)
    combined_features = data['combined_features']
    num_category_features = data['num_category_features']
    num_numerical_features = data['num_numerical_features']
    num_description_features = data['num_description_features']


def fetch_book_from_google_books(book_title: str):
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_title}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            if "volumeInfo" in data["items"][0]:
                book_info = data["items"][0]["volumeInfo"]
                return {
                    "title": book_info.get("title", ""),
                    "author": ", ".join(book_info.get("authors", [])),
                    "description": book_info.get("description", ""),
                    "categories": ", ".join(book_info.get("categories", [])),
                    "published": book_info.get("publishedDate", "0").split("-")[0],
                    "averageRating": book_info.get("averageRating", 0),
                    "ratingsCount": book_info.get("ratingsCount", 0)
                }
    return None


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    normalized_features = features / norms
    return normalized_features


def generate_feature_vector(book_info):
    categories = book_info.get('categories', '')
    categories_df = pd.DataFrame({'Categories': [categories]})
    encoded_categories = encoder.transform(categories_df)

    numerical_data = {
        'Score': [book_info.get('averageRating', 0)],
        'Ratings': [book_info.get('ratingsCount', 0)],
        'Shelvings': [0],  # Placeholder for 'Shelvings' if not available
        'Published': [int(book_info.get('published', 0)) if book_info.get('published', '0').isdigit() else 0]
    }
    numerical_df = pd.DataFrame(numerical_data)
    normalized_features = scaler.transform(numerical_df)

    description = book_info.get('description', '')
    description_tfidf = tfidf.transform([description]).toarray()

    feature_vector = np.hstack(
        (encoded_categories, normalized_features, description_tfidf))

    return feature_vector.reshape(1, -1)


def recommend_books(book_title: str, top_n=5,
                    category_weight=1.0,
                    numerical_weight=1.0,
                    description_weight=1.0):
    book_found_locally = False

    def apply_weights(features):
        features[:, :num_category_features] *= category_weight
        start_idx = num_category_features
        end_idx = start_idx + num_numerical_features
        features[:, start_idx:end_idx] *= numerical_weight
        features[:, end_idx:] *= description_weight

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms

        return features

    book_index = book_dataframe[book_dataframe['Title'].str.replace(
        r'\(.*?\)', '', regex=True).str.strip().str.lower() == re.sub(r'\(.*?\)', '', book_title).strip().lower()].index
    queried_book_title = ""
    if len(book_index) == 0:
        google_book_info = fetch_book_from_google_books(book_title)
        if google_book_info:
            existing_book = book_dataframe[
                (book_dataframe['Title'] == google_book_info['title']) &
                (book_dataframe['Author'] == google_book_info['author'])
            ]
            if not existing_book.empty:
                queried_book_index = existing_book.index[0]
                queried_book_features = combined_features[[
                    queried_book_index]]
                queried_book_title = existing_book.iloc[0]['Title']
            else:
                queried_book_title = google_book_info["title"]
                queried_book_features = generate_feature_vector(
                    google_book_info)
        else:
            return pd.DataFrame(), np.array([]), np.array([]), ""
    else:
        book_index = book_index[0]
        queried_book_title = book_dataframe.iloc[book_index].get("Title")
        queried_book_features = combined_features[book_index].reshape(1, -1)
        book_found_locally = True

    weighted_combined_features = apply_weights(combined_features.copy())
    weighted_queried_book_features = apply_weights(
        queried_book_features.copy())

    normalized_combined_features = normalize_features(
        weighted_combined_features)
    normalized_queried_book_features = normalize_features(
        weighted_queried_book_features)

    similarities = cosine_similarity(
        normalized_queried_book_features, normalized_combined_features).flatten()

    similar_books_indices = similarities.argsort(
    )[::-1][1:top_n+1] if book_found_locally else similarities.argsort()[::-1][:top_n]

    recommended_books = book_dataframe.loc[similar_books_indices, [
        'Title', 'Author', 'Score', 'Categories']]

    recommended_features = combined_features[similar_books_indices]

    return recommended_books, recommended_features, queried_book_features, queried_book_title
