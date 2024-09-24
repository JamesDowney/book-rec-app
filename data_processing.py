import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os.path

pickle_path = "./data/processed_features.pkl"

# This will take the data from the CSV and process it for cosine similarity
def process_data():

    # Let's not repeat this process if the data has already been processed
    if not os.path.isfile(pickle_path):
        print("Processed data not found! Generating now.")
        book_dataframe = pd.read_csv("./data/popular-books-enhanced.csv", dtype={
            "Title": str, "Author": str, "Score": float, "Ratings": float,
            "Shelvings": float, "Published": float, "Description": str,
            "Image": str, "Categories": str, "Language": str
        })

        # Fill in some blank values
        book_dataframe[['Title', 'Description']] = book_dataframe[[
            'Title', 'Description']].astype(str).fillna('')
        book_dataframe[['Published', 'Ratings', 'Shelvings', 'Score']] = book_dataframe[[
            'Published', 'Ratings', 'Shelvings', 'Score']].fillna(0)

        # One-hot encoding for categories
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_categories = encoder.fit_transform(
            book_dataframe[['Categories']])

        # Use a min max scaler for numerical features
        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        normalized_features = scaler.fit_transform(
            book_dataframe[['Score', 'Ratings', 'Shelvings', 'Published']])

        # tfidf vectorization for the descriptions
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        description_tfidf = tfidf.fit_transform(
            book_dataframe['Description']).toarray()

        # Let's save the data for each set of features, turned out I needed this when I would go on to use weights
        with open('./data/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        with open('./data/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('./data/tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf, f)

        # Combine features and save
        combined_features = np.hstack(
            (encoded_categories, normalized_features, description_tfidf))

        with open(pickle_path, "wb") as f:
            pickle.dump({
                'combined_features': combined_features,
                'num_category_features': encoded_categories.shape[1],
                'num_numerical_features': normalized_features.shape[1],
                'num_description_features': description_tfidf.shape[1]
            }, f)
