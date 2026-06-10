import os
import pandas as pd
import numpy as np
from pathlib import Path
from .config import DATA_CONFIG, DATA_PROCESSING, PROCESSED_DATA_DIR, BASE_DIR

def clean_users_data(users_df, config=DATA_CONFIG['users']):
    """Clean and preprocess users data."""
    users = users_df.copy()
    users[config['age']] = pd.to_numeric(users[config['age']], errors='coerce', downcast='integer')
    users = users[users[config['age']].between(5, 100)]
    users[config['location']] = users[config['location']].astype(str).str.strip()
    users['Country'] = users[config['location']].str.split(',').str[-1].str.strip()
    return users

def clean_books_data(books_df, config=DATA_CONFIG['books']):
    """Clean and preprocess books data."""
    books = books_df.copy()
    books[config['ISBN']] = books[config['ISBN']].astype(str).str.strip().str.upper()
    books[config['title']] = books[config['title']].astype(str).str.strip()
    books[config['author']] = books[config['author']].astype(str).str.strip()
    books[config['year']] = pd.to_numeric(books[config['year']], errors='coerce', downcast='integer')
    books = books[books[config['year']].between(1800, 2024)]
    return books

def clean_ratings_data(ratings_df, config=DATA_CONFIG['ratings']):
    """Clean and preprocess ratings data."""
    ratings = ratings_df.copy()
    ratings[config['rating']] = pd.to_numeric(ratings[config['rating']], errors='coerce')
    ratings = ratings[ratings[config['rating']].between(0, 10)]
    return ratings

def filter_data(books, users, ratings, 
                min_book_ratings=DATA_PROCESSING['min_book_ratings'], 
                min_user_ratings=DATA_PROCESSING['min_user_ratings'], 
                config=DATA_CONFIG):
    """Filter data based on minimum ratings thresholds."""
    # MATDIS: Himpunan - mengelompokkan rating, index-nya adalah himpunan ISBN unik                
    book_ratings_count = ratings.groupby(config['ratings']['ISBN']).size()
    # MATDIS: Himpunan - mengelompokkan rating, index-nya adalah himpunan user ID unik                
    user_ratings_count = ratings.groupby(config['ratings']['user_id']).size()

    # MATDIS: Subset - mempertahankan hanya buku yang memiliki rating >= threshold                
    valid_books = book_ratings_count[book_ratings_count >= min_book_ratings].index
    # MATDIS: Subset - mempertahankan hanya user yang memiliki rating >= threshold                
    valid_users = user_ratings_count[user_ratings_count >= min_user_ratings].index

    # MATDIS: Irisan Himpunan - menggabungkan dua kondisi keanggotaan dengan operator AND               
    filtered_ratings = ratings[
        (ratings[config['ratings']['ISBN']].isin(valid_books)) & # MATDIS: Keanggotaan (∈)
        (ratings[config['ratings']['user_id']].isin(valid_users))  # MATDIS: Keanggotaan (∈)
    ].copy()

    filtered_books = books[books[config['books']['ISBN']].isin(valid_books)].copy()
    filtered_users = users[users[config['users']['user_id']].isin(valid_users)].copy()

    return filtered_books, filtered_users, filtered_ratings

def preprocess_data(books_df, users_df, ratings_df, save=True):
    """Main preprocessing pipeline."""
    try:
        cleaned_users = clean_users_data(users_df) 
        cleaned_books = clean_books_data(books_df)
        cleaned_ratings = clean_ratings_data(ratings_df)

        filtered_books, filtered_users, filtered_ratings = filter_data(
            cleaned_books, cleaned_users, cleaned_ratings 
        )

        if save:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) 
            filtered_books.to_csv(PROCESSED_DATA_DIR / "processed_books.csv", index=False)
            filtered_users.to_csv(PROCESSED_DATA_DIR / "processed_users.csv", index=False)
            filtered_ratings.to_csv(PROCESSED_DATA_DIR / "processed_ratings.csv", index=False)

        return filtered_books, filtered_users, filtered_ratings

    except Exception as e:
        print(f"Error in preprocessing data: {str(e)}")
        return None, None, None

def load_processed_data():
    """Load preprocessed data from the processed directory"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(base_dir, "data", "processed")
        
        # Load processed datasets
        books = pd.read_csv(os.path.join(processed_dir, "processed_books.csv"))
        ratings = pd.read_csv(os.path.join(processed_dir, "processed_ratings.csv"))
        users = pd.read_csv(os.path.join(processed_dir, "processed_users.csv"))
        base_dir = Path(__file__).parent.parent
        processed_dir = base_dir / "data" / "processed"
        
        # Load processed datasets
        books = pd.read_csv(processed_dir / "processed_books.csv")
        ratings = pd.read_csv(processed_dir / "processed_ratings.csv")
        users = pd.read_csv(processed_dir / "processed_users.csv")
        
        return books, users, ratings
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None, None

if __name__ == "__main__":
    # Test preprocessing
    books, users, ratings = preprocess_data()
    if all(df is not None for df in [books, users, ratings]):
        print(f"Books shape: {books.shape}")
        print(f"Users shape: {users.shape}")
        print(f"Ratings shape: {ratings.shape}")
