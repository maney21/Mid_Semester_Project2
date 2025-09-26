# --- Step 0: Import Necessary Libraries ---
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully.")

# --- Configuration ---
input_file_path = r"C:\Users\manis\Downloads\app.xlsx"
output_file_path = 'reviews_analysis_output.xlsx'
text_column_name = 'text'
# --------------------

try:
    # --- Step 1: Load and Preprocess the Data ---
    print("\n--- Starting Step 1: Data Loading and Preprocessing ---")

    # Download necessary NLTK data (only if not already present)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Load the raw Excel file
    df = pd.read_excel(input_file_path)

    # Clean the data: drop empty rows and ensure text is string type
    df.dropna(subset=[text_column_name], inplace=True)
    df[text_column_name] = df[text_column_name].astype(str)

    # Define stop words for cleaning
    stop_words = set(stopwords.words('english'))

    # Create a function to clean the text
    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        tokens = word_tokenize(text)  # Split into words
        # Keep only alphabetic words and remove stop words
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)

    # Apply the cleaning function to the review column
    df['cleaned_text'] = df[text_column_name].apply(preprocess_text)
    print("Text preprocessing complete.")
    print(df[['text', 'cleaned_text']].head())


    # --- Step 2: Perform Sentiment Analysis ---
    print("\n--- Starting Step 2: Sentiment Analysis ---")

    # Calculate sentiment polarity score for each review
    df['sentiment_score'] = df['cleaned_text'].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Classify sentiment based on the score
    def classify_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['sentiment_score'].apply(classify_sentiment)
    print("Sentiment classification complete.")
    print(df['sentiment_category'].value_counts())


    # --- Step 3: Identify Key Drivers of Sentiment ---
    print("\n--- Starting Step 3: Identifying Key Drivers ---")

    # Separate reviews by sentiment
    positive_reviews = df[df['sentiment_category'] == 'Positive']['cleaned_text']
    negative_reviews = df[df['sentiment_category'] == 'Negative']['cleaned_text']

    # Helper function to get top n-grams
    def get_top_ngrams(corpus, ngram_range=(1, 1), top_k=10):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]

    # Analyze and print top words/phrases for positive reviews
    if not positive_reviews.empty:
        print("\n--- Top Words/Phrases in Positive Reviews ---")
        print(pd.DataFrame(get_top_ngrams(positive_reviews, ngram_range=(1, 1)), columns=['Word', 'Frequency']))
        print("\n")
        print(pd.DataFrame(get_top_ngrams(positive_reviews, ngram_range=(2, 2)), columns=['Phrase', 'Frequency']))
    else:
        print("\nNo positive reviews to analyze.")

    # Analyze and print top words/phrases for negative reviews
    if not negative_reviews.empty:
        print("\n--- Top Words/Phrases in Negative Reviews ---")
        print(pd.DataFrame(get_top_ngrams(negative_reviews, ngram_range=(1, 1)), columns=['Word', 'Frequency']))
        print("\n")
        print(pd.DataFrame(get_top_ngrams(negative_reviews, ngram_range=(2, 2)), columns=['Phrase', 'Frequency']))
    else:
        print("\nNo negative reviews to analyze.")


    # --- Step 4: Save Results and Visualize ---
    print("\n--- Starting Step 4: Saving and Visualizing Results ---")

    # Save the full analysis to a new Excel file
    df.to_excel(output_file_path, index=False)
    print(f"Full analysis saved to '{output_file_path}'")

    # Visualize the sentiment distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, order=['Positive', 'Neutral', 'Negative'], palette="viridis")
    plt.title('Distribution of Customer Review Sentiments')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Reviews')
    plt.show()
    print("\nAnalysis complete. Visualization displayed.")

# --- Error Handling ---
except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found. Please check the path.")
except KeyError:
    print(f"Error: The column '{text_column_name}' was not found in the Excel file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
