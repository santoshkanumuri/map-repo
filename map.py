import time
from collections import Counter
import streamlit as st
import yake
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from sentence_transformers import SentenceTransformer
from itertools import combinations
import pandas as pd
import numpy as np
import networkx as nx
from keybert import KeyBERT
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import requests
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
import spacy
from rake_nltk import Rake
from urllib.parse import quote_plus

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Initialize models
model = SentenceTransformer('paraphrase-mpnet-base-v2')
kw_model = KeyBERT(model=model)
nlp = spacy.load('en_core_web_sm')

# Initialize VADER sentiment analyzer and Empath lexicon
analyzer = SentimentIntensityAnalyzer()
lexicon = Empath()

def rake_extract_keywords(tweet):
    r = Rake()
    r.extract_keywords_from_text(tweet)
    return r.get_ranked_phrases()

def yake_extract_keywords(tweet):
    language = "en"
    max_ngram_size = 4
    deduplication_threshold = 0.1
    num_of_keywords = max(1, len(tweet.split()) // 2)
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        top=num_of_keywords,
        features=None
    )
    keywords = custom_kw_extractor.extract_keywords(tweet)
    return [kw[0].lower() for kw in keywords]

def retrieve_tweets(bearer_token, search_str, num_tweets):
    search_str = f"{search_str} -is:retweet"
    keyword = quote_plus(search_str)
    total_tweets = num_tweets
    max_results_per_request = 100
    tweets = []
    endpoint = f"https://api.twitter.com/2/tweets/search/recent?query={keyword}&max_results={max_results_per_request}"
    headers = {"Authorization": f"Bearer {bearer_token}"}

    def fetch_tweets(url):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    next_token = None
    while len(tweets) < total_tweets:
        if next_token:
            url = f"{endpoint}&next_token={next_token}"
        else:
            url = endpoint

        data = fetch_tweets(url)
        new_tweets = data.get('data', [])
        tweets.extend(new_tweets)

        if 'next_token' not in data.get('meta', {}):
            break
        next_token = data['meta']['next_token']

        if len(tweets) >= total_tweets:
            break

    tweets = tweets[:total_tweets]
    cleaned_tweets = []
    for tweet in tweets:
        t = tweet['text']
        t = re.sub(r"(?:\@|https?\://|\#)\S+", "", t)
        cleaned_tweets.append(t)
    return cleaned_tweets

def translate_tweets(tweets):
    translated_tweets = []
    error_index = []
    for i, tweet in enumerate(tweets):
        try:
            translate_text = GoogleTranslator(source='auto', target='en').translate(tweet[:500])
            translated_tweets.append(translate_text)
        except Exception as e:
            print(f"Translation error at index {i}: {e}")
            error_index.append(i)
    return translated_tweets, error_index

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        cleaned_tweet = extract_text(tweet)
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets

def extract_text(tweet):
    tweet = re.sub(r'.*?@\w+:?', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def analyze_sentiment(text):
    return analyzer.polarity_scores(text)

def get_sentiments(tweets):
    sentiment_scores = [analyze_sentiment(tweet) for tweet in tweets]
    return sentiment_scores

def analyze_sentiment_empath(tweets, sentiment_scores, empath_scores):
    df = pd.DataFrame({'Tweet': tweets})
    df['sentiment_neg'] = [score['neg'] for score in sentiment_scores]
    df['sentiment_neu'] = [score['neu'] for score in sentiment_scores]
    df['sentiment_pos'] = [score['pos'] for score in sentiment_scores]
    df['sentiment_compound'] = [score['compound'] for score in sentiment_scores]
    df['vader_compound_rounded'] = 'neu'
    df.loc[df['sentiment_compound'] > 0, 'vader_compound_rounded'] = 'pos'
    df.loc[df['sentiment_compound'] < 0, 'vader_compound_rounded'] = 'neg'

    dfEmpath = pd.DataFrame.from_records(empath_scores)
    dfEmpath = dfEmpath.loc[:, (dfEmpath != 0).any(axis=0)]
    df = pd.concat([df, dfEmpath], axis=1)
    return df

def get_emotion_totals(df, top_percent):
    emotion_columns = df.drop(columns=['Tweet', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound', 'vader_compound_rounded'])
    column_wise_sum = emotion_columns.sum(axis=0)
    emotion_totals = dict(column_wise_sum)
    top_count = max(1, int(len(emotion_totals) * top_percent))
    sorted_items = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
    top_percent_items = sorted_items[:top_count]
    top_percent_dict = dict(top_percent_items)
    return top_percent_dict

def create_square_matrix(top_dict, df):
    column_names = list(top_dict.keys())
    square_matrix_df = pd.DataFrame(0, index=column_names, columns=column_names)
    for key in column_names:
        count = df[key].sum()
        square_matrix_df.loc[key, key] = count
    for key1, key2 in combinations(column_names, 2):
        if key1 in top_dict and key2 in top_dict:
            count = ((df[key1] == 1) & (df[key2] == 1)).sum()
            square_matrix_df.loc[key1, key2] = count
            square_matrix_df.loc[key2, key1] = count
    return square_matrix_df

def plot_graph(square_matrix, title):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    import streamlit as st

    # Convert the square matrix to a DataFrame if it's not already one
    if not isinstance(square_matrix, pd.DataFrame):
        df = pd.DataFrame(square_matrix)
    else:
        df = square_matrix.copy()
    
    # Get emotion names as node labels
    node_labels = df.index.tolist()
    
    # Convert square matrix to a NumPy array
    matrix = df.values
    
    # Create a graph from the NumPy array
    G = nx.from_numpy_array(matrix)
    
    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Use Kamada-Kawai layout for positioning the nodes
    pos = nx.kamada_kawai_layout(G)
    
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the graph
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels={i: label for i, label in enumerate(node_labels)},
        node_color='skyblue',
        node_size=1000,
        font_size=9,
        font_weight='bold',
        edge_color='#D3D3D3',
        width=1,
        node_shape='o',
        edgecolors='none'
    )
    
    # Set title and turn off the axis
    ax.set_title(title)
    ax.axis('off')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Close the figure to free up memory
    plt.close(fig)


def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [word.strip('.,!?"') for word in words]
    words = [word for word in words if word not in stop_words]
    return words

def create_graph_with_edges(words, counter, title):
    G = nx.Graph()
    for word1 in words:
        for word2 in words:
            if word1 != word2:
                G.add_edge(word1, word2, weight=min(counter[word1], counter[word2]))
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True,
        node_color='skyblue', node_size=1000, edge_color='#D3D3D3', linewidths=1,
        font_size=7, width=1, node_shape='o', edgecolors='none'
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='#D3D3D3')
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

def filter_top_x_percent(word_set, word_counter, top_percent):
    total_words = sum(word_counter.values())
    top_x_count = max(1, int(total_words * top_percent))
    top_words = [word for word, count in word_counter.most_common() if word in word_set][:top_x_count]
    if len(top_words) > 35:
        return set(top_words[:35])
    else:
        return set(top_words)

def extract_keywords(tweet, top_n=5):
    keywords = kw_model.extract_keywords(
        tweet,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    return [keyword[0] for keyword in keywords]

def build_cooccurrence_dict(df_keywords):
    cooccurrence_dict = {}
    for keywords_list in df_keywords['keywords'].dropna():
        keywords_list = set(keywords_list)
        for pair in combinations(keywords_list, 2):
            pair = tuple(sorted(pair))
            cooccurrence_dict[pair] = cooccurrence_dict.get(pair, 0) + 1
    return cooccurrence_dict

def build_cooccurrence_matrix(cooccurrence_dict):
    unique_words = set()
    for pair in cooccurrence_dict:
        if isinstance(pair, tuple) and len(pair) == 2:
            unique_words.update(pair)
    unique_words = sorted(unique_words)
    cooccurrence_matrix = pd.DataFrame(0, index=unique_words, columns=unique_words)
    for pair, value in cooccurrence_dict.items():
        if isinstance(pair, tuple) and len(pair) == 2:
            word1, word2 = pair
            cooccurrence_matrix.loc[word1, word2] = value
            cooccurrence_matrix.loc[word2, word1] = value
    return cooccurrence_matrix

def get_specific_cooccurrence_matrix(matrix, word_set):
    words_in_matrix = [word for word in word_set if word in matrix.index]
    specific_matrix = matrix.loc[words_in_matrix, words_in_matrix]
    return specific_matrix

def remove_word(df, word):
    df['Tweet'] = df['Tweet'].apply(lambda x: x.lower().replace(word.lower(), ''))
    return df

def main():
    st.title("Twitter Tweet Analyzer")
    bearer_token = st.text_input("Enter your Twitter Bearer Token:", type="password")
    search_str = st.text_input("Enter first keyword to search for tweets:")
    search_str2 = st.text_input("Enter second keyword to search for tweets:")
    num_tweets = st.number_input(
        "Enter number of tweets to retrieve:",
        min_value=1, max_value=500, step=1, value=10
    )
    top_percent_input = st.number_input(
        "Enter the percentage for top items:",
        min_value=1, max_value=100, step=1, value=10
    )
    model_type = st.selectbox("Select Model", ['Linear SVC', 'Naive Bayes Multinomial'])
    encoding_dict = {0: 'A', 1: 'AD', 2: 'I', 3: 'T'}

    if st.button("Analyze Tweets"):
        if bearer_token and search_str and search_str2:
            tweets = retrieve_tweets(bearer_token, search_str, num_tweets)
            tweets2 = retrieve_tweets(bearer_token, search_str2, num_tweets)
            if tweets and tweets2:
                st.success("Tweets retrieved successfully!")
                translated_tweets, error_index = translate_tweets(tweets)
                translated_tweets2, error_index2 = translate_tweets(tweets2)
                cleaned_tweets = clean_tweets(translated_tweets)
                cleaned_tweets2 = clean_tweets(translated_tweets2)
                cleaned_tweets = [tweet for tweet in cleaned_tweets if tweet]
                cleaned_tweets2 = [tweet for tweet in cleaned_tweets2 if tweet]
                sentiment_scores = get_sentiments(cleaned_tweets)
                sentiment_scores2 = get_sentiments(cleaned_tweets2)
                empath_scores = [lexicon.analyze(tweet, normalize=False) for tweet in cleaned_tweets]
                empath_scores2 = [lexicon.analyze(tweet, normalize=False) for tweet in cleaned_tweets2]
                df = analyze_sentiment_empath(cleaned_tweets, sentiment_scores, empath_scores)
                df2 = analyze_sentiment_empath(cleaned_tweets2, sentiment_scores2, empath_scores2)
                top_percent = top_percent_input / 100
                top_emotions = get_emotion_totals(df, top_percent)
                top_emotions2 = get_emotion_totals(df2, top_percent)
                square_matrix = create_square_matrix(top_emotions, df)
                square_matrix2 = create_square_matrix(top_emotions2, df2)
                st.write(f"# Analysis Results for {search_str}:")
                st.write(df)
                st.write(top_emotions)
                st.write(square_matrix)
                st.markdown(f"## Emotion Graph for {search_str}")
                plot_graph(square_matrix, "Emotion Graph for " + search_str)
                st.write(f"# Analysis Results for {search_str2}:")
                st.write(df2)
                st.write(top_emotions2)
                st.write(square_matrix2)
                st.markdown(f"## Emotion Graph for {search_str2}")
                plot_graph(square_matrix2, "Emotion Graph for " + search_str2)

                if model_type in ['Linear SVC', 'Naive Bayes Multinomial']:
                    st.write(f"Model selected: {model_type}")
                    model_file = os.path.join(MODELS_DIR, model_type + '_trained_model.pkl')
                    vectorizer_file = os.path.join(MODELS_DIR, model_type + '_tfidf_vectorizer.pkl')

                    try:
                        model = joblib.load(model_file)
                        tfidf_vectorizer = joblib.load(vectorizer_file)
                        st.success("Model and TF-IDF vectorizer loaded successfully.")
                        X_retrieved_tfidf = tfidf_vectorizer.transform(df['Tweet'])
                        X_retrieved_tfidf2 = tfidf_vectorizer.transform(df2['Tweet'])
                        predictions = model.predict(X_retrieved_tfidf)
                        predictions2 = model.predict(X_retrieved_tfidf2)
                        predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])
                        predictions_df2 = pd.DataFrame(predictions2, columns=['intent_encoded'])
                        label_encoder_file = os.path.join(MODELS_DIR, 'label_encoder.pkl')
                        label_encoder = joblib.load(label_encoder_file)
                        intent_encoded = predictions_df['intent_encoded']
                        intent_encoded2 = predictions_df2['intent_encoded']
                        predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)
                        predictions_df2['intent'] = label_encoder.inverse_transform(intent_encoded2)
                        result_df = pd.concat([df, predictions_df], axis=1)
                        result_df2 = pd.concat([df2, predictions_df2], axis=1)
                        st.write("Results for ", search_str)
                        st.write(result_df)
                        st.write("Results for ", search_str2)
                        st.write(result_df2)
                    except Exception as e:
                        st.error(f"Error loading model: {e}")

                    intents = result_df['intent'].unique()
                    for intent in intents:
                        st.write(f"## Results for Intent: {intent}")
                        df_k = {}
                        df2_k = {}
                        df_intent = result_df[result_df['intent'] == intent]
                        df2_intent = result_df2[result_df2['intent'] == intent]
                        df_intent = remove_word(df_intent, search_str)
                        df2_intent = remove_word(df2_intent, search_str2)
                        df_k['keywords'] = df_intent['Tweet'].apply(yake_extract_keywords)
                        df2_k['keywords'] = df2_intent['Tweet'].apply(yake_extract_keywords)
                        keywords1 = set([keyword for keywords in df_k['keywords'] if isinstance(keywords, list) for keyword in keywords])
                        keywords2 = set([keyword for keywords in df2_k['keywords'] if isinstance(keywords, list) for keyword in keywords])
                        search_words1 = set(search_str.strip().lower().split())
                        search_words2 = set(search_str2.strip().lower().split())
                        keywords1 = keywords1 - search_words1
                        keywords2 = keywords2 - search_words2
                        common_keywords = keywords1 & keywords2
                        unique_keywords1 = keywords1 - common_keywords
                        unique_keywords2 = keywords2 - common_keywords
                        counter1 = Counter([keyword for keywords in df_k['keywords'] if isinstance(keywords, list) for keyword in keywords])
                        counter2 = Counter([keyword for keywords in df2_k['keywords'] if isinstance(keywords, list) for keyword in keywords])
                        top_keywords1 = filter_top_x_percent(unique_keywords1, counter1, top_percent)
                        top_keywords2 = filter_top_x_percent(unique_keywords2, counter2, top_percent)
                        cooccurrence_dict1 = build_cooccurrence_dict(pd.DataFrame(df_k))
                        cooccurrence_dict2 = build_cooccurrence_dict(pd.DataFrame(df2_k))
                        cooccurrence_matrix1 = build_cooccurrence_matrix(cooccurrence_dict1)
                        cooccurrence_matrix2 = build_cooccurrence_matrix(cooccurrence_dict2)
                        specific_cooccurrence_matrix1 = get_specific_cooccurrence_matrix(cooccurrence_matrix1, top_keywords1)
                        specific_cooccurrence_matrix2 = get_specific_cooccurrence_matrix(cooccurrence_matrix2, top_keywords2)
                        st.write(f'### Co-occurrence Matrix for {search_str}')
                        st.write(specific_cooccurrence_matrix1)
                        st.write(f'{search_str} Top {top_percent * 100}% Keywords Graph')
                        plot_graph(specific_cooccurrence_matrix1, f'{search_str} Top {top_percent * 100}% Keywords Graph')
                        st.write(f'### Co-occurrence Matrix for {search_str2}')
                        st.write(specific_cooccurrence_matrix2)
                        st.write(f'{search_str2} Top {top_percent * 100}% Keywords Graph')
                        plot_graph(specific_cooccurrence_matrix2, f'{search_str2} Top {top_percent * 100}% Keywords Graph')
                        specific_cooccurrence_matrix_common = get_specific_cooccurrence_matrix(cooccurrence_matrix1, common_keywords)
                        st.write(f'### Co-occurrence Matrix for Common Keywords')
                        st.write(specific_cooccurrence_matrix_common)
                        st.write(f"### Graph for Common Keywords")
                        plot_graph(specific_cooccurrence_matrix_common, "Graph for Common Keywords")
                        cols_exclude = ['intent', 'intent_encoded']
                        modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]
                        modified_df2 = df2_intent.loc[:, ~df2_intent.columns.isin(cols_exclude)]
                        top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                        top_emotions_intent2 = get_emotion_totals(modified_df2, top_percent)
                        st.write(top_emotions_intent)
                        st.write(top_emotions_intent2)
                        st.write('### Square matrix for ', intent)
                        intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                        intent_square_matrix2 = create_square_matrix(top_emotions_intent2, modified_df2)
                        st.write(intent_square_matrix)
                        st.write(intent_square_matrix2)
                        st.write("### Emotion Graph for :", intent)
                        plot_graph(intent_square_matrix, f'Emotion Graph for {search_str} - {intent}')
                        plot_graph(intent_square_matrix2, f'Emotion Graph for {search_str2} - {intent}')
                else:
                    st.write('\n')
                    st.write('No machine learning model selected.')
            else:
                st.error("Error occurred during tweet retrieval. Please check your input.")
        else:
            st.warning("Please enter all required information and click the 'Analyze Tweets' button.")

if __name__ == "__main__":
    main()
