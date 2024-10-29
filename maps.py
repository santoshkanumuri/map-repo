import time
from collections import Counter
import streamlit as st
import yake
import tweepy
# from googletrans import Translator  # Assuming you're using the googletrans library
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
from train_twitter import evaluate_saved_bert_model
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import requests
from google_trans_new import google_translator
from train_twitter import label_encode_df
from deep_translator import GoogleTranslator
import asyncio
import nltk
from nltk.corpus import stopwords
import spacy
import itertools
from rake_nltk import Rake

# Download stop words from nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

 # keybert model
# kw_model = KeyBERT('all-MiniLM-L6-v2')
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# kw_model = KeyBERT(model)
model = SentenceTransformer('paraphrase-mpnet-base-v2')
kw_model = KeyBERT(model=model)

global error_index
error_index = []
nlp = spacy.load('en_core_web_sm')


def rake_extract_keywords(tweet):
    # Function to extract keywords using Rake
    # Initialize Rake
    r = Rake()
    r.extract_keywords_from_text(tweet)
    return r.get_ranked_phrases()


def yake_extract_keywords(tweet):
    # Specify the parameters for the YAKE keyword extraction model
    language = "en"
    max_ngram_size = 4
    deduplication_threshold = 0.1
    num_of_keywords = len(tweet.split()) // 2
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=num_of_keywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(tweet)
    return [kw[0].lower() for kw in keywords]


def retrieve_tweets(bearer_token, search_str, num_tweets):
    # Function to retrieve tweets using the Twitter API
    search_str = search_str + " -is:retweet"
    search_str = search_str.replace(" ", "%20")
    keyword = search_str
    total_tweets = num_tweets
    max_results_per_request = 100
    tweets = []

    # Define the Twitter API endpoint for searching tweets with pagination
    endpoint = f"https://api.twitter.com/2/tweets/search/recent?query={keyword}&max_results={max_results_per_request}"

    # Set up the authorization header with the bearer token
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }

    # Function to fetch tweets
    def fetch_tweets(url):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    # Fetch tweets with pagination
    next_token = None
    while len(tweets) < total_tweets:
        if next_token:
            url = f"{endpoint}&next_token={next_token}"
        else:
            url = endpoint

        data = fetch_tweets(url)
        new_tweets = data.get('data', [])
        tweets.extend(new_tweets)

        # Break if there are no more tweets to fetch
        if 'next_token' not in data.get('meta', {}):
            break
        next_token = data['meta']['next_token']

        # Break if we have reached the total number of desired tweets
        if len(tweets) >= total_tweets:
            break

    # Trim the list to the desired number of tweets
    tweets = tweets[:total_tweets]
    cleaned_tweets = []
    for tweet in tweets:
        t = tweet['text']
        t = re.sub(r"(?:\@|https?\://|\#)\S+", "", t)
        print(t)
        cleaned_tweets.append(t)
    print(cleaned_tweets)
    return cleaned_tweets


def translate_tweets(tweets):
    # Function to translate tweets to English
    #translator = Translator()
    translator = google_translator()

    translated_tweets = []
    i = 0
    for tweet in tweets:
        try:
            i += 1
            #translate_text = translator.translate(tweet, lang_tgt='en')
            #translation = translator.translate(tweet, dest='en')
            #translated_tweets.append(translation.text)
            translate_text = GoogleTranslator(source='auto', target='en').translate(tweet[:4900])
            translated_tweets.append(translate_text)
        except AttributeError as e:
            print("attribute error" + str(tweet) + str(i))
            error_index.append(tweets.index(tweet))
        except TypeError as e:
            print("type error " + str(tweet) + str(i))
            error_index.append(tweets.index(tweet))

    return translated_tweets


def clean_tweets(tweets):
    # Function to clean tweets by removing URLs, hashtags, and mentions
    cleaned_tweets = []
    i = 0
    for tweet in tweets:
        try:
            i += 1
            cleaned_tweet = extract_text(tweet)
            cleaned_tweets.append(cleaned_tweet)
        except AttributeError as e:
            print("attribute error" + str(i))
            error_index.append(i)
        except TypeError as e:
            print("type error" + str(i))
            error_index.append(i)

    return cleaned_tweets


def extract_text(tweet):
    # Function to extract text from a tweet using regular expressions by removing URLs, hashtags, and mentions
    # Remove text before and including the @username
    tweet = re.sub(r'.*?@\w+:?', '', tweet)

    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet


def analyze_sentiment(text):
    # Function to analyze sentiment using VADER
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def get_sentiments(tweets):
    # Function to get sentiment scores for a list of tweets
    sentiment_scores = [analyze_sentiment(tweet) for tweet in tweets]
    return sentiment_scores


def analyze_sentiment_empath(tweets, sentiment_scores, empath_scores):
    # Function to analyze sentiment using Empath and VADER
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
    # Function to get emotion totals for the top percentage of emotions
    # Filter out non-emotion columns
    emotion_columns = df.drop(columns=['Tweet', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
                                       'vader_compound_rounded'])

    # Calculate column-wise sum for emotion columns
    column_wise_sum = emotion_columns.sum(axis=0)

    # Create a dictionary mapping emotion names to their total scores
    emotion_totals = dict(column_wise_sum)

    # Calculate the number of items corresponding to the top percentage
    top_count = int(len(emotion_totals) * top_percent)

    # Sort the dictionary items by values in descending order
    sorted_items = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)

    # Get the top key-value pairs
    top_percent_items = sorted_items[:top_count]

    # Create a new dictionary for the top percentage key-value pairs
    top_percent_dict = dict(top_percent_items)

    return top_percent_dict


def create_square_matrix(top_dict, df):
    # Function to create a square matrix
    # Create an empty square matrix represented as a DataFrame with named rows and columns
    column_names = list(top_dict.keys())
    square_matrix_df = pd.DataFrame(0, index=column_names, columns=column_names)

    # Iterate over the keys
    for key in column_names:
        # Count the occurrences of the key in the DataFrame
        count = df[key].sum()
        # Assign the count to the diagonal element in the square matrix
        square_matrix_df.loc[key, key] = count

    # Iterate over the combinations of keys
    for key1, key2 in combinations(column_names, 2):
        # Check if both keys are present in the dictionary
        if key1 in top_dict and key2 in top_dict:
            # Count the number of occurrences of the combination in the DataFrame
            count = ((df[key1] == 1) & (df[key2] == 1)).sum()
            # Assign the count to the corresponding entry in the square matrix
            square_matrix_df.loc[key1, key2] = count
            square_matrix_df.loc[key2, key1] = count  # Since it's an undirected graph, update both symmetric entries

    return square_matrix_df



def plot_graph(square_matrix,title):
    # Function to plot graph from square matrix using NetworkX and Matplotlib
    # Convert square matrix to DataFrame
    df = pd.DataFrame(square_matrix)

    # Get emotion names as node labels
    node_labels = df.index.tolist()

    # Convert square matrix to numpy array
    matrix = np.array(square_matrix)

    # Create a graph from the numpy array
    G = nx.from_numpy_array(matrix)

    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))

    # Use Kamada-Kawai layout for positioning the nodes
    pos = nx.kamada_kawai_layout(G)

    # Draw the graph with circular nodes and no outlines
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels={i: label for i, label in enumerate(node_labels)}, node_color='skyblue',
            node_size=1000, font_size=7, font_weight='bold', edge_color='#D3D3D3', width=1, node_shape='o',
            edgecolors='none')
    plt.title(title)
    plt.axis('off')  # Disable axis
    st.pyplot()


def read_from_file(excelfile):
    lines = []
    try:
        # Read the tweets from the Excel file
        tweetline_df = pd.read_excel(excelfile)
        # Get the 'Tweet' column as a list
        lines = tweetline_df['Tweet'].tolist()
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return lines


def evaluate_the_results(excelfile, predictions):
    # Convert the predictions to the actual intents
    pd_list = []
    actual = []
    accuracy = 0
    for pred in predictions:
        if pred == 0:
            pd_list.append('A')
        elif pred == 1:
            pd_list.append('AD')
        elif pred == 2:
            pd_list.append('I')
        elif pred == 3:
            pd_list.append('T')
    print(pd_list)
    try:
        # Read the actual intents from the Excel file
        tweetline_df = pd.read_excel(excelfile)
        actual = tweetline_df['actual'].tolist()
        print(actual)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    correct = 0
    # Check if the lengths of the actual and predicted lists are the same
    if len(actual) != len(pd_list):
        # If the lengths are different, remove the extra elements from the actual list
        print(len(actual), len(pd_list))
        for it in range(len(error_index)):
            actual.pop(error_index[it])
    try:
        # Calculate the accuracy of the predictions
        for i in range(len(actual)):
            if actual[i] == pd_list[i]:
                correct += 1
                print(actual[i], pd_list[i])
        accuracy = (correct / len(actual)) * 100
    except Exception as e:
        st.error(f"Error calculating accuracy: {e}")
    return accuracy


def preprocess(text):
    text=re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [word.strip('.,!?"') for word in words]  # Simple punctuation removal
    words = [word for word in words if word not in stop_words] # Remove stopwords
    return words


def create_graph_with_edges(words, counter, title):
    G = nx.Graph()

    # Add nodes and edges
    for word1 in words:
        for word2 in words:
            if word1 != word2:
                # Create an edge between two words if they co-occur in the tweets
                G.add_edge(word1, word2, weight=min(counter[word1], counter[word2]))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='#D3D3D3', linewidths=1,
            font_size=7, width=1, node_shape='o', edgecolors='none')
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='#D3D3D3')
    plt.title(title)
    plt.axis('off')
    st.pyplot()


def filter_top_x_percent(word_set, word_counter, top_percent):
    total_words = sum(word_counter.values())
    # Calculate how many words to include based on top_percent
    top_x_count = int(total_words * top_percent)

    # Sort by frequency and select top words
    top_words = [word for word, count in word_counter.most_common() if word in word_set][:top_x_count]

    # if length is greater than 35, return only top 35 words
    if len(top_words) > 35:
        return set(top_words[:35])
    else:
        return set(top_words)


def is_valid_token(token):
    # Function to check if the token is a valid word (ignores emojis and empty strings)
    # Filter out emojis, empty strings, and purely non-alphabetical tokens
    return token.is_alpha and len(token.text.strip()) > 0


def extract_keywords(tweet, top_n=5):
    """
    Extracts top N keywords from the given tweet using KeyBERT.

    Parameters:
    tweet (str): The tweet text.
    top_n (int): Number of top keywords to return (default is 5).

    Returns:
    list: A list of extracted keywords.
    """
    # Extract keywords using KeyBERT
    keywords = kw_model.extract_keywords(tweet,
                                         keyphrase_ngram_range=(1, 2),  # Capture both single and two-word phrases
                                         stop_words='english',  # Remove stop words
                                         top_n=top_n)  # Return top N keywords

    # Return only the keywords, excluding the confidence score
    return [keyword[0] for keyword in keywords]


def build_cooccurrence_dict(df):
    # Initialize co-occurrence dictionary
    cooccurrence_dict = {}

    # Iterate through each row's keyword list
    for keywords_list in df['keywords'].dropna():
        keywords_list = set(keywords_list)  # Remove duplicates by converting to set
        for pair in itertools.combinations(keywords_list, 2):
            pair = tuple(sorted(pair))  # Ensure consistent order for pairs
            cooccurrence_dict[pair] = cooccurrence_dict.get(pair, 0) + 1

    return cooccurrence_dict


def build_cooccurrence_matrix(cooccurrence_dict):
    # Step 1: Get the unique words from the dictionary keys
    unique_words = set()
    for pair in cooccurrence_dict:
        if isinstance(pair, tuple) and len(pair) == 2:
            unique_words.update(pair)
        else:
            print(f"Skipping invalid entry: {pair}")

    # Step 2: Create a DataFrame with rows and columns for the unique words
    unique_words = sorted(unique_words)  # Sort to keep the matrix ordered
    cooccurrence_matrix = pd.DataFrame(0, index=unique_words, columns=unique_words)

    # Step 3: Fill the matrix with values from the dictionary
    for pair, value in cooccurrence_dict.items():
        if isinstance(pair, tuple) and len(pair) == 2:
            word1, word2 = pair
            cooccurrence_matrix.loc[word1, word2] = value
            cooccurrence_matrix.loc[word2, word1] = value  # Since the matrix is symmetric
        else:
            print(f"Skipping invalid pair: {pair}")

    return cooccurrence_matrix


def get_specific_cooccurrence_matrix(matrix, word_set):
    # Ensure the selected words exist in the original matrix
    words_in_matrix = [word for word in word_set if word in matrix.index]

    # Filter the matrix to only include the rows and columns for the selected words
    specific_matrix = matrix.loc[words_in_matrix, words_in_matrix]

    return specific_matrix


def remove_word(df,word):
    # function removes particular word from a df['Tweet']
    df['Tweet'] = df['Tweet'].apply(lambda x: (x.lower()).replace(word.lower(), ''))
    return df

def main():
    global result_df2
    st.title("Twitter Tweet Analyzer")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Your Streamlit UI code goes here
    #bearer_token = st.text_input("Enter your Twitter Bearer Token:", type="password")
    bearer_token=os.getenv('TWITTER_API_KEY')
    search_str = st.text_input("Enter first keyword to search for tweets:")
    search_str2= st.text_input("Enter second keyword to search for tweets:")
    num_tweets = st.number_input("Enter number of tweets to retrieve:", min_value=1, max_value=500, step=1,
                                 value=10)
    top_percent_input = st.number_input("Enter the percentage for top items:", min_value=1, max_value=100, step=1,
                                        value=10)

    model_type = st.selectbox("Select Model", ['Linear SVC', 'Naive Bayes Multinomial'])
    encoding_dict = {0: 'A', 1: 'AD', 2: 'I', 3: 'T'}

    if st.button("Analyze Tweets"):
        if search_str and search_str2:
            # Retrieve tweets
            tweets = retrieve_tweets(bearer_token, search_str, num_tweets)
            tweets2 = retrieve_tweets(bearer_token, search_str2, num_tweets)

            if tweets is not None and tweets2 is not None:
                st.success("Tweets retrieved successfully!")

                # Translate tweets
                translated_tweets = translate_tweets(tweets)
                translated_tweets2 = translate_tweets(tweets2)

                # Clean tweets
                cleaned_tweets = clean_tweets(translated_tweets)
                cleaned_tweets2 = clean_tweets(translated_tweets2)

                # Remove empty strings
                cleaned_tweets = [tweet for tweet in cleaned_tweets if tweet]
                cleaned_tweets2 = [tweet for tweet in cleaned_tweets2 if tweet]

                # Analyze sentiment
                sentiment_scores = get_sentiments(cleaned_tweets)
                sentiment_scores2 = get_sentiments(cleaned_tweets2)

                # Initialize VADER sentiment analyzer
                analyzer = SentimentIntensityAnalyzer()

                # Analyze with Empath
                lexicon = Empath()

                empath_scores = [lexicon.analyze(tweet, normalize=False) for tweet in cleaned_tweets]
                empath_scores2 = [lexicon.analyze(tweet, normalize=False) for tweet in cleaned_tweets2]

                # Convert cleaned tweets, sentiment scores, and empath scores to DataFrame
                df = analyze_sentiment_empath(cleaned_tweets, sentiment_scores, empath_scores)
                df2 = analyze_sentiment_empath(cleaned_tweets2, sentiment_scores2, empath_scores2)


                # Get emotion totals
                top_percent = top_percent_input / 100

                # Retrieve top emotions
                top_emotions = get_emotion_totals(df, top_percent)
                top_emotions2 = get_emotion_totals(df2, top_percent)

                square_matrix = create_square_matrix(top_emotions, df)
                square_matrix2 = create_square_matrix(top_emotions2, df2)

                # Display DataFrame
                st.write(f"# Analysis Results for {search_str}:")
                st.write(df)
                st.write(top_emotions)
                st.write(square_matrix)
                # Plot the graph
                st.markdown(f"## Emotion Graph for {search_str}")
                plot_graph(square_matrix,"Emotion Graph for "+search_str)

                st.write(f"# Analysis Results for {search_str2}:")
                st.write(df2)
                st.write(top_emotions2)
                st.write(square_matrix2)
                # Plot the graph
                st.markdown(f"## Emotion Graph for {search_str2}")
                plot_graph(square_matrix2,"Emotion Graph for "+search_str2)


                #model_type = st.selectbox("Select Model", ['Linear SVC', 'Naive Bayes Multinomial'])

                if model_type == 'Linear SVC' or model_type == 'Naive Bayes Multinomial':
                    st.write(f"Model selected: {model_type}")
                    model_file = "./models/"+model_type + 'trained_model.pkl'
                    vectorizer_file = "./models/"+model_type + 'tfidf_vectorizer.pkl'


                    try:
                        model = joblib.load(model_file)
                        tfidf_vectorizer = joblib.load(vectorizer_file)

                        st.success("Model and TF-IDF vectorizer loaded successfully.")

                        # Convert the cleaned retrieved tweets to TF-IDF features
                        X_retrieved_tfidf = tfidf_vectorizer.transform(df['Tweet'])
                        X_retrieved_tfidf2 = tfidf_vectorizer.transform(df2['Tweet'])

                        # Predict intents using the loaded model
                        predictions = model.predict(X_retrieved_tfidf)
                        predictions2 = model.predict(X_retrieved_tfidf2)

                        # Convert predictions to DataFrame
                        predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])
                        predictions_df2 = pd.DataFrame(predictions2, columns=['intent_encoded'])

                        label_encoder = joblib.load('./models/label_encoder.pkl')

                        intent_encoded = predictions_df['intent_encoded']
                        intent_encoded2 = predictions_df2['intent_encoded']

                        predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)
                        predictions_df2['intent'] = label_encoder.inverse_transform(intent_encoded2)

                        mapping_values = dict(
                            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                        # Concatenate 'tweet' column with predictions DataFrame
                        result_df = pd.concat([df, predictions_df], axis=1)
                        result_df2 = pd.concat([df2, predictions_df2], axis=1)

                        # Display result
                        st.write("Results for ", search_str)
                        st.write(result_df)
                        st.write("Results for ", search_str2)
                        st.write(result_df2)


                    except Exception as e:
                        st.error(f"Error loading model: {e}")

                    intents = result_df['intent'].unique()

                    for intent in intents:
                        st.write(f"## Results for  Intent: {intent}")
                        df_k = {}
                        df2_k = {}

                        # Filter the dataframe for the current intent
                        df_intent = result_df[result_df['intent'] == intent]
                        df2_intent = result_df2[result_df2['intent'] == intent]

                        df_intent=remove_word(df_intent,search_str)
                        df2_intent=remove_word(df2_intent,search_str2)

                        st.write(df_intent)
                        st.write(df2_intent)

                        # df_k['keywords'] = df_intent['Tweet'].apply(extract_keywords)
                        # df2_k['keywords'] = df2_intent['Tweet'].apply(extract_keywords)

                        df_k['keywords'] = df_intent['Tweet'].apply(yake_extract_keywords)
                        df2_k['keywords'] = df2_intent['Tweet'].apply(yake_extract_keywords)

                        # df_k['keywords'] = df_intent['Tweet'].apply(rake_extract_keywords)
                        # df2_k['keywords'] = df2_intent['Tweet'].apply(rake_extract_keywords)

                        print(df_k['keywords'])
                        print(df2_k['keywords'])



                        # Flatten the keyword lists to sets
                        keywords1 = set([keyword for keywords in df_k['keywords'] if isinstance(keywords, list) for keyword in keywords])
                        keywords2 = set([keyword for keywords in df2_k['keywords'] if isinstance(keywords, list) for keyword in keywords])

                        li=list(search_str.strip().lower())
                        li2=list(search_str2.strip().lower())
                        #remove all words from the keywords that are in the search string and any string have the words of search string
                        keywords1 = [x for x in keywords1 if x not in li]
                        keywords2 = [x for x in keywords2 if x not in li2]
                        keywords1=set(keywords1)
                        keywords2=set(keywords2)



                        # Find common and unique keywords
                        common_keywords = keywords1 & keywords2
                        unique_keywords1 = keywords1 - common_keywords
                        unique_keywords2 = keywords2 - common_keywords


                        # Create a counter for the keywords
                        counter1 = Counter([keyword for keywords in df_k['keywords'] if isinstance(keywords, list) for keyword in keywords])
                        counter2 = Counter([keyword for keywords in df2_k['keywords'] if isinstance(keywords, list) for keyword in keywords])



                        # Filter the top x% of keywords
                        top_keywords1 = filter_top_x_percent(unique_keywords1, counter1, top_percent)
                        top_keywords2 = filter_top_x_percent(unique_keywords2, counter2, top_percent)

                        # # Create a graph with edges
                        # create_graph_with_edges(top_keywords1, counter1, f'{search_str} Top {top_percent}% Keywords Graph')
                        # create_graph_with_edges(top_keywords2, counter2, f'{search_str2} Top {top_percent}% Keywords Graph')
                        # create_graph_with_edges(common_keywords, counter1, 'Common Keywords Graph')

                        # Define a function to generate co-occurrence counts
                        cooccurrence_dict1 = build_cooccurrence_dict(df_k)
                        cooccurrence_dict2 = build_cooccurrence_dict(df2_k)

                        # print(cooccurrence_dict1)
                        # print(cooccurrence_dict2)

                        # Build the co-occurrence matrix
                        cooccurrence_matrix1 = build_cooccurrence_matrix(cooccurrence_dict1)
                        cooccurrence_matrix2 = build_cooccurrence_matrix(cooccurrence_dict2)

                        # st.write(f'## Co-occurrence Matrix for {search_str}')
                        # st.write(cooccurrence_matrix1)
                        # st.write(f'## Co-occurrence Matrix for {search_str2}')
                        # st.write(cooccurrence_matrix2)


                        # Get the specific co-occurrence matrix for the top keywords
                        specific_cooccurrence_matrix1 = get_specific_cooccurrence_matrix(cooccurrence_matrix1, top_keywords1)
                        specific_cooccurrence_matrix2 = get_specific_cooccurrence_matrix(cooccurrence_matrix2, top_keywords2)

                        # draw the graph for the specific co-occurrence matrix
                        st.write(f'### Co-occurrence Matrix for {search_str}')
                        st.write(specific_cooccurrence_matrix1)
                        st.write(f'{search_str} Top {top_percent*100}% Keywords Graph')
                        plot_graph(specific_cooccurrence_matrix1,f'{search_str} Top {top_percent*100}% Keywords Graph')

                        st.write(f'### Co-occurrence Matrix for {search_str2}')
                        st.write(specific_cooccurrence_matrix2)
                        st.write(f'{search_str2} Top {top_percent*100}% Keywords Graph')
                        plot_graph(specific_cooccurrence_matrix2,f'{search_str2} Top {top_percent*100}% Keywords Graph')

                        # Get the specific co-occurrence matrix for the common keywords
                        specific_cooccurrence_matrix_common = get_specific_cooccurrence_matrix(cooccurrence_matrix1, common_keywords)
                        st.write(f'### Co-occurrence Matrix for Common Keywords')
                        st.write(specific_cooccurrence_matrix_common)
                        st.write(f"### Graph for Common Keywords")
                        plot_graph(specific_cooccurrence_matrix_common,"Graph for Common Keywords")

                        cols_exclude = ['intent', 'intent_encoded']

                        modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]
                        modified_df2 = df2_intent.loc[:, ~df2_intent.columns.isin(cols_exclude)]

                        # Retrieve top emotions
                        top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                        top_emotions_intent2 = get_emotion_totals(modified_df2, top_percent)
                        st.write(top_emotions_intent)
                        st.write(top_emotions_intent2)

                        st.write('### Square matrix for ', intent)
                        # Generate the square matrix for the current intent
                        intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                        intent_square_matrix2 = create_square_matrix(top_emotions_intent2, modified_df2)
                        st.write(intent_square_matrix)
                        st.write(intent_square_matrix2)

                        # Plot the graph
                        st.write("### Emotion Graph for :", intent)
                        #
                        # # Plot the graph for the current intent

                        plot_graph(intent_square_matrix, f'Emotion Graph for {search_str} - {intent}')
                        plot_graph(intent_square_matrix2, f'Emotion Graph for {search_str2} - {intent}')
                else:
                    st.write('\n')
                    st.write('No machine learning model selected.')

                # if bert_model_type == 'bert-base-uncased' or bert_model_type == 'bert-large-uncased':
                #     st.write(f"Model selected: {bert_model_type}")
                #     bert_model_file = "./models/"+bert_model_type + 'trained_bert_model.pth'
                #     tokenizer = BertTokenizer.from_pretrained(bert_model_type)
                #
                #     # Check if CUDA is available
                #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #     # Get the current working directory
                #     current_directory = os.getcwd()
                #
                #     try:
                #         # Load the BERT model
                #         saved_model_path = os.path.join(current_directory, bert_model_file)
                #         bert_model = BertForSequenceClassification.from_pretrained(bert_model_type, num_labels=4)
                #         bert_model.load_state_dict(torch.load(saved_model_path, map_location=device))
                #         bert_model.eval()
                #         st.write("BERT model loaded successfully.")
                #         # Preprocess test data and create DataLoader
                #         test_encodings = tokenizer(df['Tweet'].tolist(), truncation=True, padding=True,
                #                                    return_tensors='pt')
                #         test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
                #         test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
                #
                #         # Evaluate the model
                #         predictions = []
                #
                #         with torch.no_grad():
                #             for batch in test_loader:
                #                 input_ids, attention_mask = batch
                #                 input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                #                 outputs = bert_model(input_ids, attention_mask=attention_mask)
                #                 logits = outputs.logits
                #                 predictions.extend(logits.argmax(dim=1).cpu().numpy())
                #
                #         st.write('Got predictions')
                #
                #         # Convert predictions to DataFrame
                #         predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])
                #
                #         st.write(predictions_df)
                #
                #         label_encoder = joblib.load('./models/label_encoder.pkl')
                #
                #         intent_encoded = predictions_df['intent_encoded']
                #
                #         predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)
                #
                #         mapping_values = dict(
                #             zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                #
                #         st.write(mapping_values)
                #
                #         # Concatenate 'tweet' column with predictions DataFrame
                #         result_df = pd.concat([df, predictions_df], axis=1)
                #
                #         # Display result
                #         st.write(result_df)
                #
                #     except Exception as e:
                #         st.error(f"Error loading BERT model: {e}")
                #
                #     intents = result_df['intent'].unique()
                #
                #     for intent in intents:
                #         st.write(intent)
                #
                #         # Filter the dataframe for the current intent
                #         df_intent = result_df[result_df['intent'] == intent]
                #         st.write(df_intent)
                #
                #         cols_exclude = ['intent', 'intent_encoded']
                #
                #         modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]
                #
                #         # Retrieve top emotions
                #         top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                #         st.write(top_emotions_intent)
                #
                #         st.write('Square matrix for ', intent)
                #         # Generate the square matrix for the current intent
                #         intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                #         st.write(intent_square_matrix)
                #
                #         # Plot the graph
                #         st.write("### Emotion Graph for :", intent)
                #         #
                #         # # Plot the graph for the current intent
                #
                #         plot_graph(intent_square_matrix)
        else:
            st.error("Error occurred during tweet retrieval. Please check your input.")
    else:
        st.warning("Please enter the required information and click the 'Analyze Tweets' button.")


if __name__ == "__main__":
    main()
