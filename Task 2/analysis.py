import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import networkx as nx
from scipy.stats import ttest_rel
from typing import List, Union


class Analysis:
    def __init__(self, initial_df: pd.DataFrame, final_df: pd.DataFrame):
        self.plot_urls = []
        # call all the analysis functions
        # drop questions and answers that are na
        initial_df = initial_df.dropna(subset=["question", "answer"])
        final_df = final_df.dropna(subset=["question", "answer"])
        self.word_frequency_comparison(initial_df=initial_df, final_df=final_df)
        self.answer_length_change(initial_df=initial_df, final_df=final_df)
        self.answer_sentiment_analysis(initial_df=initial_df, final_df=final_df)
        self.text_complexity_analysis(initial_df=initial_df, final_df=final_df)
        self.question_similarity_network(initial_df=initial_df, final_df=final_df)

    def preprocess_text(self, text: Union[str, List[str]]) -> List[str]:
        """Preprocesses the text by tokenizing, removing stopwords and lemmatizing the words

        Args:
            text (Union[str, List[str]]):Text to be pre-preprocessed

        Returns:
            List[str]: Returns the processed text
        """

        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]

    def append_to_image_plots(self, plt: plt):
        """Appends the plot to the list of plot_urls

        Args:
            plt (plt): Plot to be appended to the list of plot_urls
        """
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode("utf8")
        self.plot_urls.append(plot_url)

    def get_sentiment(self, text:str)->float:
        """Get the sentiment of the text

        Args:
            text (str): Text for which sentiment is to be calculated
        Returns:
            float:Returns sentiment in float value
        """
        return TextBlob(text).sentiment.polarity

    def text_complexity(self, text:str)->float:
        """Get the text complexity based on the length of the words

        Args:
            text (str): Text to calculate the complexity

        Returns:
            float: Returns the text complexity float value
        """
        words = text.split()
        return sum(len(word) for word in words) / len(words)
    
    def word_frequency_comparison(
        self, initial_df: pd.DataFrame, final_df: pd.DataFrame
    ):
        """Compares the word frequence in question and answer of both versions of the questionnaire

        Args:
            initial_df (pd.DataFrame): Initial version of the questionnaire
            final_df (pd.DataFrame): Final version of the questionnaire
        """
        # Combine all text from both versions
        initial_text = " ".join(initial_df["question"] + " " + initial_df["answer"])
        final_text = " ".join(final_df["question"] + " " + final_df["answer"])
        # Preprocess the text
        initial_words = self.preprocess_text(text=initial_text)
        final_words = self.preprocess_text(text=final_text)

        # Get word frequencies
        initial_freq = Counter(initial_words)
        final_freq = Counter(final_words)
        # Get the plot
        plt.figure(figsize=(12, 6))
        plt.bar(
            np.arange(10) * 2,
            [initial_freq.most_common(10)[i][1] for i in range(10)],
            align="center",
            alpha=0.5,
            label="Initial",
        )
        plt.bar(
            np.arange(10) * 2 + 1,
            [final_freq.most_common(10)[i][1] for i in range(10)],
            align="center",
            alpha=0.5,
            label="Final",
        )
        plt.xticks(
            np.arange(10) * 2,
            [initial_freq.most_common(10)[i][0] for i in range(10)],
            rotation=45,
        )
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title("Top 10 Word Frequencies: Initial vs Final")
        plt.legend()
        plt.tight_layout()
        self.append_to_image_plots(plt)

    def answer_length_change(self, initial_df: pd.DataFrame, final_df: pd.DataFrame):
        """Compares the answer length change in both versions of the questionnaire
        Args:
            initial_df (pd.DataFrame): Initial version of the questionnaire
            final_df (pd.DataFrame): Final version of the questionnaire
        """
        initial_df["answer_length"] = initial_df["answer"].str.len()
        final_df["answer_length"] = final_df["answer"].str.len()
        length_diff = final_df["answer_length"] - initial_df["answer_length"]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(length_diff)), length_diff)
        plt.xlabel("Question Index")
        plt.ylabel("Change in Answer Length")
        plt.title("Changes in Answer Length: Final vs Initial")
        self.append_to_image_plots(plt)
    
    def answer_sentiment_analysis(self, initial_df: pd.DataFrame, final_df: pd.DataFrame):
        """Compares the answer sentiment in both versions of the questionnaire
        Args:
            initial_df (pd.DataFrame): Initial version of the questionnaire
            final_df (pd.DataFrame): Final version of the questionnaire
        """
        initial_df['sentiment'] = initial_df['answer'].apply(self.get_sentiment)
        final_df['sentiment'] = final_df['answer'].apply(self.get_sentiment)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=np.arange(len(initial_df))*2, y='sentiment', data=initial_df, label='Initial')
        sns.scatterplot(x=np.arange(len(final_df))*2, y='sentiment', data=final_df, label='Final', alpha=0.5)
        plt.xlabel('Question Index')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Analysis: Initial vs Final')
        plt.legend()
        self.append_to_image_plots(plt)
        
    def text_complexity_analysis(self, initial_df: pd.DataFrame, final_df: pd.DataFrame):
        """Compares the text complexity analysis in both versions of the questionnaire
        Args:
            initial_df (pd.DataFrame): Initial version of the questionnaire
            final_df (pd.DataFrame): Final version of the questionnaire
        """
        initial_df['complexity'] = initial_df['answer'].apply(self.text_complexity)
        final_df['complexity'] = final_df['answer'].apply(self.text_complexity)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(initial_df)), initial_df['complexity'],  label='Initial')
        plt.bar(range(len(final_df)), final_df['complexity'], alpha=0.5, label='Final')
        plt.xlabel('Question Index')
        plt.ylabel('Average Word Length')
        plt.title('Text Complexity: Initial vs Final')
        plt.legend()
        self.append_to_image_plots(plt)
    
    def question_similarity_network(self, initial_df: pd.DataFrame, final_df: pd.DataFrame):
        """Creates a question similarity network based on the cosine similarity of the questions
        Args:
            initial_df (pd.DataFrame): Initial version of the questionnaire
            final_df (pd.DataFrame): Final version of the questionnaire
        """
        tfidf_vectorizer = TfidfVectorizer()
        all_questions = pd.concat([initial_df['question'], final_df['question']])
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_questions)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        G = nx.Graph()
        for i in range(len(initial_df)):
            for j in range(len(initial_df), len(initial_df) + len(final_df)):
                if similarity_matrix[i, j] > 0.5:  # Adjust threshold as needed
                    G.add_edge(f'I{i+1}', f'F{j-len(initial_df)+1}', weight=similarity_matrix[i, j])

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
        plt.title('Question Similarity Network')
        plt.legend()
        self.append_to_image_plots(plt)
