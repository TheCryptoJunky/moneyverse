import logging
import requests
from typing import Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..database.db_connection import DatabaseConnection

class SentimentAnalyzer:
    """
    Analyzes market sentiment from various sources and integrates real-time sentiment scoring.
    
    Attributes:
    - db (DatabaseConnection): Database connection for logging sentiment scores.
    - sources (list): List of sentiment data sources (e.g., news sites, social media).
    - analyzer (SentimentIntensityAnalyzer): Sentiment analysis tool.
    """

    def __init__(self, db: DatabaseConnection, sources: List[str] = None):
        self.db = db
        self.sources = sources or [
            "https://api.twitter.com/...",  # Placeholder for Twitter API
            "https://newsapi.org/..."       # Placeholder for News API
        ]
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)

    def fetch_sentiment_data(self) -> Dict[str, List[str]]:
        """
        Fetches real-time sentiment data from configured sources.

        Returns:
        - dict: Dictionary of sentiment data by source.
        """
        sentiment_data = {}
        headers = {"Authorization": "Bearer YOUR_API_KEY"}  # Replace with actual token

        for source in self.sources:
            try:
                response = requests.get(source, headers=headers)
                if response.status_code == 200:
                    sentiment_data[source] = response.json()["content"]
                    self.logger.info(f"Fetched sentiment data from {source}.")
                else:
                    self.logger.warning(f"Failed to fetch data from {source}. Status code: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error fetching sentiment data from {source}: {e}")

        return sentiment_data

    def analyze_sentiment(self, text_data: List[str]) -> float:
        """
        Analyzes and aggregates sentiment scores for a list of text data.

        Args:
        - text_data (list): List of text entries to analyze.

        Returns:
        - float: Average sentiment score.
        """
        scores = [self.analyzer.polarity_scores(text)["compound"] for text in text_data]
        average_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.info(f"Calculated average sentiment score: {average_score}")
        return average_score

    def log_sentiment_score(self, source: str, score: float):
        """
        Logs sentiment scores into the database for each source.

        Args:
        - source (str): The source of the sentiment data.
        - score (float): Calculated sentiment score.
        """
        self.db.log_sentiment(source, score)
        self.logger.info(f"Logged sentiment score for {source}: {score}")

    def update_sentiment_scores(self):
        """
        Fetches sentiment data, calculates sentiment scores, and logs results for each source.
        """
        sentiment_data = self.fetch_sentiment_data()

        for source, data in sentiment_data.items():
            score = self.analyze_sentiment(data)
            self.log_sentiment_score(source, score)

        self.logger.info("Updated sentiment scores from all sources.")
