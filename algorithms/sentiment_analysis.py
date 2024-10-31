import logging
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List
from ..database.db_connection import DatabaseConnection

class SentimentAnalyzer:
    """
    Analyzes market sentiment by aggregating data from multiple sources, including news and social media.

    Attributes:
    - db (DatabaseConnection): Database for logging sentiment data.
    - sources (list): List of sentiment data sources.
    - analyzer (SentimentIntensityAnalyzer): Analyzer for calculating sentiment scores.
    """

    def __init__(self, db: DatabaseConnection, sources: List[str] = None):
        self.db = db
        self.sources = sources or [
            "https://api.twitter.com/...",  # Placeholder for Twitter API
            "https://newsapi.org/..."       # Placeholder for News API
        ]
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)

    def fetch_data(self) -> Dict[str, List[str]]:
        """
        Fetches sentiment data from the configured sources.

        Returns:
        - dict: Collected sentiment data by source.
        """
        sentiment_data = {}
        headers = {"Authorization": "Bearer YOUR_API_KEY"}  # Placeholder for API key

        for source in self.sources:
            try:
                response = requests.get(source, headers=headers)
                if response.status_code == 200:
                    sentiment_data[source] = response.json()["content"]
                    self.logger.info(f"Fetched data from {source}")
                else:
                    self.logger.warning(f"Failed to fetch data from {source}, Status: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error fetching data from {source}: {e}")

        return sentiment_data

    def analyze_sentiment(self, text_data: List[str]) -> float:
        """
        Analyzes and calculates an average sentiment score.

        Args:
        - text_data (list): Text entries to analyze.

        Returns:
        - float: Average sentiment score.
        """
        scores = [self.analyzer.polarity_scores(text)["compound"] for text in text_data]
        average_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.info(f"Calculated average sentiment score: {average_score}")
        return average_score

    def log_sentiment(self, source: str, score: float):
        """
        Logs sentiment score for a particular source in the database.

        Args:
        - source (str): Source of the sentiment data.
        - score (float): Sentiment score.
        """
        self.db.log_sentiment(source, score)
        self.logger.info(f"Logged sentiment score from {source}: {score}")

    def update_sentiments(self):
        """
        Aggregates and logs sentiment data from all sources.
        """
        sentiment_data = self.fetch_data()
        for source, data in sentiment_data.items():
            score = self.analyze_sentiment(data)
            self.log_sentiment(source, score)

        self.logger.info("Updated sentiment scores from all sources.")
