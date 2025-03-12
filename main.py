import streamlit as st
import tweepy
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from astrapy.db import AstraDB
from collections import Counter
import random
import streamlit.components.v1 as components


# Load environment variables
load_dotenv()

print("Bearer Token:", os.getenv("TWITTER_BEARER_TOKEN"))
print("API Key:", os.getenv("TWITTER_API_KEY"))

class AstraDBManager:
    def __init__(self):
        self.db = AstraDB(
            token=os.getenv('ASTRA_TOKEN'),
            api_endpoint=os.getenv('ASTRA_API_ENDPOINT'),
            namespace=os.getenv('ASTRA_KEYSPACE', 'twitter_analysis')
        )
        self.tweets_collection = self.db.collection('tweets')
        try:
            collections = self.db.get_collections()
            if 'tweets' not in collections:
                self.db.create_collection('tweets')
                print("Created 'tweets' collection")
        except Exception as e:
            print(f"Error connecting to Astra DB: {e}")

    def save_tweet(self, tweet_data):
        try:
            self.tweets_collection.insert_one(tweet_data)
            return True
        except Exception as e:
            print(f"Error saving tweet: {e}")
            return False

    def get_flagged_tweets(self):
        try:
            cursor = self.tweets_collection.find({"sentiment_category": "negative"})
            return pd.DataFrame(cursor)
        except Exception as e:
            print(f"Error retrieving flagged tweets: {e}")
            return pd.DataFrame()
    
    def get_tweet_history(self, limit=100):
        try:
            cursor = self.tweets_collection.find({}, limit=limit)
            return pd.DataFrame(cursor)
        except Exception as e:
            print(f"Error retrieving tweet history: {e}")
            return pd.DataFrame()

class TwitterAnalyzer:
    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        self.db_manager = AstraDBManager()
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def fetch_tweets(self, query, limit=100):
        tweets = []
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                expansions=['author_id'],
                user_fields=['username', 'profile_image_url']
            )
            
            # Create user lookup dictionary
            users = {user.id: user for user in response.includes.get('users', [])}
            
            if response.data:
                for tweet in response.data:
                    sentiment = self.analyze_sentiment(tweet.text)
                    user = users.get(tweet.author_id)
                    username = user.username if user else "unknown"
                    profile_image = user.profile_image_url if user and hasattr(user, 'profile_image_url') else ""
                    
                    tweet_data = {
                        'id': str(tweet.id),
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat(),
                        'author_id': str(tweet.author_id),
                        'username': username,
                        'profile_image': profile_image,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'sentiment_score': sentiment['compound'],
                        'sentiment_category': self.get_sentiment_category(sentiment['compound']),
                        'analysis_timestamp': datetime.utcnow().isoformat()
                    }
                    tweets.append(tweet_data)
                    self.db_manager.save_tweet(tweet_data)
        except Exception as e:
            st.error(f"Error fetching tweets: {e}")
        
        return pd.DataFrame(tweets)

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)

    def get_sentiment_category(self, compound_score):
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def analyze_content_issues(self, text):
        text_lower = text.lower()
        issues = []
        toxic_keywords = ['hate', 'awful', 'terrible', 'worst', 'stupid', 'sucks', 'bad', 'useless']
        misinfo_keywords = ['fake', 'hoax', 'conspiracy', 'false', 'scam', 'misleading']
        critical_keywords = ['urgent', 'emergency', 'crisis', 'failure', 'broken', 'disaster']

        if any(word in text_lower for word in toxic_keywords):
            issues.append('toxicity')
        if any(word in text_lower for word in misinfo_keywords):
            issues.append('misinformation')
        if any(word in text_lower for word in critical_keywords):
            issues.append('critical_issues')

        return issues if issues else ['general_negative']

    def generate_damage_control_suggestions(self, tweet):
        issues = self.analyze_content_issues(tweet['text'])
        suggestions = {
            'toxicity': [
                "Respond professionally and avoid defensive language",
                "Report if the tweet violates platform guidelines",
                "Engage constructively by addressing the underlying concern"
            ],
            'misinformation': [
                "Share verified sources and correct information",
                "Request third-party fact-checking",
                "Prepare an informative thread with accurate details"
            ],
            'critical_issues': [
                "Escalate to appropriate team members",
                "Draft an official response with action steps",
                "Monitor related hashtags for broader impact"
            ],
            'general_negative': [
                "Acknowledge the customer's concerns",
                "Provide helpful information or resources",
                "Offer direct message support for detailed resolution"
            ]
        }
        return [s for issue in issues for s in suggestions.get(issue, [])]

class AdGenerator:
    def __init__(self):
        self.positive_templates = [
            "Experience what everyone's talking about! {benefit}",
            "Join the conversation! {benefit}",
            "Discover why {benefit} matters to you",
            "The smart choice for {benefit}",
            "Transform your experience with {benefit}"
        ]
        self.response_templates = [
            "We hear you! That's why we {solution}",
            "Your concerns matter. See how we {solution}",
            "We're improving! Now with better {solution}",
            "New and improved: We've enhanced our {solution}",
            "You spoke, we listened. Introducing our {solution}"
        ]

    def analyze_trending_topics(self, tweets_df):
        if tweets_df.empty:
            return []
        
        all_text = ' '.join(tweets_df['text'].astype(str))
        words = [word.lower() for word in all_text.split() if len(word) > 3]
        word_freq = Counter(words)
        return [word for word, count in word_freq.most_common(5)]

    def extract_key_benefits(self, positive_tweets_df):
        positive_words = []
        for text in positive_tweets_df['text']:
            words = str(text).lower().split()
            for i, word in enumerate(words):
                if word in {'great', 'amazing', 'excellent', 'good', 'best', 'love', 'perfect'}:
                    if i + 1 < len(words):
                        positive_words.append(words[i + 1])
        return Counter(positive_words).most_common(5)

    def identify_pain_points(self, negative_tweets_df):
        pain_points = []
        for text in negative_tweets_df['text']:
            words = str(text).lower().split()
            for i, word in enumerate(words):
                if word in {'bad', 'poor', 'terrible', 'worst', 'slow', 'expensive', 'difficult'}:
                    if i + 1 < len(words):
                        pain_points.append(words[i + 1])
        return Counter(pain_points).most_common(5)

    def generate_ad_ideas(self, tweets_df):
        if tweets_df.empty:
            return []
            
        positive_tweets = tweets_df[tweets_df['sentiment_score'] > 0.2]
        negative_tweets = tweets_df[tweets_df['sentiment_score'] < -0.2]
        
        trending = self.analyze_trending_topics(tweets_df)
        benefits = self.extract_key_benefits(positive_tweets)
        pain_points = self.identify_pain_points(negative_tweets)
        
        ad_ideas = []
        
        for benefit, _ in benefits:
            template = random.choice(self.positive_templates)
            ad_ideas.append({
                "type": "positive",
                "text": template.format(benefit=benefit),
                "keywords": [benefit]
            })
            
        for pain_point, _ in pain_points:
            template = random.choice(self.response_templates)
            solution = f"approach to {pain_point}"
            ad_ideas.append({
                "type": "response",
                "text": template.format(solution=solution),
                "keywords": [pain_point]
            })
            
        return ad_ideas

def get_sentiment_color(sentiment_score):
    if sentiment_score >= 0.05:
        # Green for positive
        return f"rgba(0, 128, 0, {min(abs(sentiment_score), 1.0)})"
    elif sentiment_score <= -0.05:
        # Red for negative
        return f"rgba(255, 0, 0, {min(abs(sentiment_score), 1.0)})"
    else:
        # Gray for neutral
        return "rgba(128, 128, 128, 0.6)"

def create_custom_css():
    return """
    <style>
        .dashboard-container {
            font-family: 'Roboto', sans-serif;
        }
        
        .metrics-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            width: 23%;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        
        .tweet-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .tweet-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .tweet-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .tweet-username {
            font-weight: bold;
        }
        
        .tweet-content {
            margin-bottom: 10px;
        }
        
        .tweet-metrics {
            display: flex;
            color: #666;
            font-size: 14px;
        }
        
        .tweet-metric {
            margin-right: 15px;
        }
        
        .sentiment-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            color: white;
            display: inline-block;
            margin-left: 10px;
        }
        
        .positive {
            background-color: #28a745;
        }
        
        .negative {
            background-color: #dc3545;
        }
        
        .neutral {
            background-color: #6c757d;
        }
        
        .ad-idea-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #007bff;
        }
        
        .ad-idea-response {
            border-left: 5px solid #28a745;
        }
        
        .ad-idea-text {
            font-size: 18px;
            margin-bottom: 10px;
        }
        
        .ad-idea-tag {
            background-color: #e9ecef;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 5px;
        }
        
        .suggestion-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #dc3545;
        }
        
        .suggestion-list {
            list-style-type: none;
            padding-left: 0;
        }
        
        .suggestion-item {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .suggestion-item:last-child {
            border-bottom: none;
        }
        
        .tab-content {
            padding: 20px 0;
        }
        
        /* Custom styling for the search bar */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #007bff;
            color: white;
            border-radius: 20px;
        }
        
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #0056b3;
        }
    </style>
    """

def create_dashboard_html(tweets_df, sentiment_counts, ad_ideas=None):
    if tweets_df.empty:
        return "<p>No tweets to display</p>"
        
    # Calculate average sentiment
    avg_sentiment = tweets_df['sentiment_score'].mean()
    positive_pct = sentiment_counts.get('positive', 0) / len(tweets_df) * 100 if len(tweets_df) > 0 else 0
    negative_pct = sentiment_counts.get('negative', 0) / len(tweets_df) * 100 if len(tweets_df) > 0 else 0
    neutral_pct = sentiment_counts.get('neutral', 0) / len(tweets_df) * 100 if len(tweets_df) > 0 else 0
    
    metrics_html = f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-label">Average Sentiment</div>
            <div class="metric-value" style="color: {get_sentiment_color(avg_sentiment)};">{avg_sentiment:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Positive</div>
            <div class="metric-value" style="color: rgba(0, 128, 0, 0.8);">{positive_pct:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Neutral</div>
            <div class="metric-value" style="color: rgba(128, 128, 128, 0.8);">{neutral_pct:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Negative</div>
            <div class="metric-value" style="color: rgba(255, 0, 0, 0.8);">{negative_pct:.1f}%</div>
        </div>
    </div>
    """
    
    tweets_html = ""
    for _, tweet in tweets_df.iterrows():
        sentiment_class = tweet['sentiment_category']
        sentiment_score = tweet['sentiment_score']
        profile_img = tweet.get('profile_image', 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png')
        
        tweets_html += f"""
        <div class="tweet-container">
            <div class="tweet-header">
                <img class="tweet-avatar" src="{profile_img}" onerror="this.src='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'">
                <div>
                    <div class="tweet-username">@{tweet.get('username', 'user')}</div>
                    <div style="color: #666; font-size: 12px;">{tweet.get('created_at', '').split('T')[0]}</div>
                </div>
                <div style="margin-left: auto;">
                    <span class="sentiment-badge {sentiment_class}">{sentiment_class.capitalize()} ({sentiment_score:.2f})</span>
                </div>
            </div>
            <div class="tweet-content">{tweet['text']}</div>
            <div class="tweet-metrics">
                <div class="tweet-metric">❤️ {tweet.get('like_count', 0)}</div>
                <div class="tweet-metric">🔁 {tweet.get('retweet_count', 0)}</div>
                <div class="tweet-metric">💬 {tweet.get('reply_count', 0)}</div>
            </div>
        </div>
        """
    
    return f"""
    <div class="dashboard-container">
        {metrics_html}
        <h3>Recent Tweets</h3>
        {tweets_html}
    </div>
    """

def create_ad_ideas_html(ad_ideas):
    if not ad_ideas:
        return "<p>No ad ideas generated</p>"
    
    ad_html = ""
    for i, idea in enumerate(ad_ideas):
        idea_type = idea.get('type', 'positive')
        idea_class = "ad-idea-response" if idea_type == "response" else ""
        
        keywords_html = ""
        for keyword in idea.get('keywords', []):
            keywords_html += f'<span class="ad-idea-tag">{keyword}</span>'
        
        ad_html += f"""
        <div class="ad-idea-card {idea_class}">
            <div class="ad-idea-text">{idea.get('text', '')}</div>
            <div>{keywords_html}</div>
        </div>
        """
    
    return f"""
    <div class="dashboard-container">
        <h3>Ad Campaign Ideas</h3>
        {ad_html}
    </div>
    """

def create_damage_control_html(flagged_tweets, analyzer):
    if flagged_tweets.empty:
        return "<p>No flagged tweets requiring damage control</p>"
    
    suggestions_html = ""
    for _, tweet in flagged_tweets.iterrows():
        suggestions = analyzer.generate_damage_control_suggestions(tweet)
        suggestions_items = ""
        for suggestion in suggestions:
            suggestions_items += f'<li class="suggestion-item">{suggestion}</li>'
        
        profile_img = tweet.get('profile_image', 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png')
        
        suggestions_html += f"""
        <div class="suggestion-card">
            <div class="tweet-header">
                <img class="tweet-avatar" src="{profile_img}" onerror="this.src='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'">
                <div>
                    <div class="tweet-username">@{tweet.get('username', 'user')}</div>
                    <div style="color: #666; font-size: 12px;">{tweet.get('created_at', '').split('T')[0]}</div>
                </div>
                <div style="margin-left: auto;">
                    <span class="sentiment-badge negative">Negative ({tweet.get('sentiment_score', 0):.2f})</span>
                </div>
            </div>
            <div class="tweet-content">{tweet['text']}</div>
            <div class="tweet-metrics">
                <div class="tweet-metric">❤️ {tweet.get('like_count', 0)}</div>
                <div class="tweet-metric">🔁 {tweet.get('retweet_count', 0)}</div>
                <div class="tweet-metric">💬 {tweet.get('reply_count', 0)}</div>
            </div>
            <h4>Suggested Actions:</h4>
            <ul class="suggestion-list">
                {suggestions_items}
            </ul>
        </div>
        """
    
    return f"""
    <div class="dashboard-container">
        <h3>Damage Control Recommendations</h3>
        {suggestions_html}
    </div>
    """

def load_html():
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content



def main():
    st.set_page_config(page_title="Trendalyze – Twitter trends + analytics.", layout="wide")

    # Load and render the HTML file inside Streamlit
    components.html(load_html(), height=800, scrolling=True)

if __name__ == "__main__":
    main()
