Football YouTube Comments Analysis
A data analysis project that mines YouTube comments from football videos and performs sentiment analysis, player mentions tracking, and visualization of fan sentiment across different topics.
Overview
This project uses the YouTube API to collect comments from popular football-related videos and analyzes them to understand fan sentiment, trending topics, and player popularity. It covers major football events and matchups including Champions League highlights, Premier League goals, player comparisons (Messi vs Ronaldo), World Cup qualifiers, and rising stars like Haaland and Mbappé.
Project Performance

Total Comments Mined: 2,411
Videos Analyzed: 25
Search Queries: 5
Data Sources: YouTube API
Analysis Output: CSV files + Comprehensive dashboard visualization

Data Collection
YouTube Searches:

Champions League 2024 highlights
Premier League best goals 2024
Messi vs Ronaldo 2024
World Cup 2026 qualifiers
Haaland Mbappé 2024

Each search returns the 5 most relevant videos, with up to 100 comments extracted per video.
Key Findings
Top Mentioned Players:

Messi — 205 mentions
Ronaldo — 167 mentions
Mbappé — 109 mentions
Son — 92 mentions
Haaland — 87 mentions

Sentiment Distribution:

Neutral: 1,508 comments (62.5%)
Positive: 670 comments (27.8%)
Negative: 233 comments (9.7%)

Most Popular Topics:

Champions League matches
Premier League goals
Iconic player rivalries
World Cup qualifications
Emerging talent showcases

Requirements
googleapiclient
pandas
textblob
wordcloud
nltk
matplotlib
seaborn
Install dependencies:
bashpip install google-api-python-client pandas textblob wordcloud nltk matplotlib seaborn
Setup
1. Get YouTube API Key

Go to Google Cloud Console
Create a new project
Enable YouTube Data API v3
Create an API key (Credentials → Create Credentials → API Key)

2. Add API Key to Code
Replace API_KEY = "xxx" with your actual YouTube API key:
pythonAPI_KEY = "your_api_key_here"
Usage
bashpython youtube_football.py
The script will:

Connect to YouTube API
Search for football-related videos
Extract comments from each video
Clean and process text data
Perform sentiment analysis
Generate visualizations
Export cleaned data for further analysis

Output Files
1. raw_comments.csv
Raw data with all extracted comments including:

Comment text
Author name
Like count
Publication date

2. football_youtube_clean.csv
Cleaned and processed data ready for Tableau/BI tools:

Cleaned text
Sentiment classification
Player mentions
Hashtags extracted
Author information

3. football_youtube_analysis.png
Comprehensive dashboard with 7 visualizations:

Sentiment Distribution (Pie Chart) — Shows breakdown of positive, negative, and neutral comments
Most Mentioned Players (Bar Chart) — Top 10 most discussed players
Top Hashtags (Bar Chart) — Trending hashtags in comments
Average Likes by Sentiment (Bar Chart) — Engagement metrics by sentiment
Comments Over Time (Line Chart) — Comment activity timeline
Sentiment by Player (Grouped Bar Chart) — How each player is discussed
Word Cloud — Most frequent words in all comments

Code Structure
get_video_ids(query, max_results=10)
Searches YouTube for videos matching the query.

Parameters:

query: Search term
max_results: Number of videos to retrieve (default 10)


Returns: List of video IDs and titles

get_comments(video_id)
Extracts comments from a specific YouTube video.

Parameters:

video_id: YouTube video identifier


Returns: List of comment dictionaries with text, likes, date, author

clean_text(text)
Cleans comment text for analysis.

Converts to lowercase
Removes URLs
Removes mentions (@username)
Removes special characters
Trims whitespace

get_sentiment(text)
Performs sentiment analysis on text using TextBlob.

Returns: 'positive', 'negative', or 'neutral'

Sentiment Analysis
Uses TextBlob sentiment polarity scoring:

Positive: score > 0.1
Negative: score < -0.1
Neutral: -0.1 ≤ score ≤ 0.1

Data Processing

Text Cleaning:

URL removal
Special character removal
Lowercase conversion
Whitespace trimming


Feature Extraction:

Hashtag extraction
Player mention detection
Date/time parsing
Like count tracking


Sentiment Classification:

TextBlob polarity analysis
Three-class classification (positive/neutral/negative)



Visualizations
Color Scheme:

Positive: Green (#00ff87)
Neutral: Orange (#ffa502)
Negative: Red (#ff4757)
Dark background for better readability

Chart Types:

Pie charts for distributions
Bar charts for rankings
Line charts for trends
Grouped bar charts for comparisons
Word clouds for frequency visualization

Future Enhancements

Real-time comment streaming
Language detection and multi-language support
Advanced NLP (topic modeling, emotion detection)
Temporal analysis (trends over time)
Network analysis (comment threads and discussions)
Integration with Tableau for interactive dashboards
Expand to other sports (cricket, basketball, etc.)
Predict viral comments based on characteristics
Track player sentiment changes over seasons

API Rate Limits
YouTube API has quotas:

Daily quota: 10,000 units (default)
Search request: 100 units
Comments request: 1 unit per comment

Adjust max_results parameter if hitting rate limits.
Notes

Uses Python 3.9+
Requires internet connection for YouTube API access
Comment data may include emojis and special characters
API key should be kept private (add to .env file for production)
Some videos may have comments disabled

Troubleshooting
No comments found:

Check if video has comments enabled
Verify API key is valid
Ensure API quota is not exceeded

API Key Error:

Regenerate API key in Google Cloud Console
Check that YouTube Data API v3 is enabled
Verify the key has appropriate permissions

Import Errors:

Run pip install --upgrade <package_name>
Check Python version (3.9+)
Verify all dependencies are installed

References

YouTube API Documentation: https://developers.google.com/youtube/v3
TextBlob: https://textblob.readthedocs.io/
WordCloud: https://amueller.github.io/word_cloud/
Pandas: https://pandas.pydata.org/
Matplotlib: https://matplotlib.org/
Seaborn: https://seaborn.pydata.org/

License
This project is for educational and research purposes.
Author
Created as part of data analysis portfolio project.
