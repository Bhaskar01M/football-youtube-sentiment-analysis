import googleapiclient.discovery
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')

API_KEY = "xxx"

# ── 1. CONNECT TO YOUTUBE API ────────────────────────────────────
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

# ── 2. SEARCH FOR FOOTBALL VIDEOS ───────────────────────────────
def get_video_ids(query, max_results=10):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        relevanceLanguage="en"
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response['items']]
    titles = [item['snippet']['title'] for item in response['items']]
    return video_ids, titles

# ── 3. GET COMMENTS FROM VIDEOS ──────────────────────────────────
def get_comments(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text': comment['textDisplay'],
                'likes': comment['likeCount'],
                'date': comment['publishedAt'],
                'author': comment['authorDisplayName']
            })
    except Exception as e:
        print(f"  Skipping video {video_id}: {e}")
    return comments

# ── 4. MINE DATA ─────────────────────────────────────────────────
print("⚽ Mining YouTube football comments...")
queries = [
    "Champions League 2024 highlights",
    "Premier League best goals 2024",
    "Messi vs Ronaldo 2024",
    "World Cup 2026 qualifiers",
    "Haaland Mbappe 2024"
]

all_comments = []
all_titles = []

for query in queries:
    print(f"\nSearching: {query}")
    video_ids, titles = get_video_ids(query, max_results=5)
    all_titles.extend(titles)
    for vid_id, title in zip(video_ids, titles):
        print(f"  Getting comments from: {title[:50]}")
        comments = get_comments(vid_id)
        all_comments.extend(comments)
        print(f"  Got {len(comments)} comments")

df = pd.DataFrame(all_comments)
print(f"\n✅ Total comments mined: {len(df)}")
df.to_csv('raw_comments.csv', index=False)
print("✅ Raw data saved as raw_comments.csv")

# ── 5. CLEAN TEXT ────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9#\s]', '', text)
    return text.strip()

df['cleaned'] = df['text'].apply(clean_text)
df['date'] = pd.to_datetime(df['date']).dt.date
df['hour'] = pd.to_datetime(df['date']).apply(lambda x: x)

# ── 6. EXTRACT HASHTAGS ──────────────────────────────────────────
df['hashtags'] = df['text'].apply(lambda x: re.findall(r'#\w+', str(x).lower()))
all_hashtags = [tag for tags in df['hashtags'] for tag in tags]
hashtag_counts = Counter(all_hashtags).most_common(15)
print(f"\nTop hashtags: {hashtag_counts[:5]}")

# ── 7. PLAYER MENTIONS ───────────────────────────────────────────
players = ['messi', 'ronaldo', 'mbappe', 'neymar', 'benzema',
           'haaland', 'kane', 'salah', 'modric', 'de bruyne',
           'vinicius', 'bellingham', 'rashford', 'saka', 'son']

df['mentions'] = df['cleaned'].apply(
    lambda x: [p for p in players if p in str(x).lower()])
all_mentions = [m for mentions in df['mentions'] for m in mentions]
mention_counts = Counter(all_mentions).most_common(15)
print(f"Top mentions: {mention_counts[:5]}")

# ── 8. SENTIMENT ANALYSIS ────────────────────────────────────────
print("\nRunning sentiment analysis...")
def get_sentiment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['cleaned'].apply(get_sentiment)
print(f"Sentiment counts:\n{df['sentiment'].value_counts()}")

# ── 9. VISUALISATIONS ────────────────────────────────────────────
print("\nCreating visualisations...")
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Football YouTube Comments Analysis 2024',
             fontsize=24, fontweight='bold', color='white', y=0.98)

# Chart 1 - Sentiment Pie
ax1 = fig.add_subplot(4, 2, 1)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#00ff87', '#ffa502', '#ff4757']
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
        colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# Chart 2 - Player Mentions
ax2 = fig.add_subplot(4, 2, 2)
if mention_counts:
    mention_df = pd.DataFrame(mention_counts, columns=['Player', 'Mentions'])
    sns.barplot(data=mention_df.head(10), x='Mentions', y='Player',
                palette='cool', ax=ax2)
    ax2.set_title('Most Mentioned Players', fontsize=14, fontweight='bold')

# Chart 3 - Top Hashtags
ax3 = fig.add_subplot(4, 2, 3)
if hashtag_counts:
    hashtag_df = pd.DataFrame(hashtag_counts, columns=['Hashtag', 'Count'])
    sns.barplot(data=hashtag_df.head(10), x='Count', y='Hashtag',
                palette='YlOrRd', ax=ax3)
    ax3.set_title('Top Hashtags', fontsize=14, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No hashtags found', ha='center', va='center',
             color='white', fontsize=12)
    ax3.set_title('Top Hashtags', fontsize=14, fontweight='bold')

# Chart 4 - Likes by Sentiment
ax4 = fig.add_subplot(4, 2, 4)
df.groupby('sentiment')['likes'].mean().plot(
    kind='bar', ax=ax4, color=['#00ff87', '#ffa502', '#ff4757'])
ax4.set_title('Average Likes by Sentiment', fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=0)

# Chart 5 - Comments over Time
ax5 = fig.add_subplot(4, 2, 5)
comments_over_time = df.groupby('date').size()
ax5.plot(range(len(comments_over_time)),
         comments_over_time.values, color='#00ff87', linewidth=2, marker='o')
ax5.set_title('Comments Over Time', fontsize=14, fontweight='bold')
ax5.set_xlabel('Days')
ax5.set_ylabel('Number of Comments')

# Chart 6 - Sentiment by Player
ax6 = fig.add_subplot(4, 2, 6)
if all_mentions:
    df_exploded = df.explode('mentions').dropna(subset=['mentions'])
    if len(df_exploded) > 0:
        player_sentiment = df_exploded.groupby(
            ['mentions', 'sentiment']).size().unstack(fill_value=0)
        player_sentiment.head(8).plot(
            kind='bar', ax=ax6,
            color=['#ff4757', '#ffa502', '#00ff87'])
        ax6.set_title('Sentiment by Player', fontsize=14, fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)

# Chart 7 - Word Cloud
ax7 = fig.add_subplot(4, 1, 4)
stop_words = set(stopwords.words('english'))
stop_words.update(['im', 'dont', 'cant', 'one', 'get', 'got',
                   'like', 'just', 'know', 'think', 'yeah', 'lol'])
all_text = ' '.join(df['cleaned'].values)
wordcloud = WordCloud(
    width=1600, height=400,
    background_color='black',
    colormap='cool',
    stopwords=stop_words,
    max_words=150
).generate(all_text)
ax7.imshow(wordcloud, interpolation='bilinear')
ax7.axis('off')
ax7.set_title('Football Fan Word Cloud', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('football_youtube_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='black')
plt.show()
print("\n✅ Dashboard saved as football_youtube_analysis.png")

# ── 10. EXPORT FOR TABLEAU ───────────────────────────────────────
export_df = df[['text', 'cleaned', 'likes', 'date',
                'sentiment', 'mentions', 'hashtags', 'author']]
export_df.to_csv('football_youtube_clean.csv', index=False)
print("✅ Clean data exported as football_youtube_clean.csv for Tableau!")
print(f"\n📊 Summary:")
print(f"  Total comments: {len(df)}")
print(f"  Videos analysed: {len(all_titles)}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
