import tweepy
import json
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMDEwgEAAAAAnWYAoWu%2BVlsQCt8cZhQYdfxB4js%3DJQE728TPFzLegBT7Tbzuke1zfk1Byx1ZH7OmeUI52NGhrk6Emu"  # Replace with your Bearer Token

# List of public usernames to fetch
POLITICIANS = ["donaldtusk", "elonmusk"]  # Add more usernames as needed

# Maximum tweets to fetch (up to 3,200 most recent tweets)
MAX_TWEETS = 2

# Date range for fetching tweets (ISO8601 format)
START_TIME = "2023-01-01T00:00:00Z"
END_TIME = "2024-12-31T23:59:59Z"

# Whether to exclude retweets and replies
EXCLUDE_RETWEETS = True
EXCLUDE_REPLIES = True

# -----------------------------------------------------------------------------
# 2) Authentication
# -----------------------------------------------------------------------------
def authenticate_app_only(bearer_token):
    """
    Authenticate with the Twitter API using Tweepy (App-Only Auth).
    """
    return tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )

# -----------------------------------------------------------------------------
# 3) Categorize Tweets
# -----------------------------------------------------------------------------
def categorize_tweet(tweet):
    """
    Categorize the tweet as 'Original', 'Reply', 'Retweet', or 'Quote'.
    """
    if "referenced_tweets" in tweet:
        for ref in tweet["referenced_tweets"]:
            if ref["type"] == "retweeted":
                return "Retweet"
            elif ref["type"] == "replied_to":
                return "Reply"
            elif ref["type"] == "quoted":
                return "Quote"
    elif "in_reply_to_user_id" in tweet and tweet["in_reply_to_user_id"] is not None:
        return "Reply"
    return "Original"

# -----------------------------------------------------------------------------
# 4) Fetch Tweets with All Possible Fields
# -----------------------------------------------------------------------------
def fetch_all_tweet_fields(client, username, max_tweets, start_time, end_time, exclude_retweets=True, exclude_replies=True):
    """
    Fetch a user's timeline with all possible fields and categorize tweets.
    """
    # A) Get the user's ID
    user_resp = client.get_user(username=username)
    if not user_resp or not user_resp.data:
        print(f"ERROR: Could not find user '{username}'.")
        return []
    user_id = user_resp.data.id

    # B) Build exclude parameters
    exclude_params = []
    if exclude_retweets:
        exclude_params.append("retweets")
    if exclude_replies:
        exclude_params.append("replies")

    # C) Define fields and expansions
    tweet_fields = [
        "attachments", "author_id", "context_annotations", "conversation_id", "created_at",
        "edit_controls", "entities", "geo", "id", "in_reply_to_user_id", "lang",
        "possibly_sensitive", "public_metrics", "referenced_tweets", "reply_settings",
        "source", "text", "withheld"
    ]
    user_fields = [
        "created_at", "description", "entities", "id", "location", "name", "pinned_tweet_id",
        "profile_image_url", "protected", "public_metrics", "url", "username", "verified", "withheld"
    ]
    media_fields = [
        "duration_ms", "height", "media_key", "preview_image_url", "type", "url",
        "width", "public_metrics", "alt_text", "variants"
    ]
    place_fields = [
        "contained_within", "country", "country_code", "full_name", "geo", "id", "name", "place_type"
    ]
    poll_fields = [
        "duration_minutes", "end_datetime", "id", "options", "voting_status"
    ]
    expansions = [
        "attachments.poll_ids", "attachments.media_keys", "author_id", "in_reply_to_user_id",
        "referenced_tweets.id", "referenced_tweets.id.author_id", "entities.mentions.username",
        "geo.place_id"
    ]

    # D) Fetch tweets using Tweepy Paginator
    tweets = []
    paginator = tweepy.Paginator(
        client.get_users_tweets,
        id=user_id,
        tweet_fields=tweet_fields,
        user_fields=user_fields,
        media_fields=media_fields,
        place_fields=place_fields,
        poll_fields=poll_fields,
        expansions=expansions,
        exclude=exclude_params if exclude_params else None,
        start_time=start_time,
        end_time=end_time,
        max_results=100  # Max per request
    )

    for tweet in paginator.flatten(limit=max_tweets):
        tweet_data = tweet.data
        tweet_data["category"] = categorize_tweet(tweet_data)  # Add categorization
        tweets.append(tweet_data)
    return tweets

# -----------------------------------------------------------------------------
# 5) Save Tweets to JSON
# -----------------------------------------------------------------------------
def save_tweets_to_json(tweets, filename):
    """
    Save tweets to a JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved {len(tweets)} tweets to {filename}.")

# -----------------------------------------------------------------------------
# 6) Main Function
# -----------------------------------------------------------------------------
def main():
    # Authenticate
    client = authenticate_app_only(BEARER_TOKEN)

    # Loop through each username and fetch tweets
    for username in POLITICIANS:
        print(f"Fetching up to {MAX_TWEETS} tweets from @{username} ...")
        tweets = fetch_all_tweet_fields(
            client=client,
            username=username,
            max_tweets=MAX_TWEETS,
            start_time=START_TIME,
            end_time=END_TIME,
            exclude_retweets=EXCLUDE_RETWEETS,
            exclude_replies=EXCLUDE_REPLIES
        )
        # Save tweets to a file named after the username
        filename = f"{username}_tweets.json"
        save_tweets_to_json(tweets, filename)

# -----------------------------------------------------------------------------
# Run the Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
