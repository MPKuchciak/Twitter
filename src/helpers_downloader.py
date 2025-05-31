import tweepy
import json
from datetime import datetime, timedelta
import os 
from time import sleep


def authenticate_app_only(bearer_token):
    """
    Authenticate with the Twitter API using Tweepy (App-Only Auth).
    """
    return tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )

# Custom function for feetching all tweet fields
def fetch_all_tweet_fields(client, username, start_time, end_time, exclude_retweets=True, exclude_replies=True):
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
        max_results = 100  # Max per request                  
    )
    for tweet in paginator.flatten(limit=1000):
        tweet_data = tweet.data
        tweet_data["category"] = categorize_tweet(tweet_data)  # Add categorization
        tweets.append(tweet_data)
    return tweets

# Custom function for categorizing tweets
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

def save_tweets_to_json(tweets, filename, folder="../data/01.raw/tweets_data_final"):

    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved {len(tweets)} tweets to {file_path}.")

def download_tweets(username, party, START_TIME, END_TIME, before, client, EXCLUDE_RETWEETS=True, EXCLUDE_REPLIES=False):
    org_START_TIME = START_TIME
    org_END_TIME = END_TIME

    tweets = []

    new_tweets = fetch_all_tweet_fields(
            client=client,
            username=username,
            start_time=START_TIME,
            end_time=END_TIME,
            exclude_retweets=EXCLUDE_RETWEETS,
            exclude_replies=EXCLUDE_REPLIES
        )
    tweets.extend(new_tweets)
    print(f'Fetched {len(new_tweets)} tweets')

    while (len(new_tweets) > 1):
        last_tweet_time = datetime.fromisoformat(tweets[-1]["created_at"].replace("Z", "+00:00"))
        new_time = last_tweet_time + timedelta(seconds=1)
        new_time_iso = new_time.isoformat().replace("+00:00", "Z")
        END_TIME = new_time_iso
        print(f"Fetching tweets from {START_TIME} to {END_TIME}")
        new_tweets = fetch_all_tweet_fields(
            client=client,
            username=username,
            start_time=START_TIME,
            end_time=END_TIME,
            exclude_retweets=EXCLUDE_RETWEETS,
            exclude_replies=EXCLUDE_REPLIES
        )
        print(f'Fetched {len(new_tweets)} tweets')
        tweets.extend(new_tweets)    
        sleep(10)
    if before:
        folder = f"../data/01.raw/tweets_before_elections/{party}"
    else:
        folder = f"../data/01.raw/tweets_after_elections/{party}"
    save_tweets_to_json(tweets, f"{username}_{org_START_TIME[0:10]}_{org_END_TIME[0:10]}.json",folder=folder)