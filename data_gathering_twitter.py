import tweepy
import json
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# 0) Disclosure 
# -----------------------------------------------------------------------------

# Is Tweepy Library Free?
# Certainly! Tweepy Library is an open-source, completely free-of-cost library. However, keep in mind that Twitter's API has both free and paid tiers.

# We can use it for our article without worries 


# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
BEARER_TOKEN = "BEARER CODE"

# Single username to fetch from. 
# USERNAME = "donaldtusk"  # Example politician or public figure
# politicians' usernames
POLITICIANS = ["donaldtusk"]

# Maximum tweets you'd like to attempt to fetch. Twitter typically allows
# up to 3,200 of the user's most recent Tweets with get_users_tweets.
MAX_TWEETS = 5 #4000 # 3200

# Start & end times (ISO8601). Even though you can fetch up to 3,200,
# you might filter by date if you want only a certain window.
START_TIME = "2023-01-01T00:00:00Z"
END_TIME   = "2024-12-31T23:59:59Z"

# Whether to exclude retweets/replies (only original tweets)
EXCLUDE_RETWEETS = True
EXCLUDE_REPLIES  = True

# Final JSON filename
OUTPUT_FILE = "all_fields_tweets.json"


# -----------------------------------------------------------------------------
# 2) Authenticate with Tweepy (rate-limit aware - provided by Tweepy)
# -----------------------------------------------------------------------------
def authenticate_app_only(bearer_token):
    """
    App-only authentication with a Bearer Token. 
    This won't give you private metrics or user context.
    """
    return tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )


# -----------------------------------------------------------------------------
# 3) Fetch All Possible Fields from a User's Timeline
# -----------------------------------------------------------------------------
def fetch_all_tweet_fields(
    client,
    username,
    max_tweets,
    start_time,
    end_time,
    exclude_retweets=True,
    exclude_replies=True
):
    """
    Fetch a user's timeline (their own tweets) using get_users_tweets, including
    every possible field and expansion documented. By default, excludes retweets
    and replies. Returns a list of Tweepy Tweet objects.
    """

    # -------------------------------------------------------------------------
    # A) Look up the user's ID from the username
    # -------------------------------------------------------------------------
    user_resp = client.get_user(username=username)
    if not user_resp or not user_resp.data:
        print(f"ERROR: Could not find user '{username}'.")
        return []
    user_id = user_resp.data.id

    # -------------------------------------------------------------------------
    # B) Build the exclude list, if applicable
    # -------------------------------------------------------------------------
    exclude_params = []
    if exclude_retweets:
        exclude_params.append("retweets")
    if exclude_replies:
        exclude_params.append("replies")

    # -------------------------------------------------------------------------
    # C) Define ALL expansions & fields you want
    # -------------------------------------------------------------------------
    #  - Tweet fields
    tweet_fields = [
        "attachments",
        "author_id",
        "context_annotations",
        "conversation_id",
        "created_at",
        "edit_controls",        # For editable tweets
        "entities",
        "geo",
        "id",
        "in_reply_to_user_id",
        "lang",
        "non_public_metrics",   # Requires OAuth2 user context w/ correct scopes
        "public_metrics",
        "organic_metrics",      # Requires OAuth2 user context
        "promoted_metrics",     # Requires OAuth2 user context
        "possibly_sensitive",
        "referenced_tweets",
        "reply_settings",
        "source",
        "text",
        "withheld",
        "note_tweet"            # For tweets longer than 280 chars
    ]

    #  - User fields
    user_fields = [
        "created_at",
        "description",
        "entities",
        "id",
        "location",
        "most_recent_tweet_id",
        "name",
        "pinned_tweet_id",
        "profile_image_url",
        "protected",
        "public_metrics",
        "url",
        "username",
        "verified",
        "verified_type",
        "withheld"
    ]

    #  - Media fields
    media_fields = [
        "duration_ms",
        "height",
        "media_key",
        "preview_image_url",
        "type",
        "url",
        "width",
        "public_metrics",       # For video view counts, etc.
        "non_public_metrics",   # Requires OAuth2 user context
        "organic_metrics",      # Requires OAuth2 user context
        "promoted_metrics",     # Requires OAuth2 user context
        "alt_text",
        "variants"
    ]

    #  - Place fields
    place_fields = [
        "contained_within",
        "country",
        "country_code",
        "full_name",
        "geo",
        "id",
        "name",
        "place_type"
    ]

    #  - Poll fields
    poll_fields = [
        "duration_minutes",
        "end_datetime",
        "id",
        "options",
        "voting_status"
    ]

    #  - Expansions
    expansions = [
        "attachments.poll_ids",
        "attachments.media_keys",
        "author_id",
        "in_reply_to_user_id",
        "referenced_tweets.id",
        "referenced_tweets.id.author_id",
        "entities.mentions.username",
        "geo.place_id",
        "edit_history_tweet_ids"
    ]

    # -------------------------------------------------------------------------
    # D) Use Tweepy Paginator to gather up to max_tweets
    # -------------------------------------------------------------------------
    tweets_data = []

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
        max_results=100  # Max per API call
    )

    for tweet in paginator.flatten(limit=max_tweets):
        tweets_data.append(tweet)

    return tweets_data


# -----------------------------------------------------------------------------
# 4) Save Tweets to JSON (Raw Dump)
# -----------------------------------------------------------------------------
def save_tweets_to_json(tweets, filename):
    """
    Saves a list of Tweepy Tweet objects to JSON. By default, we dump
    'tweet.data' which is the Tweet's raw dictionary. We do NOT fully
    merge expansions here; if you need that, you'd parse the original
    Response objects in the Paginator for 'includes'.
    """
    tweets_as_dicts = [t.data for t in tweets]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(tweets_as_dicts, f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved {len(tweets_as_dicts)} tweets to {filename}.")


# -----------------------------------------------------------------------------
# 5) Main Execution
# -----------------------------------------------------------------------------
def main():
    # 1) App-only Auth
    client = authenticate_app_only(BEARER_TOKEN)

    # 2) Loop over politician usernames
    for username in POLITICIANS:
        print(f"Fetching up to {MAX_TWEETS} tweets from @{username} ...")
        # fetch
        tweets = fetch_all_tweet_fields(
            client=client,
            username=username,
            max_tweets=MAX_TWEETS,
            start_time=START_TIME,
            end_time=END_TIME,
            exclude_retweets=EXCLUDE_RETWEETS,
            exclude_replies=EXCLUDE_REPLIES
        )
        # save
        filename = f"{username}_tweets.json"
        save_tweets_to_json(tweets, filename)


if __name__ == "__main__":
    main()