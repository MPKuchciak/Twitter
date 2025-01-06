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
    # -------------------------------------------------------------------------
    # 1) TWEET FIELDS
    #   These define what attributes of each Tweet object you want returned.
    # -------------------------------------------------------------------------
    tweet_fields = [
        "attachments",         # Tweet may have polls or media (images/videos).
        "author_id",           # ID of the tweet's author (linked to user object).
        "context_annotations", # Additional context from Twitter's annotation system.
        "conversation_id",     # The 'root' tweet ID for this conversation.
        "created_at",          # When the tweet was created (ISO 8601).
        "edit_controls",       # Info about whether the tweet is editable, edit window, etc.
        "entities",            # Hashtags, URLs, user mentions, symbols.
        "geo",                 # Geo location info if the user tagged a place.
        "id",                  # Unique identifier of this tweet (string).
        "in_reply_to_user_id", # If this is a reply, the user ID it replies to.
        "lang",                # Language (BCP47) if detected.
        "non_public_metrics",  # Private metrics (requires OAuth 2.0 User Context).
        "public_metrics",      # Public engagement counts (likes, retweets, replies, quotes).
        "organic_metrics",     # Organic (non-promoted) metrics (requires OAuth 2.0 User Context).
        "promoted_metrics",    # Metrics for promoted (advertised) tweets (requires OAuth 2.0).
        "possibly_sensitive",  # Indicates if the tweet may contain sensitive content.
        "referenced_tweets",   # Data about retweets, quotes, or replies referencing another tweet.
        "reply_settings",      # Who can reply? (everyone, mentionedUsers, following).
        "source",              # Source app used to post (e.g., Twitter for iPhone).
        "text",                # Full text content of the tweet.
        "withheld",            # Info if the tweet is withheld in certain countries, etc.
        "note_tweet"           # Long-form tweet data if above 280 chars (note feature).
    ]

    # -------------------------------------------------------------------------
    # 2) USER FIELDS
    #   If you expand author_id or mention usernames, you can get these user attrs.
    # -------------------------------------------------------------------------
    user_fields = [
        "created_at",          # When the user account was created.
        "description",         # The user’s bio text.
        "entities",            # URLs, hashtags, and mentions in the user's profile.
        "id",                  # The user’s unique ID (string).
        "location",            # User-defined location (profile field).
        "most_recent_tweet_id",# ID of the user's most recent tweet (if accessible).
        "name",                # The user’s display name (not always unique).
        "pinned_tweet_id",     # ID of a tweet pinned on the user's profile (if any).
        "profile_image_url",   # URL to the user’s profile pic.
        "protected",           # True if this account is private (protected).
        "public_metrics",      # Follower count, following count, tweet count, etc.
        "url",                 # A URL in the user's profile.
        "username",            # The user’s handle (e.g. @someone), unique.
        "verified",            # True if the account is verified.
        "verified_type",       # Indicates the type of verification (blue check, etc.).
        "withheld"             # Info about withheld user content by country, etc.
    ]

    # -------------------------------------------------------------------------
    # 3) MEDIA FIELDS
    #   If tweets have attached media (images, GIFs, videos), these fields describe it.
    # -------------------------------------------------------------------------
    media_fields = [
        "duration_ms",         # Video duration in milliseconds (for video/gif).
        "height",              # Media height in pixels.
        "media_key",           # Unique media identifier referencing the media object.
        "preview_image_url",   # Preview image for video or GIF.
        "type",                # Type of media: photo, video, animated_gif.
        "url",                 # Direct URL of the media (if it's an image).
        "width",               # Media width in pixels.
        "public_metrics",      # Video metrics like view_count, etc. (for public).
        "non_public_metrics",  # Private metrics (requires user context).
        "organic_metrics",     # Organic metrics (requires user context).
        "promoted_metrics",    # Ad metrics (requires user context).
        "alt_text",            # Text description for accessibility (if provided).
        "variants"             # Different bitrates/encodings for video content.
    ]

    # -------------------------------------------------------------------------
    # 4) PLACE FIELDS
    #   If a tweet is tagged with a place (like a city), these fields describe it.
    # -------------------------------------------------------------------------
    place_fields = [
        "contained_within",    # Other places/cities that contain this location.
        "country",             # Country for this place.
        "country_code",        # ISO country code, e.g. "US", "PL", "FR".
        "full_name",           # Full text name, e.g. "Paris, France".
        "geo",                 # Geospatial info about this place.
        "id",                  # Unique place ID used by Twitter.
        "name",                # Short name, e.g. "Paris".
        "place_type"           # Type of place (city, admin, country, etc.).
    ]

    # -------------------------------------------------------------------------
    # 5) POLL FIELDS
    #   If a tweet references a poll, these fields describe that poll.
    # -------------------------------------------------------------------------
    poll_fields = [
        "duration_minutes",    # How long the poll runs (minutes).
        "end_datetime",        # When the poll ends.
        "id",                  # Unique poll ID referencing the poll object.
        "options",             # Poll options (e.g. choices + their vote counts).
        "voting_status"        # Whether the poll is open, closed, etc.
    ]

    # -------------------------------------------------------------------------
    # 6) EXPANSIONS
    #   These reference which additional objects you want included ("includes").
    #   Each expansion indicates a relationship to fetch:
    # -------------------------------------------------------------------------
    expansions = [
        "attachments.poll_ids",        # Expands poll objects if a tweet references them.
        "attachments.media_keys",      # Expands media objects (photos, GIFs, videos).
        "author_id",                   # Expands user object for the tweet's author.
        "in_reply_to_user_id",         # Expands user object for the user being replied to.
        "referenced_tweets.id",        # Expands any referenced tweets (RTs, quotes).
        "referenced_tweets.id.author_id", # Also expands authors of referenced tweets.
        "entities.mentions.username",  # Expands user objects for @mentions in text.
        "geo.place_id",                # Expands the place object if tweet is geotagged.
        "edit_history_tweet_ids"       # Expands historical versions of edited tweets.
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