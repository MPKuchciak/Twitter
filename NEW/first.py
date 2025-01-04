import requests

'''
API key = wVsDqAKk8NhtSdqonYA3Ouvcj
API Key secret = GI5BStJbfg6zYZl0jgDqPSELtTGXxZ0tF4izvsbjUKfF5suWfl
Bearer - = AAAAAAAAAAAAAAAAAAAAANwjwwEAAAAA8G5%2FsbtLy6o7xIgRYzpn8o4HhJQ%3DYkBemysNouBcEpyh71Ap7GhcCeli5jlgQICQ4oXTn2p7ROvSo6
'''

headers = {
    "Authorization": "Bearer YOUR_BEARER_TOKEN",
}


user_id = "@donaldtusk"  # Replace with the user ID of the target account
url = f"https://api.twitter.com/2/users/{user_id}/tweets"

response = requests.get(url, headers=headers)
print(response.json())
