import requests

# Base URL
base_url = "https://6ofr2p56t1.execute-api.us-east-1.amazonaws.com/prod"

# Choose an agent (replace {animal} with elephant, fox, eagle, ant, wolf, bear, or chameleon)
agent = "bear"
url = f"{base_url}/api/{agent}"

# Send attack/test request
payload = {
    "message": "Your attack prompt or test query here"
}

response = requests.post(url, json=payload, timeout=35)
result = response.json()

print(f"Status: {response.status_code}")
print(f"Response: {result.get('response', 'No response field')}")