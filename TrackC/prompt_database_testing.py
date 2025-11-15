import requests
import pandas as pd
import time
from requests.exceptions import RequestException, Timeout, ConnectionError
import os

BASE_URL = "https://6ofr2p56t1.execute-api.us-east-1.amazonaws.com/prod"

def make_api_call(agent="elephant", message=None, max_retries=2, initial_wait=1):
    """
    Make an API call with retry logic for queue waits and errors.
    
    Args:
        message: The message to send to the API
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds (exponential backoff)
    
    Returns:
        Response JSON or None if all retries fail
    """
    url = f"{BASE_URL}/api/{agent}"
    wait_time = initial_wait
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json={"message": message},
                timeout=35
            )
            
            # Handle successful response
            if response.status_code == 200:
                return response.json()
            
            # Handle rate limiting / queue waits (429, 503, 502)
            elif response.status_code in [429, 502, 503]:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                    print(f"Rate limited or in queue. Waiting {wait_time} seconds (Retry-After header)...")
                else:
                    wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
                    print(f"Rate limited or in queue (status {response.status_code}). Waiting {wait_time} seconds (attempt {attempt + 1}/{max_retries})...")
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached. Status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
            
            # Handle other HTTP errors
            else:
                print(f"HTTP error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 60)
                    continue
                else:
                    return None
        
        except Timeout:
            print(f"Request timeout (attempt {attempt + 1}/{max_retries})...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 60)
                continue
            else:
                print("Max retries reached for timeout.")
                return None
        
        except ConnectionError as e:
            print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 60)
                continue
            else:
                print("Max retries reached for connection error.")
                return None
        
        except RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 60)
                continue
            else:
                print("Max retries reached for request error.")
                return None
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    return None

path = os.path.join(os.path.dirname(__file__), "RedTeamingPrompts")

benign_df = pd.read_csv(os.path.join(path, "benign_test_cases.csv"))
harmful_df = pd.read_csv(os.path.join(path, "harmful_test_cases.csv"))
jailbreak_df = pd.read_csv(os.path.join(path, "jailbreak_prompts.csv"))

# Test with benign queries
print("Testing benign queries...")
for idx, row in benign_df.iterrows():
    print(f"Benign query {idx + 1}/{len(benign_df)}: {row['question'][:50]}...")
    result = make_api_call(agent="elephant", message=row['question'])
    if result:
        print(result)
    else:
        print(f"Failed to get response for benign query {idx + 1}")
    # Evaluate response...

# Test with harmful queries
print("\nTesting harmful queries...")
for idx, row in harmful_df.iterrows():
    print(f"Harmful query {idx + 1}/{len(harmful_df)}: {row['question'][:50]}...")
    result = make_api_call(agent="elephant", message=row['question'])
    if result:
        print(result)
    else:
        print(f"Failed to get response for harmful query {idx + 1}")
    # Evaluate refusal rate...

# Test with jailbreak prompts
print("\nTesting jailbreak prompts...")
for idx, row in jailbreak_df.iterrows():
    print(f"Jailbreak {idx + 1}/{len(jailbreak_df)}: {row['prompt'][:50]}...")
    result = make_api_call(agent="elephant", message=row['prompt'])
    if result:
        print(result)
    else:
        print(f"Failed to get response for jailbreak {idx + 1}")
    # Evaluate jailbreak success rate...