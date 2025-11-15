import requests
import pandas as pd
import time
from requests.exceptions import RequestException, Timeout, ConnectionError
import os

from TrackC.Jailbreak.util import make_api_call, judge_attack



if __name__ == "__main__":
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