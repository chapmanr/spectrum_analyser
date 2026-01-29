"""Test script to verify the signal generation endpoint."""
import requests
import json

url = "http://127.0.0.1:5000/generate_sine"
payload = {
    "freq": 15000,
    "duration": 1.0,
    "sample_rate": 2000000,
    "amplitude": 0.5,
    "noise_level": 0.01,
    "harmonics": 0,
    "sine_stop_frac": 0.85,
    "num_segments": 1,
    "freq_drift": 0.0,
    "make_nonnegative": True
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)[:500]}")
except Exception as e:
    print(f"Error: {e}")
