import requests
import json
import sys

# Test the normative modeling endpoint
data = {
    'user_id': 'test123',
    'age': 25,
    'sex': 'Male',
    'brain_data': {
        'left_accumbens_area': 500,
        'right_accumbens_area': 480,
        'brain_stem': 12000,
        'left_caudate': 3500,
        'right_caudate': 3400
    }
}

try:
    response = requests.post('http://localhost:8000/normative', json=data, timeout=30)
    print(f'Status Code: {response.status_code}')
    print(f'Response Size: {len(response.content)} bytes')
    if response.status_code == 200:
        result = response.json()
        print(f'Number of regions: {len(result.get("region_analyses", {}))}')
        if result.get('region_analyses'):
            sample_region = next(iter(result['region_analyses'].keys()))
            sample_data = result['region_analyses'][sample_region]
            print(f'Sample region ages: {len(sample_data.get("ages", []))}')
            print(f'Sample region percentiles: {len(sample_data.get("percentile_curves", {}))}')
    else:
        print(f'Error response: {response.text[:500]}')
except Exception as e:
    print(f'Request failed: {e}')