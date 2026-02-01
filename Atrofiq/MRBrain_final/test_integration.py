#!/usr/bin/env python3
"""
Integration Test Script for MRBrain Final
Tests the new API endpoints and validates responses
"""

import json
import requests
import pandas as pd
import tempfile
from pathlib import Path

API_BASE = "http://localhost:8001"

def create_test_feature_data(participant_id="test_participant", age=45, gender="M"):
    """Create test feature data CSV file"""
    data = {
        'participant_id': [participant_id],
        'Age': [age],
        'Sex': [gender],
        'left_cerebral_white_matter': [250000],
        'left_cerebral_cortex': [230000],
        'left_lateral_ventricle': [12000],
        'left_thalamus': [8500],
        'left_caudate': [3800],
        'left_putamen': [5200],
        'left_pallidum': [1800],
        'left_hippocampus': [4200],
        'left_amygdala': [1600],
        'right_cerebral_white_matter': [250000],
        'right_cerebral_cortex': [230000],
        'right_lateral_ventricle': [12000],
        'right_thalamus': [8500],
        'right_caudate': [3800],
        'right_putamen': [5200],
        'right_pallidum': [1800],
        'right_hippocampus': [4200],
        'right_amygdala': [1600],
    }
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_api_status():
    """Test API status endpoint"""
    print("Testing API status...")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API Version: {data.get('version', 'Unknown')}")
            print(f"BrainAge Available: {data.get('brainage_available', False)}")
            print(f"Normative Available: {data.get('normative_available', False)}")
            return True
        else:
            print(f"Failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("\nTesting health check...")
    try:
        response = requests.get(f"{API_BASE}/status")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Health: {data.get('status', 'Unknown')}")
            print(f"BrainAge Module: {data.get('brainage_module', 'Unknown')}")
            print(f"Normative Module: {data.get('normative_module', 'Unknown')}")
            return True
        else:
            print(f"Failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_brain_age_prediction():
    """Test brain age prediction endpoint"""
    print("\nTesting brain age prediction...")
    
    # Create test data
    test_file = create_test_feature_data("test_001", 45, "M")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'feature_data': f}
            data = {
                'age': '45',
                'gender': 'M',
                'participant_id': 'test_001'
            }
            
            response = requests.post(f"{API_BASE}/brain-age", files=files, data=data)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Job ID: {result.get('job_id', 'N/A')}")
            print(f"Participant: {result.get('participant_id', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            print(f"Chronological Age: {result.get('chronological_age', 'N/A')}")
            print(f"Predicted Brain Age: {result.get('predicted_brain_age', 'N/A')}")
            print(f"Brain Age Gap: {result.get('brain_age_gap', 'N/A')}")
            print(f"Processing Time: {result.get('processing_time_seconds', 'N/A')}s")
            return True
        else:
            print(f"Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up
        try:
            Path(test_file).unlink()
        except:
            pass

def test_regions_endpoint():
    """Test available regions endpoint"""
    print("\nTesting available regions...")
    try:
        response = requests.get(f"{API_BASE}/regions")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            regions = data.get('available_regions', {})
            print(f"Male regions: {len(regions.get('male', []))}")
            print(f"Female regions: {len(regions.get('female', []))}")
            if regions.get('male'):
                print(f"Sample male regions: {regions['male'][:3]}")
            return True
        else:
            print(f"Failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("MRBrain Final Integration Tests")
    print("=" * 40)
    
    tests = [
        ("API Status", test_api_status),
        ("Health Check", test_health_check),
        ("Brain Age Prediction", test_brain_age_prediction),
        ("Available Regions", test_regions_endpoint),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("=" * 40)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)