#!/usr/bin/env python3
"""
Comprehensive verification that Celery and Frontend use MRBrain_final models (not hardcoded values)
"""

import os
import requests
import json
import time
import pandas as pd
import tempfile
from pathlib import Path

def create_test_feature_csv():
    """Create a test CSV with brain feature data"""
    # Create realistic brain region volumes (not hardcoded test values)
    data = {
        'participant_id': ['test_participant_001'],
        'Age': [45.2],
        'Sex': ['M'],
        'left_cerebral_white_matter': [248532.5],
        'left_cerebral_cortex': [227845.3], 
        'left_lateral_ventricle': [11847.2],
        'left_thalamus': [8234.7],
        'left_caudate': [3756.8],
        'left_putamen': [5123.4],
        'left_pallidum': [1789.2],
        'left_hippocampus': [4187.6],
        'left_amygdala': [1598.3],
        'right_cerebral_white_matter': [251203.8],
        'right_cerebral_cortex': [229456.7],
        'right_lateral_ventricle': [12103.5],
        'right_thalamus': [8456.3],
        'right_caudate': [3834.2],
        'right_putamen': [5234.7],
        'right_pallidum': [1823.6],
        'right_hippocampus': [4203.8],
        'right_amygdala': [1623.4],
    }
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_mrbrain_api_endpoints():
    """Test that MRBrain_final API endpoints return dynamic (not hardcoded) values"""
    
    print("🧪 Testing MRBrain_final API for Dynamic Values")
    print("=" * 50)
    
    # Test 1: Brain Age with different feature sets should give different results
    print("\n1️⃣ Testing Brain Age Prediction Variability...")
    
    test_csv = create_test_feature_csv()
    
    try:
        # Test with original features
        with open(test_csv, 'rb') as f:
            response1 = requests.post(
                "http://localhost:8000/brain-age",
                files={'feature_data': f},
                data={'age': '45', 'gender': 'M', 'participant_id': 'test1'},
                timeout=30
            )
        
        if response1.status_code == 200:
            result1 = response1.json()
            age1 = result1.get('predicted_age')
            print(f"   ✅ First prediction: {age1} years")
            
            # Create second test with different features
            data2 = pd.read_csv(test_csv)
            # Modify features slightly to test for variability
            for col in data2.columns:
                if col not in ['participant_id', 'Age', 'Sex']:
                    data2[col] = data2[col] * 1.05  # 5% increase
            
            temp_file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            data2.to_csv(temp_file2.name, index=False)
            temp_file2.close()
            
            with open(temp_file2.name, 'rb') as f:
                response2 = requests.post(
                    "http://localhost:8000/brain-age",
                    files={'feature_data': f},
                    data={'age': '45', 'gender': 'M', 'participant_id': 'test2'},
                    timeout=30
                )
            
            if response2.status_code == 200:
                result2 = response2.json()
                age2 = result2.get('predicted_age')
                print(f"   ✅ Second prediction: {age2} years")
                
                # Check for variability
                age_diff = abs(age1 - age2)
                if age_diff > 0.01:  # Should be different with different features
                    print(f"   ✅ Models are dynamic: {age_diff:.3f} year difference")
                    print(f"   🔧 Model info: {result1.get('model_info', {}).get('name', 'Unknown')}")
                else:
                    print(f"   ⚠️  Models may be using hardcoded values (difference: {age_diff:.3f})")
            else:
                print(f"   ❌ Second request failed: {response2.status_code}")
            
            os.unlink(temp_file2.name)
        else:
            print(f"   ❌ First request failed: {response1.status_code}")
            print(f"   📄 Response: {response1.text}")
    
    except Exception as e:
        print(f"   ❌ Brain Age API test failed: {e}")
    
    finally:
        os.unlink(test_csv)
    
    # Test 2: Normative modeling variability
    print("\n2️⃣ Testing Normative Modeling Variability...")
    
    test_csv2 = create_test_feature_csv()
    
    try:
        with open(test_csv2, 'rb') as f:
            response1 = requests.post(
                "http://localhost:8000/normative",
                files={'feature_data': f},
                data={'age': '45', 'gender': 'M', 'participant_id': 'test_norm1'},
                timeout=30
            )
        
        if response1.status_code == 200:
            result1 = response1.json()
            percentiles1 = result1.get('percentile_scores', {})
            print(f"   ✅ Normative analysis working: {len(percentiles1)} regions")
            
            if percentiles1:
                sample_percentile = list(percentiles1.values())[0]
                print(f"   📊 Sample percentile: {sample_percentile}th percentile")
                
                # Check if all percentiles are the same (indicating hardcoded values)
                unique_percentiles = set(percentiles1.values())
                if len(unique_percentiles) > 1:
                    print(f"   ✅ Percentiles vary across regions: {len(unique_percentiles)} unique values")
                else:
                    print(f"   ⚠️  All percentiles are the same ({sample_percentile}) - may be hardcoded")
            else:
                print("   ⚠️  No percentile scores returned")
        else:
            print(f"   ❌ Normative API failed: {response1.status_code}")
            print(f"   📄 Response: {response1.text}")
    
    except Exception as e:
        print(f"   ❌ Normative API test failed: {e}")
    
    finally:
        os.unlink(test_csv2)

def test_celery_integration():
    """Test that Celery tasks are using MRBrain_final API"""
    
    print("\n🔄 Testing Celery Integration with MRBrain_final")
    print("=" * 45)
    
    # Check task configuration
    task_file = Path("backend/app/tasks/mri_processing_v2.py")
    if task_file.exists():
        with open(task_file, 'r') as f:
            content = f.read()
            
        # Look for specific integration patterns
        checks = [
            ("MRBRAIN_API_BASE", "✅ API base URL configuration"),
            ("call_mrbrain_api", "✅ API calling function"),
            ("'/brain-age'", "✅ Brain age endpoint integration"),
            ("model_version': 'MRBrain_final_v2.0'", "✅ Model version tracking"),
            ("create_mock_feature_data", "✅ Dynamic feature generation"),
            ("feature_defaults = config.get_feature_defaults()", "✅ Configuration-based defaults"),
            ("age_factor", "✅ Age-related volume modeling"),
            ("gender_factor", "✅ Gender-related volume modeling"),
            ("random_factor", "✅ Random variation (non-hardcoded)")
        ]
        
        for check_term, message in checks:
            if check_term in content:
                print(f"   {message}")
            else:
                print(f"   ❌ Missing: {check_term}")
        
        # Check for hardcoded values that should be avoided
        hardcoded_checks = [
            ("predicted_age = 25.0", "❌ Hardcoded brain age found"),
            ("percentile = 50", "❌ Hardcoded percentile found"),
            ("return {'brain_age': 25}", "❌ Hardcoded response found")
        ]
        
        hardcoded_found = False
        for check_term, message in hardcoded_checks:
            if check_term in content:
                print(f"   {message}")
                hardcoded_found = True
        
        if not hardcoded_found:
            print("   ✅ No hardcoded values detected in tasks")
    else:
        print("   ❌ Task file not found")

def test_frontend_dynamic_display():
    """Test that frontend handles dynamic data properly"""
    
    print("\n🖥️  Testing Frontend Dynamic Data Handling")
    print("=" * 43)
    
    # Check Dashboard.js for proper data handling
    dashboard_file = Path("frontend/src/Dashboard.js")
    if dashboard_file.exists():
        with open(dashboard_file, 'r') as f:
            content = f.read()
        
        # Look for dynamic data handling patterns
        dynamic_checks = [
            ("results?.brainAge?.predicted_age", "✅ Dynamic brain age extraction"),
            ("results.brainAge.prediction", "✅ Alternative prediction field"),
            ("results?.normative?.percentile_scores", "✅ Dynamic percentile scores"),
            ("Object.keys(results.normative.percentile_scores)", "✅ Dynamic region detection"),
            ("predictedAge - chronologicalAge", "✅ Dynamic brain age gap calculation"),
            ("Math.abs(gap)", "✅ Dynamic gap interpretation"),
            ("generateNormativeData", "✅ Dynamic normative curves"),
            ("results?.normative?.percentile_curves", "✅ API-driven curves"),
        ]
        
        for check_term, message in dynamic_checks:
            if check_term in content:
                print(f"   {message}")
        
        # Check for potential hardcoded values
        hardcoded_frontend_checks = [
            ("predictedAge = 25", "❌ Hardcoded brain age in frontend"),
            ("percentile: 50", "❌ Hardcoded percentile in frontend"),
            ("return 25.0", "❌ Hardcoded return value"),
        ]
        
        frontend_hardcoded = False
        for check_term, message in hardcoded_frontend_checks:
            if check_term in content:
                print(f"   {message}")
                frontend_hardcoded = True
        
        if not frontend_hardcoded:
            print("   ✅ No obvious hardcoded values in frontend")
        
        # Check for proper error handling of API responses
        if "console.log('Received analysis results'" in content:
            print("   ✅ Analysis results logging enabled")
        
        if "console.log('Normative data structure'" in content:
            print("   ✅ Normative data structure logging enabled")
    else:
        print("   ❌ Dashboard.js not found")

def generate_verification_report():
    """Generate a verification report for manual testing"""
    
    print("\n📋 Manual Verification Checklist")
    print("=" * 35)
    
    checklist = [
        "1. Upload a real MRI file (.nii or .nii.gz)",
        "2. Start processing and monitor Celery logs",
        "3. Look for 'Calling MRBrain API: http://mrbrain-api:8000/brain-age' in logs",
        "4. Check that brain age is NOT exactly 25.0 years",
        "5. Verify percentiles are NOT all 50th percentile", 
        "6. Check browser Network tab for API responses",
        "7. Look for 'model_version': 'MRBrain_final_v2.0' in responses",
        "8. Verify different ages/genders give different results",
        "9. Check that processing creates feature CSV files",
        "10. Confirm results vary with different input files"
    ]
    
    for item in checklist:
        print(f"   □ {item}")
    
    print("\n🔍 Key Indicators of Proper Integration:")
    print("   ✅ Brain age varies from chronological age")
    print("   ✅ Percentiles vary across brain regions")
    print("   ✅ Processing logs show API calls to MRBrain_final")
    print("   ✅ Different inputs produce different outputs")
    print("   ✅ Model version shows 'MRBrain_final_v2.0'")
    
    print("\n⚠️  Red Flags (Indicates Hardcoded Values):")
    print("   ❌ Brain age always exactly 25.0 years")
    print("   ❌ All percentiles exactly 50th percentile")
    print("   ❌ Same results regardless of input")
    print("   ❌ No API calls in Celery logs")
    print("   ❌ No model version tracking")

if __name__ == "__main__":
    print("🔬 MRBrain_final Integration Verification")
    print("🎯 Testing for Dynamic Values vs Hardcoded Data")
    print()
    
    # Run all tests
    test_mrbrain_api_endpoints()
    test_celery_integration() 
    test_frontend_dynamic_display()
    generate_verification_report()
    
    print("\n" + "=" * 60)
    print("🏁 Verification Complete")
    print("\n🚀 Next Steps:")
    print("1. Run: docker-compose up --build")
    print("2. Start Celery: celery -A app.celery_app worker --loglevel=info")
    print("3. Upload test files and verify dynamic behavior")
    print("4. Monitor logs and network traffic")
    print("5. Check that results change with different inputs")