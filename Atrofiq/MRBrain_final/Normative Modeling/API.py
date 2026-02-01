#!/usr/bin/env python3
"""
Command-line interface for normative modeling analysis.
Updated to work with participant IDs, metadata, and feature importance data.
Only uses chronological age and gender from metadata - no volumes data.
"""

import os
import re
import json
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def parse_filename(fname: str) -> Tuple[str, str]:
    """Extract (sex, region) from a filename like 'female_left_hippocampus.xlsx'."""
    base = os.path.basename(fname)
    name, ext = os.path.splitext(base)
    if ext.lower() == ".gz":
        name, ext2 = os.path.splitext(name)
    
    # Expect prefixes male_ or female_
    if name.startswith("male_"):
        return ("male", name[len("male_"):])
    if name.startswith("female_"):
        return ("female", name[len("female_"):])
    
    # Fallback: unknown, return the whole as region
    return ("unknown", name)


def scan_folder(base_folder: str) -> Dict[str, List[str]]:
    """Return available regions per sex found in base_folder."""
    regions: Dict[str, set] = {"male": set(), "female": set()}
    
    if not os.path.isdir(base_folder):
        return {"male": [], "female": []}

    for fname in os.listdir(base_folder):
        if not fname.lower().endswith(".xlsx"):
            continue
        sex, region = parse_filename(fname)
        if sex in regions:
            regions[sex].add(region)
    
    # Sort naturally
    def sort_key(x):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", x)]

    return {k: sorted(list(v), key=sort_key) for k, v in regions.items()}


def load_metadata(metadata_file: str) -> Dict:
    """Load metadata.json file."""
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_feature_importance(importance_file: str) -> Dict:
    """Load feature importance JSON file."""
    if not os.path.exists(importance_file):
        raise FileNotFoundError(f"Feature importance file not found: {importance_file}")
    
    with open(importance_file, 'r') as f:
        return json.load(f)


def get_participant_info(metadata: Dict, participant_id: str) -> Tuple[int, str]:
    """Get age and sex for a specific participant ID."""
    try:
        patient_ids = metadata["metadata"]["patient ids"]
        ages = metadata["metadata"]["age"]
        sexes = metadata["metadata"]["Sex"]
        
        if participant_id not in patient_ids:
            raise ValueError(f"Participant ID '{participant_id}' not found in metadata")
        
        idx = patient_ids.index(participant_id)
        age = ages[idx]
        sex = sexes[idx].lower()  # Convert M/F to male/female
        sex = "male" if sex == "m" else "female"
        
        return age, sex
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid metadata format: {e}")


def select_feature_importance_file(age: int, base_path: str) -> str:
    """Select appropriate feature importance file based on age."""
    if age < 40:
        pattern = re.compile(r"before_40_.*\.json$")
    else:
        pattern = re.compile(r"after_40_.*\.json$")

    candidates = [f for f in os.listdir(base_path) if pattern.match(f)]
    if not candidates:
        raise FileNotFoundError(f"No feature importance file found for age group in {base_path}")
    # If multiple, pick the first alphabetically
    importance_file = os.path.join(base_path, sorted(candidates)[0])
    return importance_file


def normalize_region_name(region_name: str) -> str:
    """Normalize region names for matching."""
    # Convert to lowercase and replace spaces with underscores, hyphens with underscores
    normalized = region_name.lower().replace(" ", "_").replace("-", "_")
    return normalized

def match_regions_to_available(feature_regions: List[str], available_regions: List[str]) -> List[str]:
    """Match feature importance regions to available percentile regions using regex and fuzzy matching."""
    matched_regions = []
    
    print(f"DEBUG: Trying to match {len(feature_regions)} feature regions to {len(available_regions)} available regions")
    
    for feature_region in feature_regions:
        print(f"DEBUG: Processing feature region: '{feature_region}'")
        
        # Normalize the feature region name
        normalized_feature = normalize_region_name(feature_region)
        print(f"DEBUG: Normalized to: '{normalized_feature}'")
        
        # Direct match first
        if normalized_feature in available_regions:
            matched_regions.append(normalized_feature)
            print(f"DEBUG: Direct match found: {normalized_feature}")
            continue
        
        # Handle cortical regions (ctx-lh- and ctx-rh- prefixes)
        if "ctx_lh_" in normalized_feature or "ctx-lh-" in feature_region.lower():
            if "left_cerebral_cortex" in available_regions:
                matched_regions.append("left_cerebral_cortex")
                print(f"DEBUG: Mapped ctx-lh region to left_cerebral_cortex")
                continue
        elif "ctx_rh_" in normalized_feature or "ctx-rh-" in feature_region.lower():
            if "right_cerebral_cortex" in available_regions:
                matched_regions.append("right_cerebral_cortex")
                print(f"DEBUG: Mapped ctx-rh region to right_cerebral_cortex")
                continue
        
        # Try fuzzy matching with regex patterns
        best_match = None
        
        # Create search patterns from the feature region
        feature_words = re.findall(r'\w+', feature_region.lower())
        print(f"DEBUG: Feature words: {feature_words}")
        
        for available_region in available_regions:
            available_words = re.findall(r'\w+', available_region.lower())
            
            # Count matching words
            common_words = set(feature_words) & set(available_words)
            if len(common_words) >= 2:  # At least 2 words in common
                best_match = available_region
                print(f"DEBUG: Fuzzy match found: '{feature_region}' -> '{available_region}' (common words: {common_words})")
                break
            elif len(common_words) >= 1 and len(feature_words) <= 2:  # For shorter names, 1 word is enough
                if not best_match:  # Only take first match
                    best_match = available_region
                    print(f"DEBUG: Partial match found: '{feature_region}' -> '{available_region}' (common words: {common_words})")
        
        if best_match:
            matched_regions.append(best_match)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matched = []
    for region in matched_regions:
        if region not in seen:
            seen.add(region)
            unique_matched.append(region)
    
    print(f"DEBUG: Final matched regions: {unique_matched}")
    return unique_matched


def extract_brain_regions(feature_importance: Dict, top_n: int = 10) -> List[str]:
    """Extract top brain regions from feature importance data."""
    try:
        feature_names = feature_importance["feature_names"]
        
        # Get SHAP importance scores (you can modify this to use other methods)
        if "importance_scores" in feature_importance and "shap" in feature_importance["importance_scores"]:
            shap_scores_str = feature_importance["importance_scores"]["shap"]
            # Parse the numpy array string representation
            shap_scores_str = shap_scores_str.strip("[]").replace("\n", " ")
            shap_scores = np.fromstring(shap_scores_str, sep=" ")
        else:
            # Fallback: use all features with equal importance
            shap_scores = np.ones(len(feature_names))
        
        # Filter out non-brain regions (like SEX, CSF)
        brain_regions = []
        brain_scores = []
        
        for i, feature in enumerate(feature_names):
            # Skip demographic features
            if feature.lower() in ["sex", "csf"]:
                continue
            # Clean up region names
            if i < len(shap_scores):
                brain_regions.append(feature)
                brain_scores.append(abs(shap_scores[i]))  # Use absolute importance
        
        # Sort by importance and take top N
        if brain_scores:
            sorted_indices = np.argsort(brain_scores)[::-1]  # Descending order
            top_regions = [brain_regions[i] for i in sorted_indices[:top_n]]
        else:
            top_regions = brain_regions[:top_n]
        
        return top_regions
        
    except Exception as e:
        raise ValueError(f"Error extracting brain regions from feature importance: {e}")


def load_percentiles(base_folder: str, sex: str, region: str) -> pd.DataFrame:
    """Load a percentile table for the given sex & region."""
    candidate = os.path.join(base_folder, f"{sex}_{region}.xlsx")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"File not found: {candidate}")
    
    df = pd.read_excel(candidate)

    # Normalize columns: ensure 'Age' exists and percentile columns are numeric labels
    rename_map = {}
    for c in df.columns:
        if isinstance(c, str) and c.lower().endswith("th") and c[:-2].isdigit():
            rename_map[c] = int(c[:-2])
    
    df = df.rename(columns=rename_map)
    
    # Ensure Age is first column for convenience
    cols = [c for c in ["Age"] + sorted([x for x in df.columns if isinstance(x, int)]) if c in df.columns]
    df = df[cols]
    
    return df


def smooth_series(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if window <= 1:
        return y
    
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="CLI Normative Modeling Tool for Specific Participants")
    
    # Required arguments
    parser.add_argument("--participant-id", "-pid", required=True,
                       help="Participant ID to analyze")
    parser.add_argument("--metadata", "-m", required=True,
                       help="Path to metadata.json file")
    parser.add_argument("--importance-folder", "-if", required=True,
                       help="Folder containing before_40.json and after_40.json files")
    parser.add_argument("--percentiles-folder", "-pf", required=True,
                       help="Base folder containing percentile Excel files")
    
    # Optional arguments
    parser.add_argument("--top-regions", "-tr", type=int, default=10,
                       help="Number of top important brain regions to analyze (default: 10)")
    parser.add_argument("--percentiles", "-p", nargs="+", type=int,
                       default=[1, 5, 10, 25, 50, 75, 90, 95, 99],
                       help="Percentile curves to include (default: 1 5 10 25 50 75 90 95 99)")
    parser.add_argument("--smooth", action="store_true", default=False,
                       help="Apply smoothing to curves")
    parser.add_argument("--smooth-window", type=int, default=5,
                       help="Smoothing window size (default: 5)")
    parser.add_argument("--output", "-o", 
                       help="Output JSON file path (default: stdout)")
    parser.add_argument("--pretty", action="store_true",
                       help="Pretty print JSON output")
    
    args = parser.parse_args()
    
    try:
        # Load metadata
        metadata = load_metadata(args.metadata)
        
        # Get participant info (only age and sex from metadata)
        age, sex = get_participant_info(metadata, args.participant_id)
        
        # Select appropriate feature importance file based on age
        importance_file = select_feature_importance_file(age, args.importance_folder)
        feature_importance = load_feature_importance(importance_file)
        
        # Extract top brain regions from feature importance
        top_regions = extract_brain_regions(feature_importance, args.top_regions)
        
        # Scan available regions in percentiles folder
        available = scan_folder(args.percentiles_folder)
        available_regions = available.get(sex, [])
        
        # Filter top regions to those available in percentiles data using matching
        valid_regions = match_regions_to_available(top_regions, available_regions)
        
        # If no matches found, use all available regions as fallback
        if not valid_regions:
            print(f"Warning: No matching regions found between feature importance and available percentiles")
            print(f"Top regions from feature importance: {top_regions}")
            print(f"Available regions for {sex}: {available_regions}")
            print(f"Fallback: Using all available regions instead")
            valid_regions = available_regions[:args.top_regions]  # Limit to requested number
        
        # Prepare output data
        result = {
            "participant_info": {
                "participant_id": args.participant_id,
                "chronological_age": age,  # This is the key info for plotting the dot
                "sex": sex,
                "age_group": "before_40" if age < 40 else "after_40"
            },
            "analysis_metadata": {
                "importance_file_used": importance_file,
                "percentiles_folder": args.percentiles_folder,
                "top_regions_requested": args.top_regions,
                "valid_regions_found": len(valid_regions),
                "smoothed": args.smooth,
                "smooth_window": args.smooth_window if args.smooth else None
            },
            "region_analyses": {}
        }
        
        # Analyze each valid region - generate percentile curves only
        for region in valid_regions:
            try:
                # Load percentile data for this region
                df_percentiles = load_percentiles(args.percentiles_folder, sex, region)
                available_percentiles = [c for c in df_percentiles.columns if isinstance(c, int)]
                
                # Filter requested percentiles
                selected_pcts = [p for p in args.percentiles if p in available_percentiles]
                
                if not selected_pcts:
                    print(f"Warning: No valid percentiles found for region {region}")
                    continue
                
                # Prepare region analysis - only curves and age marker
                region_analysis = {
                    "region_name": region,
                    "available_percentiles": sorted(available_percentiles),
                    "selected_percentiles": sorted(selected_pcts),
                    "percentile_curves": {}
                }
                
                # Extract age values for x-axis
                ages = df_percentiles["Age"].astype(float).tolist()
                region_analysis["ages"] = ages
                
                # Process each percentile curve
                for pct in sorted(selected_pcts):
                    values = df_percentiles[pct].astype(float).values
                    
                    if args.smooth:
                        values = smooth_series(values, args.smooth_window)
                    
                    region_analysis["percentile_curves"][str(pct)] = values.tolist()
                
                result["region_analyses"][region] = region_analysis
                
            except Exception as e:
                print(f"Warning: Error analyzing region {region}: {e}")
                continue
        
        # Add summary statistics
        if result["region_analyses"]:
            result["summary"] = {
                "regions_analyzed": len(result["region_analyses"]),
                "participant_chronological_age": age,
                "age_group": "before_40" if age < 40 else "after_40",
                "total_regions_available": len(available_regions),
                "regions_with_percentile_data": len(valid_regions)
            }
        
        # Output JSON
        json_output = json.dumps(result, indent=2 if args.pretty else None)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"Results saved to {args.output}")
        else:
            print(json_output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())