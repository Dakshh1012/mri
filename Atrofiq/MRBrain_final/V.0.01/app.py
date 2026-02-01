import os
import sys
import glob
import time
import json
import shutil
import base64
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory

# Add parent dir to path if strictly needed (but we are in V.0.01 now)
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'V.0.01'))
# Now we can just import Pipeline (it's in the same dir)
from Pipeline import MRPipeline

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pydicom

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

# Ensure dirs
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global State (Simple for V1)
# Note: For multi-user, use a database or session-based storage
CURRENT_SESSION = {
    'input_path': None,
    'results': None,
    'pipeline': None,
    'logs': [],
    'status': 'idle'
}

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_session():
    global CURRENT_SESSION
    CURRENT_SESSION = {
        'input_path': None,
        'results': None,
        'pipeline': None,
        'logs': [],
        'status': 'idle'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    reset_session()
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
        
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No selected file'}), 400

    #Create unique upload session dir
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(upload_dir, exist_ok=True)

    saved_paths = []
    
    # Handle files
    for file in files:
        if file.filename:
            filename = file.filename
            # If folder structure, keep it or flatten? 
            # Flattening is safer for simple DICOM series
            name = os.path.basename(filename)
            save_path = os.path.join(upload_dir, name)
            file.save(save_path)
            saved_paths.append(save_path)

    CURRENT_SESSION['input_path'] = upload_dir
    CURRENT_SESSION['status'] = 'uploaded'
    
    # Auto-detect Metadata
    meta = {'age': None, 'sex': None}
    
    # Check for DICOMs
    dcm_files = [f for f in saved_paths if f.lower().endswith(('.dcm', '.ima'))]
    if not dcm_files and len(saved_paths) == 1 and saved_paths[0].lower().endswith('.zip'):
        # Handle ZIP extract logic later if needed
        pass
    elif dcm_files:
        try:
            dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            age_str = getattr(dcm, 'PatientAge', '')
            if age_str:
                if 'Y' in age_str:
                    meta['age'] = float(age_str.replace('Y', ''))
                elif 'M' in age_str:
                    meta['age'] = float(age_str.replace('M', '')) / 12.0
                else:
                    try: 
                        meta['age'] = float(age_str)
                    except: 
                        pass
            
            sex_str = getattr(dcm, 'PatientSex', '')
            if sex_str:
                if sex_str.upper() in ['M', 'MALE']:
                    meta['sex'] = 'Male'
                elif sex_str.upper() in ['F', 'FEMALE']:
                    meta['sex'] = 'Female'
        except:
            pass
            
    return jsonify({'message': 'Upload successful', 'count': len(files), 'metadata': meta})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    age = float(data.get('age', 0))
    sex = data.get('sex', 'Female')
    force_contrast = data.get('force_contrast', False)
    
    if not CURRENT_SESSION['input_path']:
        return jsonify({'error': 'No input data uploaded'}), 400

    CURRENT_SESSION['status'] = 'running'
    CURRENT_SESSION['logs'] = []
    
    def log_callback(msg):
        # We can implement a streaming log or polling log
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        CURRENT_SESSION['logs'].append(entry)
        print(entry)

    def run_pipeline_thread():
        try:
            # Initialize Pipeline
            pipeline = MRPipeline()
            
            # Monkey Patch print to capture logs
            import builtins
            original_print = builtins.print
            def custom_print(*args, **kwargs):
                msg = " ".join(map(str, args))
                CURRENT_SESSION['logs'].append(msg)
                original_print(*args, **kwargs)
            builtins.print = custom_print
            
            try:
                out_dir = os.path.join(app.config['RESULTS_FOLDER'], os.path.basename(CURRENT_SESSION['input_path']))
                
                results = pipeline.run(
                    CURRENT_SESSION['input_path'],
                    age,
                    sex,
                    output_dir=out_dir,
                    force_contrast=force_contrast
                )
                
                CURRENT_SESSION['results'] = results
                CURRENT_SESSION['status'] = 'completed' if results else 'failed'
                
            except Exception as e:
                CURRENT_SESSION['status'] = 'error'
                CURRENT_SESSION['logs'].append(f"ERROR: {str(e)}")
                traceback.print_exc()
            finally:
                builtins.print = original_print
                
        except Exception as e:
             CURRENT_SESSION['status'] = 'error'
             logging.error(e)

    thread = threading.Thread(target=run_pipeline_thread)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/status')
def status():
    return jsonify({
        'status': CURRENT_SESSION['status'],
        'logs': CURRENT_SESSION['logs'][-20:] # Return last 20 logs
    })

@app.route('/results_data')
def results_data():
    if CURRENT_SESSION['results']:
        res = CURRENT_SESSION['results']
        
        # Calculate Best Slices (Max area)
        try:
            vol_path = res.get('processed_volume')
            if vol_path and os.path.exists(vol_path):
                # SMART AXIAL LOGIC (Replicating Training Pipeline)
                img = nib.load(vol_path)
                img = nib.as_closest_canonical(img) # Force RAS
                data = img.get_fdata()
                
                # Transpose to (Z, Y, X) - Axial First
                # Z = Superior-Inferior (Axial)
                # Y = Anterior-Posterior
                # X = Right-Left
                data_axial = np.transpose(data, (2, 1, 0))
                
                # Heuristic: Slice with max non-zero pixels (Threshold to ignore noise)
                threshold = np.percentile(data_axial, 90) * 0.1 
                mask = data_axial > threshold
                
                # Best Axial: Max sum along dim 1 (Y) and 2 (X) - i.e., determining best Z plane
                axial_sums = np.sum(mask, axis=(1, 2))
                best_axial = int(np.argmax(axial_sums))
                
                res['best_slices'] = {
                    'axial': best_axial
                }
        except Exception as e:
            logger.error(f"Error calculating best slices: {e}")
            res['best_slices'] = {'axial': 50}

        # Dynamic Regions List
        try:
            vol_path = res.get('volumes_csv')
            if vol_path and os.path.exists(vol_path):
                df = pd.read_csv(vol_path)
                # Exclude non-region columns
                ignore = ['subject', 'id', 'filename', 'total intracranial']
                cols = [c for c in df.columns if c.lower() not in ignore]
                res['available_regions'] = cols
        except:
             res['available_regions'] = ['left_hippocampus', 'right_hippocampus']

        return jsonify(res)
    return jsonify({'error': 'No results'}), 404

@app.route('/slice/<axis>/<int:index>')
def get_slice(axis, index):
    try:
        res = CURRENT_SESSION['results']
        if not res or not res.get('processed_volume'):
            return "No volume", 404
            
        vol_path = res.get('processed_volume')
        
        # SMART AXIAL LOGIC
        img = nib.load(vol_path)
        img = nib.as_closest_canonical(img) # 1. Force RAS
        data = img.get_fdata()
        data_axial = np.transpose(data, (2, 1, 0)) # 2. Transpose to Z, Y, X
        
        # Normalize
        data_axial = np.nan_to_num(data_axial)
        vmin, vmax = np.percentile(data_axial, [1, 99])
        data_axial = np.clip(data_axial, vmin, vmax)
        data_axial = (data_axial - vmin) / (vmax - vmin)
        
        slice_img = None
        
        if axis == 'axial':
            idx = min(index, data_axial.shape[0]-1)
            # data_axial[idx] is (Y, X). 
            # Y = P->A (rows 0..N). X = L->R (cols 0..N).
            # Image Origin (0,0) is Top-Left.
            # Row 0 (Posterior) at Top. 
            # We want Anterior at Top (Standard). Max row index = Anterior.
            # So we FLIP vertically (flip axis 0).
            raw_slice = data_axial[idx, :, :]
            slice_img = np.flipud(raw_slice)
            
            # Note: X axis. L->R.
            # Col 0 = Left. 
            # Radiologists often like L on R (flip output).
            # But standard viewing is often Left on Left unless specified.
            # We'll stick to neurological (Left on Left) for now unless asked.
            
        # We only support axial for now based on UI
        
        if slice_img is None:
             raise ValueError("Slice generation failed")
            
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(slice_img, cmap='gray')
        ax.axis('off')
        
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Slice error: {e}")
        return "Error", 500

@app.route('/normative_plot/<region>')
def normative_plot(region):
    try:
        if not CURRENT_SESSION.get('results'):
             return jsonify({'error': "No active results"}), 404
             
        res = CURRENT_SESSION['results']
        vol_csv = res.get('volumes_csv')
        age = res.get('chronological_age')
        sex = res.get('gender')
        is_contrast = res.get('is_contrast')
        
        logger.info(f"Data Request: {region}, Age:{age}, Sex:{sex}")

        pipeline = MRPipeline()
        contrast_mode = "Post" if is_contrast else "Pre"
        df_norm = pipeline.get_normative_data(region, sex, contrast_mode=contrast_mode)
        
        if df_norm is None:
            return jsonify({'error': f"No normative data for {region}"}), 404
            
        val = 0
        icv = 1.0
        
        if vol_csv and os.path.exists(vol_csv):
            df_vol = pd.read_csv(vol_csv)
            # Robust Matching
            target = region.lower().replace('-', '').replace('_', '').replace(' ', '')
            
            match_col = None
            for c in df_vol.columns:
                c_clean = c.lower().replace('-', '').replace('_', '').replace(' ', '')
                if c_clean == target:
                    match_col = c
                    break
            
            if match_col:
                val = df_vol[match_col].values[0]
            
            if 'total intracranial' in df_vol.columns:
                 icv = df_vol['total intracranial'].values[0]
        
        # Prepare Data for Chart.js
        # Filter range
        df_norm = df_norm[(df_norm['Age'] >= 21) & (df_norm['Age'] <= 100)].copy()
        
        # Scale
        centile_cols = [c for c in df_norm.columns if c.startswith('Centile_')]
        if icv > 0:
            for c in centile_cols:
                 df_norm[c] = df_norm[c] * icv
                 
        # Format Response
        response = {
            'ages': df_norm['Age'].tolist(),
            'centiles': {},
            'subject': {
                'age': age,
                'volume': val
            }
        }
        
        for c in centile_cols:
            centile_key = c.split('_')[1] # e.g. "50"
            response['centiles'][centile_key] = df_norm[c].tolist()
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Data error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
