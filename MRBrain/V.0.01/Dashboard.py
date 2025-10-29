import streamlit as st
import pandas as pd
import json
import subprocess
import sys
import os
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import tempfile
import shutil

# Page config
st.set_page_config(
    page_title="Customizable MRI Processing Pipeline v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .step-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        font-weight: bold;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        color: #0d47a1;
        margin: 1rem 0;
    }
    .participant-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .matched {
        border-color: #28a745;
        background-color: #d4edda;
    }
    .unmatched {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
</style>
""", unsafe_allow_html=True)

class CustomizableStreamlitInterface:
    def __init__(self):
        self.base_dir = Path(os.path.abspath('V.0.01'))
        self.pipeline_script = self.base_dir / "Pipeline.py"
        
        # Initialize session state variables
        if 'participants_df' not in st.session_state:
            st.session_state.participants_df = None
        if 'nii_files' not in st.session_state:
            st.session_state.nii_files = []
        if 'matched_participants' not in st.session_state:
            st.session_state.matched_participants = {}
        if 'selected_participants' not in st.session_state:
            st.session_state.selected_participants = []
        
    def scan_nii_files(self, directory):
        """Scan directory for NII files"""
        if not directory or not Path(directory).exists():
            return []
        
        nii_files = []
        for pattern in ["*.nii", "*.nii.gz"]:
            nii_files.extend(Path(directory).glob(pattern))
        
        return sorted([f.name for f in nii_files])
    
    def parse_uploaded_csv(self, uploaded_file, separator='\t', has_header=False):
        """Parse uploaded CSV file with flexible options"""
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file, sep=separator, header=0 if has_header else None)
            
            # If no header, assign generic column names
            if not has_header:
                if len(df.columns) >= 3:
                    df.columns = ['participant_id', 'age', 'sex'] + [f'col_{i}' for i in range(3, len(df.columns))]
                else:
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
            
            return df, None
        except Exception as e:
            return None, str(e)
    
    def create_custom_metadata(self, selected_data, output_path):
        """Create metadata JSON from selected participants"""
        try:
            metadata = {
                "metadata": {
                    "patient ids": [row['participant_id'] for _, row in selected_data.iterrows()],
                    "ages": [row['age'] for _, row in selected_data.iterrows()],
                    "sexes": [row['sex'] for _, row in selected_data.iterrows()],
                    "total_participants": len(selected_data),
                    "created_date": datetime.now().isoformat(),
                    "custom_selection": True
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def create_custom_csv(self, selected_data, output_path):
        """Create CSV file from selected participants"""
        try:
            # Create CSV in expected format (tab-separated, no header)
            output_df = selected_data[['participant_id', 'age', 'sex']].copy()
            output_df.to_csv(output_path, sep='\t', index=False)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def copy_selected_nii_files(self, selected_participants, source_dir, target_dir):
        """Copy only selected NII files to target directory"""
        try:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            copied_files = []
            errors = []
            
            for participant_id in selected_participants:
                if participant_id in st.session_state.matched_participants:
                    nii_file = st.session_state.matched_participants[participant_id]
                    source_file = Path(source_dir) / nii_file
                    target_file = target_path / nii_file
                    
                    if source_file.exists():
                        shutil.copy2(source_file, target_file)
                        copied_files.append(nii_file)
                    else:
                        errors.append(f"Source file not found: {nii_file}")
                else:
                    errors.append(f"No NII file matched for participant: {participant_id}")
            
            return copied_files, errors
        except Exception as e:
            return [], [str(e)]
    
    def run_pipeline(self, command, progress_bar, status_text):
        """Run pipeline command and update progress"""
        try:
            status_text.text("🚀 Starting pipeline execution...")
            progress_bar.progress(0.1)
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            output_lines = []
            step_progress = {
                "METADATA GENERATION": 0.2,
                "SEGMENTATION": 0.4,
                "BRAIN AGE PREDICTION": 0.6,
                "NORMATIVE MODELING": 0.8
            }
            
            current_progress = 0.1
            
            for line in process.stdout:
                output_lines.append(line.strip())
                
                for step, progress in step_progress.items():
                    if step in line and progress > current_progress:
                        current_progress = progress
                        progress_bar.progress(current_progress)
                        status_text.text(f"🔄 {step.title()}...")
                        break
                
                if "completed successfully" in line.lower():
                    if current_progress < 0.9:
                        current_progress += 0.1
                        progress_bar.progress(min(current_progress, 0.9))
            
            return_code = process.wait()
            
            if return_code == 0:
                progress_bar.progress(1.0)
                status_text.text("✅ Pipeline completed successfully!")
                return True, output_lines
            else:
                status_text.text("❌ Pipeline failed!")
                return False, output_lines
                
        except Exception as e:
            status_text.text(f"❌ Error running pipeline: {e}")
            return False, [str(e)]

def main():
    st.markdown('<div class="main-header">🧠 Fully Customizable MRI Processing Pipeline</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Customizable Features:</strong><br>
    📋 Upload your own CSV files with any format<br>
    🔗 Manually match participants with NII files<br>
    ✅ Select exactly which participants to process<br>
    🎯 Run pipeline on your custom selection only
    </div>
    """, unsafe_allow_html=True)
    
    interface = CustomizableStreamlitInterface()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Setup", 
        "🔗 Match Participants", 
        "✅ Select & Configure", 
        "🚀 Run Pipeline", 
        "📊 Results"
    ])
    
    with tab1:
        st.header("📋 Data Setup")
        
        # CSV Upload Section
        st.subheader("1. Upload Participant Data CSV")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_csv = st.file_uploader(
                "Upload CSV file", 
                type=['csv', 'txt', 'tsv'],
                help="Upload a CSV/TSV file with participant information"
            )
        
        with col2:
            separator = st.selectbox("Separator", ['\t', ',', ';', ' '], index=0)
            has_header = st.checkbox("File has header row", value=False)
        
        if uploaded_csv:
            df, error = interface.parse_uploaded_csv(uploaded_csv, separator, has_header)
            
            if error:
                st.error(f"Error reading CSV: {error}")
            else:
                st.success(f"✅ CSV loaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column mapping
                st.subheader("Column Mapping")
                col_map1, col_map2, col_map3 = st.columns(3)
                
                with col_map1:
                    participant_col = st.selectbox(
                        "Participant ID Column", 
                        df.columns.tolist(),
                        help="Column containing participant/subject IDs"
                    )
                
                with col_map2:
                    age_col = st.selectbox(
                        "Age Column", 
                        df.columns.tolist(),
                        index=1 if len(df.columns) > 1 else 0,
                        help="Column containing participant ages"
                    )
                
                with col_map3:
                    sex_col = st.selectbox(
                        "Sex/Gender Column", 
                        df.columns.tolist(),
                        index=2 if len(df.columns) > 2 else 0,
                        help="Column containing participant sex/gender"
                    )
                
                if st.button("✅ Confirm Column Mapping"):
                    # Create standardized dataframe
                    participants_df = pd.DataFrame({
                        'participant_id': df[participant_col].astype(str),
                        'age': pd.to_numeric(df[age_col], errors='coerce'),
                        'sex': df[sex_col].astype(str)
                    })
                    
                    # Remove rows with missing data
                    participants_df = participants_df.dropna()
                    
                    st.session_state.participants_df = participants_df
                    st.success(f"✅ Participant data prepared! {len(participants_df)} valid participants found.")
                    
                    # Show final data
                    st.dataframe(participants_df, use_container_width=True)
        
        # MRI Directory Section
        st.subheader("2. MRI Data Directory")
        mri_directory = st.text_input(
            "MRI Directory Path",
            value="../SynthSeg/Post-contrast-Data/",
            help="Directory containing .nii or .nii.gz files"
        )
        
        if st.button("🔍 Scan for NII Files"):
            if mri_directory:
                nii_files = interface.scan_nii_files(mri_directory)
                st.session_state.nii_files = nii_files
                st.session_state.mri_directory = mri_directory
                
                if nii_files:
                    st.success(f"✅ Found {len(nii_files)} NII files")
                    
                    # Show files in expandable section
                    with st.expander("View NII Files", expanded=False):
                        for i, file in enumerate(nii_files):
                            st.write(f"{i+1}. {file}")
                else:
                    st.error("❌ No NII files found in the specified directory")
    
    with tab2:
        st.header("🔗 Match Participants with NII Files")
        
        if st.session_state.participants_df is None:
            st.warning("⚠️ Please upload and configure participant data in the 'Data Setup' tab first.")
        elif not st.session_state.nii_files:
            st.warning("⚠️ Please scan for NII files in the 'Data Setup' tab first.")
        else:
            st.subheader("Participant-NII File Matching")
            
            participants = st.session_state.participants_df['participant_id'].tolist()
            nii_files = st.session_state.nii_files
            
            st.info(f"📊 {len(participants)} participants | 📁 {len(nii_files)} NII files")
            
            # Auto-matching options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Auto-Match (Exact Names)"):
                    matched_count = 0
                    for participant in participants:
                        # Try exact match
                        for nii_file in nii_files:
                            if participant in nii_file:
                                st.session_state.matched_participants[participant] = nii_file
                                matched_count += 1
                                break
                    
                    st.success(f"✅ Auto-matched {matched_count} participants")
            
            with col2:
                if st.button("🔄 Clear All Matches"):
                    st.session_state.matched_participants = {}
                    st.success("✅ All matches cleared")
            
            # Manual matching interface
            st.subheader("Manual Matching")
            
            # Search functionality
            search_participant = st.text_input("🔍 Search Participant ID", "")
            
            # Filter participants based on search
            if search_participant:
                filtered_participants = [p for p in participants if search_participant.lower() in p.lower()]
            else:
                filtered_participants = participants
            
            # Pagination for large datasets
            participants_per_page = 10
            total_pages = (len(filtered_participants) + participants_per_page - 1) // participants_per_page
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1))
                start_idx = (page - 1) * participants_per_page
                end_idx = min(start_idx + participants_per_page, len(filtered_participants))
                page_participants = filtered_participants[start_idx:end_idx]
            else:
                page_participants = filtered_participants
            
            # Display participants for matching
            for participant in page_participants:
                participant_data = st.session_state.participants_df[
                    st.session_state.participants_df['participant_id'] == participant
                ].iloc[0]
                
                # Check if already matched
                is_matched = participant in st.session_state.matched_participants
                card_class = "matched" if is_matched else "unmatched"
                
                with st.container():
                    st.markdown(f'<div class="participant-card {card_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**{participant}**")
                        st.write(f"Age: {participant_data['age']} | Sex: {participant_data['sex']}")
                    
                    with col2:
                        current_match = st.session_state.matched_participants.get(participant, "")
                        
                        # Add "No match" option
                        nii_options = [""] + nii_files
                        current_index = 0
                        if current_match in nii_files:
                            current_index = nii_files.index(current_match) + 1
                        
                        selected_nii = st.selectbox(
                            "Match with NII file",
                            nii_options,
                            index=current_index,
                            key=f"match_{participant}",
                            format_func=lambda x: "No match selected" if x == "" else x
                        )
                        
                        if selected_nii:
                            st.session_state.matched_participants[participant] = selected_nii
                        elif participant in st.session_state.matched_participants:
                            del st.session_state.matched_participants[participant]
                    
                    with col3:
                        status = "✅ Matched" if is_matched else "❌ Not matched"
                        st.write(status)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Matching summary
            st.subheader("Matching Summary")
            matched_count = len(st.session_state.matched_participants)
            total_participants = len(participants)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Participants", total_participants)
            with col2:
                st.metric("Matched", matched_count)
            with col3:
                st.metric("Match Rate", f"{matched_count/total_participants*100:.1f}%" if total_participants > 0 else "0%")
    
    with tab3:
        st.header("✅ Select Participants & Configure Pipeline")
        
        if not st.session_state.matched_participants:
            st.warning("⚠️ Please match participants with NII files in the 'Match Participants' tab first.")
        else:
            # Participant selection
            st.subheader("Select Participants to Process")
            
            matched_participants = list(st.session_state.matched_participants.keys())
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # Multi-select for participants
                selected_participants = st.multiselect(
                    "Choose participants to process",
                    matched_participants,
                    default=st.session_state.selected_participants,
                    help="Select the participants you want to include in the pipeline"
                )
                st.session_state.selected_participants = selected_participants
            
            with col2:
                if st.button("Select All Matched"):
                    st.session_state.selected_participants = matched_participants
                    st.rerun()
                
                if st.button("Clear Selection"):
                    st.session_state.selected_participants = []
                    st.rerun()
            
            if selected_participants:
                st.success(f"✅ {len(selected_participants)} participants selected for processing")
                
                # Show selected participants details
                with st.expander("View Selected Participants", expanded=False):
                    selected_df = st.session_state.participants_df[
                        st.session_state.participants_df['participant_id'].isin(selected_participants)
                    ].copy()
                    selected_df['nii_file'] = selected_df['participant_id'].map(st.session_state.matched_participants)
                    st.dataframe(selected_df, use_container_width=True)
                
                # Pipeline configuration
                st.subheader("Pipeline Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    output_dir = st.text_input("Output Directory", value="Custom_Pipeline_Results")
                    threads = st.number_input("Number of Threads", min_value=1, max_value=100, value=90)
                
                with col2:
                    run_parc = st.checkbox("Enable Parcellation", value=True)
                    run_brainage = st.checkbox("Run Brain Age Prediction", value=True)
                    run_normative = st.checkbox("Run Normative Modeling", value=True)
                
                if run_normative:
                    percentiles_str = st.text_input(
                        "Percentiles (space-separated)",
                        value="25 50 75",
                        help="E.g., '25 50 75' for 25th, 50th, 75th percentiles"
                    )
                    percentiles = [int(x) for x in percentiles_str.split() if x.isdigit()]
                else:
                    percentiles = [25, 50, 75]
                
                # Pipeline steps selection
                st.subheader("Pipeline Steps")
                step_options = {
                    "Full Pipeline": "full",
                    "Metadata Only": "metadata",
                    "Segmentation Only": "segmentation", 
                    "Brain Age Only": "brainage",
                    "Normative Only": "normative"
                }
                selected_step = st.selectbox("Select Pipeline Steps", list(step_options.keys()))
                
                # Store configuration in session state
                st.session_state.pipeline_config = {
                    'output_dir': output_dir,
                    'threads': threads,
                    'run_parc': run_parc,
                    'run_brainage': run_brainage,
                    'run_normative': run_normative,
                    'percentiles': percentiles,
                    'selected_step': selected_step
                }
                
                st.success("✅ Configuration saved!")
            else:
                st.info("👆 Please select at least one participant to process")
    
    with tab4:
        st.header("🚀 Run Custom Pipeline")
        
        if not st.session_state.selected_participants:
            st.warning("⚠️ Please select participants to process in the 'Select & Configure' tab first.")
        else:
            config = st.session_state.pipeline_config
            selected_participants = st.session_state.selected_participants
            
            # Display configuration summary
            st.subheader("Pipeline Configuration Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Selected Participants:** {len(selected_participants)}  
                **Output Directory:** {config['output_dir']}  
                **Threads:** {config['threads']}  
                **Pipeline Steps:** {config['selected_step']}
                """)
            
            with col2:
                st.info(f"""
                **Parcellation:** {'✅' if config['run_parc'] else '❌'}  
                **Brain Age:** {'✅' if config['run_brainage'] else '❌'}  
                **Normative:** {'✅' if config['run_normative'] else '❌'}  
                **Percentiles:** {config['percentiles']}
                """)
            
            # Prepare custom files
            if st.button("🔧 Prepare Custom Files", type="secondary"):
                with st.spinner("Preparing custom files..."):
                    # Create temporary directory for custom files
                    temp_dir = Path("temp_custom_pipeline")
                    temp_dir.mkdir(exist_ok=True)
                    
                    # Create custom MRI directory and copy selected files
                    custom_mri_dir = temp_dir / "custom_mri_files"
                    if hasattr(st.session_state, 'mri_directory'):
                        copied_files, errors = interface.copy_selected_nii_files(
                            selected_participants, 
                            st.session_state.mri_directory, 
                            custom_mri_dir
                        )
                        
                        if errors:
                            st.error("❌ Errors copying files:")
                            for error in errors:
                                st.write(f"• {error}")
                        else:
                            st.success(f"✅ Copied {len(copied_files)} NII files")
                    
                    # Create custom CSV
                    selected_df = st.session_state.participants_df[
                        st.session_state.participants_df['participant_id'].isin(selected_participants)
                    ]
                    
                    custom_csv = temp_dir / "custom_participants.csv"
                    csv_success, csv_error = interface.create_custom_csv(selected_df, custom_csv)
                    
                    if csv_success:
                        st.success(f"✅ Created custom CSV with {len(selected_df)} participants")
                    else:
                        st.error(f"❌ Error creating CSV: {csv_error}")
                    
                    # Store paths in session state
                    st.session_state.custom_files = {
                        'mri_dir': str(custom_mri_dir),
                        'csv_file': str(custom_csv),
                        'temp_dir': str(temp_dir)
                    }
            
            # Run pipeline
            if hasattr(st.session_state, 'custom_files'):
                st.subheader("Execute Custom Pipeline")
                
                if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
                    custom_files = st.session_state.custom_files
                    
                    # Build command
                    cmd = [sys.executable, str(interface.pipeline_script)]
                    
                    if config['selected_step'] == "Full Pipeline":
                        cmd.extend([
                            "--mri_dir", custom_files['mri_dir'],
                            "--csv", custom_files['csv_file'],
                            "--output_dir", config['output_dir'],
                            "--threads", str(config['threads'])
                        ])
                        if not config['run_parc']:
                            cmd.append("--no_parc")
                        if not config['run_brainage']:
                            cmd.append("--no_brainage")
                        if not config['run_normative']:
                            cmd.append("--no_normative")
                        if config['run_normative'] and config['percentiles']:
                            cmd.extend(["--percentiles"] + [str(p) for p in config['percentiles']])
                    
                    # Show command
                    st.code(" ".join(cmd))
                    
                    # Execute pipeline
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    success, output_lines = interface.run_pipeline(cmd, progress_bar, status_text)
                    
                    if success:
                        st.balloons()
                        st.markdown('<div class="success-message">', unsafe_allow_html=True)
                        st.write("🎉 **Custom pipeline completed successfully!**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Clean up temporary files
                        if st.button("🗑️ Clean Up Temporary Files"):
                            try:
                                shutil.rmtree(st.session_state.custom_files['temp_dir'])
                                st.success("✅ Temporary files cleaned up")
                                del st.session_state.custom_files
                            except Exception as e:
                                st.error(f"Error cleaning up: {e}")
                    else:
                        st.markdown('<div class="error-message">', unsafe_allow_html=True)
                        st.write("❌ **Pipeline failed!**")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store output in session state for results tab
                    st.session_state.pipeline_output = output_lines
            else:
                st.info("👆 Please prepare custom files first")
    
    with tab5:
        st.header("📊 Pipeline Results")
        
        if hasattr(st.session_state, 'pipeline_config'):
            results_dir = st.session_state.pipeline_config['output_dir']
            
            if Path(results_dir).exists():
                st.subheader(f"Results from: {results_dir}")
                
                # Show result files
                result_files = {
                    "Metadata": Path(results_dir) / "metadata.json",
                    "Volumes": Path(results_dir) / "volumes.csv",
                    "QC Scores": Path(results_dir) / "qc_scores.csv",
                    "Brain Age Results": Path(results_dir) / "brain_age_results.csv",
                    "Normative Results": Path(results_dir) / "normative_results"
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📁 Generated Files")
                    for name, file_path in result_files.items():
                        if file_path.exists():
                            if file_path.is_file():
                                st.success(f"✅ {name}: {file_path.name}")
                            else:
                                # Directory
                                file_count = len(list(file_path.glob("*")))
                                st.success(f"✅ {name}: {file_count} files")
                        else:
                            st.error(f"❌ {name}: Not found")
                
                with col2:
                    st.subheader("📊 Custom Results Stats")
                    
                    # Show selected participants count
                    if st.session_state.selected_participants:
                        st.metric("Processed Participants", len(st.session_state.selected_participants))
                    
                    # Show volumes data if available
                    volumes_file = result_files["Volumes"]
                    if volumes_file.exists():
                        try:
                            df = pd.read_csv(volumes_file)
                            st.metric("Volume Measurements", len(df))
                        except:
                            st.warning("Could not read volumes file")
                
                # Display sample results
                volumes_file = result_files["Volumes"]
                if volumes_file.exists():
                    st.subheader("📊 Volumes Data Preview")
                    try:
                        df = pd.read_csv(volumes_file)
                        
                        # Filter to show only selected participants
                        if st.session_state.selected_participants:
                            # Try to match participant IDs in the volumes data
                            participant_cols = [col for col in df.columns if 'id' in col.lower() or 'subject' in col.lower() or 'participant' in col.lower()]
                            if participant_cols:
                                participant_col = participant_cols[0]
                                df_filtered = df[df[participant_col].astype(str).isin(st.session_state.selected_participants)]
                                if not df_filtered.empty:
                                    st.write(f"Showing results for {len(df_filtered)} selected participants:")
                                    st.dataframe(df_filtered.head(10), use_container_width=True)
                                else:
                                    st.dataframe(df.head(10), use_container_width=True)
                            else:
                                st.dataframe(df.head(10), use_container_width=True)
                        else:
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Create visualization for selected participants
                        if len(df.columns) > 1:
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                            if len(numeric_cols) > 0:
                                selected_col = st.selectbox("Select column for visualization", numeric_cols)
                                
                                fig = px.histogram(
                                    df, 
                                    x=selected_col, 
                                    title=f"Distribution of {selected_col} (Custom Selection)",
                                    color_discrete_sequence=['#1f77b4']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Error reading volumes file: {e}")
                
                # Display normative modeling results for selected participants
                normative_dir = result_files["Normative Results"]
                if normative_dir.exists() and normative_dir.is_dir():
                    st.subheader("📐 Normative Modeling Results")
                    
                    # Show summary
                    summary_file = normative_dir / "normative_modeling_summary.json"
                    if summary_file.exists():
                        try:
                            with open(summary_file) as f:
                                summary = json.load(f)
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Processed", summary.get("total_participants", 0))
                            with col_b:
                                st.metric("Successful", len(summary.get("successful_participants", [])))
                            with col_c:
                                st.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
                            
                            # Show successful participants
                            if summary.get("successful_participants"):
                                st.write("**Successfully processed participants:**")
                                success_df = pd.DataFrame({
                                    'Participant ID': summary["successful_participants"]
                                })
                                st.dataframe(success_df, use_container_width=True)
                            
                            # Show failed participants if any
                            if summary.get("failed_participants"):
                                with st.expander("Failed Participants", expanded=False):
                                    fail_df = pd.DataFrame({
                                        'Participant ID': summary["failed_participants"]
                                    })
                                    st.dataframe(fail_df, use_container_width=True)
                                    
                        except Exception as e:
                            st.error(f"Error reading normative summary: {e}")
                    
                    # Show individual participant results
                    result_files_list = list(normative_dir.glob("*_normative_results.json"))
                    if result_files_list:
                        st.write(f"📊 {len(result_files_list)} individual participant result files generated")
                        
                        # Sample result viewer
                        selected_participant_file = st.selectbox(
                            "View results for participant:",
                            [f.stem.replace("_normative_results", "") for f in result_files_list],
                            key="participant_result_viewer"
                        )
                        
                        if selected_participant_file:
                            result_file = normative_dir / f"{selected_participant_file}_normative_results.json"
                            if result_file.exists():
                                try:
                                    with open(result_file) as f:
                                        participant_data = json.load(f)
                                    
                                    st.write(f"**Results for {selected_participant_file}:**")
                                    
                                    # Display in a nice format
                                    if isinstance(participant_data, dict):
                                        # Create metrics if percentile data exists
                                        if 'percentiles' in participant_data:
                                            percentile_cols = st.columns(len(participant_data['percentiles']))
                                            for i, (percentile, value) in enumerate(participant_data['percentiles'].items()):
                                                with percentile_cols[i]:
                                                    st.metric(f"{percentile}th Percentile", f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                                        
                                        # Show full JSON in expander
                                        with st.expander("View Full Results JSON", expanded=False):
                                            st.json(participant_data)
                                    else:
                                        st.json(participant_data)
                                        
                                except Exception as e:
                                    st.error(f"Error reading participant results: {e}")
            else:
                st.info("No results found yet. Run the pipeline first.")
        
        # Display pipeline logs
        if hasattr(st.session_state, 'pipeline_output'):
            st.subheader("📝 Latest Pipeline Output")
            log_text = "\n".join(st.session_state.pipeline_output[-100:])  # Show last 100 lines
            st.text_area("Pipeline Logs", log_text, height=300)
        
        # Download results button
        if hasattr(st.session_state, 'pipeline_config'):
            results_dir = st.session_state.pipeline_config['output_dir']
            if Path(results_dir).exists():
                st.subheader("💾 Download Results")
                
                if st.button("📦 Create Results Package"):
                    with st.spinner("Creating downloadable package..."):
                        try:
                            # Create a zip file with all results
                            import zipfile
                            
                            zip_path = f"{results_dir}_results.zip"
                            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for root, dirs, files in os.walk(results_dir):
                                    for file in files:
                                        file_path = Path(root) / file
                                        arcname = file_path.relative_to(Path(results_dir))
                                        zipf.write(file_path, arcname)
                            
                            st.success(f"✅ Results package created: {zip_path}")
                            
                            # Provide download link
                            with open(zip_path, "rb") as file:
                                st.download_button(
                                    label="⬇️ Download Results Package",
                                    data=file,
                                    file_name=f"{results_dir}_results.zip",
                                    mime="application/zip"
                                )
                                
                        except Exception as e:
                            st.error(f"Error creating results package: {e}")

    # Sidebar with current status
    st.sidebar.header("📊 Current Status")
    
    # Data upload status
    if st.session_state.participants_df is not None:
        st.sidebar.success(f"✅ CSV Data: {len(st.session_state.participants_df)} participants")
    else:
        st.sidebar.error("❌ No CSV data loaded")
    
    # NII files status
    if st.session_state.nii_files:
        st.sidebar.success(f"✅ NII Files: {len(st.session_state.nii_files)} found")
    else:
        st.sidebar.error("❌ No NII files scanned")
    
    # Matching status
    matched_count = len(st.session_state.matched_participants)
    if matched_count > 0:
        st.sidebar.success(f"✅ Matched: {matched_count} participants")
    else:
        st.sidebar.error("❌ No participants matched")
    
    # Selection status
    selected_count = len(st.session_state.selected_participants)
    if selected_count > 0:
        st.sidebar.success(f"✅ Selected: {selected_count} for processing")
        
        # Show selected participants in sidebar
        with st.sidebar.expander("View Selected", expanded=False):
            for participant in st.session_state.selected_participants:
                st.write(f"• {participant}")
    else:
        st.sidebar.error("❌ No participants selected")
    
    # Pipeline status
    if hasattr(st.session_state, 'pipeline_config'):
        st.sidebar.success("✅ Pipeline configured")
    else:
        st.sidebar.error("❌ Pipeline not configured")
    
    # Quick stats
    if st.session_state.participants_df is not None:
        st.sidebar.subheader("📈 Quick Stats")
        df = st.session_state.participants_df
        
        # Age statistics
        if 'age' in df.columns and df['age'].dtype in ['int64', 'float64']:
            age_mean = df['age'].mean()
            age_std = df['age'].std()
            st.sidebar.metric("Avg Age", f"{age_mean:.1f} ± {age_std:.1f}")
        
        # Sex distribution
        if 'sex' in df.columns:
            sex_counts = df['sex'].value_counts()
            for sex, count in sex_counts.items():
                st.sidebar.metric(f"Sex: {sex}", count)
    
    # Help section
    st.sidebar.subheader("❓ Need Help?")
    with st.sidebar.expander("Usage Guide", expanded=False):
        st.write("""
        **Step-by-Step Guide:**
        
        1. **Data Setup**: Upload your CSV and scan for NII files
        2. **Match Participants**: Connect participant IDs with NII files
        3. **Select & Configure**: Choose participants and set pipeline options
        4. **Run Pipeline**: Execute the customized pipeline
        5. **View Results**: Analyze the output and download results
        
        **CSV Format**: Should contain participant ID, age, and sex columns
        **NII Files**: Should be in .nii or .nii.gz format
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    🧠 Fully Customizable MRI Processing Pipeline v2.0<br>
    Select • Match • Process • Analyze<br>
    Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()