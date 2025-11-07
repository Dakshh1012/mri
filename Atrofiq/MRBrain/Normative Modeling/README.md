
## Project Structure
```
Normative Modeling/  
        ├── Input Data/                 # Raw Excel files (volume, age) by gender and anatomy  
        ├── Percentiles/                # Generated percentile curves (output)  
        ├── output/                     # Statistical results from distribution fitting  
        ├── dashboard.py                # Streamlit visualization dashboard  
        ├── API.py                      # CLI tool to analyze a specific participant  
        ├── API_plot.py                 # CLI tool to plot normative curves from results  
        ├── Gamlss_BestFit_Finder.R     # R script to find optimal distributions  
        ├── Gamlss_Percentile_Curves.R  # R script to generate percentile curves  
        └── README.txt  
```
---

## System Requirements

- **Python Requirements**  
  ```bash
  pip install streamlit pandas numpy plotly openpyxl
   ```

* **R Requirements**

  ```r
  install.packages(c("gamlss", "gamlss.dist", "readxl", "writexl"))
  ```

---

## Workflow Instructions

### 1. Data Preparation

* Split your dataset by gender (male/female).
* For each gender and brain anatomy, create Excel files with two columns: **Volume** and **Age**.
* Name files following the pattern: `{gender}_{anatomy}.xlsx` (e.g., `female_left_hippocampus.xlsx`).
* Place all Excel files in the **Input Data** folder.

---

### 2. Find Optimal Distributions

* Open `Gamlss_BestFit_Finder.R`.
* Update the paths:

  ```r
  data_folder <- "C:/Your/Path/To/Input Data"
  output_directory <- "C:/Your/Path/To/output"
  ```
* Run the script to test candidate distributions (`NO`, `GA`, `BCCG`, `BCPE`).
* Check `output/statistical_results.txt` for AIC/BIC values.
* Select the best distribution for each anatomy-gender combination.

---

### 3. Generate Percentile Curves

* Open `Gamlss_Percentile_Curves.R`.
* Update the feature list with your optimal distributions:

  ```r
  c('female_left_hippocampus', 'BCPE'),
  c('male_left_hippocampus', 'BCCG'),
  ```
* Update paths:

  ```r
  input_folder <- "C:/Your/Path/To/Input Data"
  output_folder <- "C:/Your/Path/To/Percentiles"
  ```
* Run the script to generate percentile curves (1st–99th) for ages 1–100.

---

### 4. Visualize Results with Dashboard

* Open `dashboard.py`.
* Update the default path if needed:

  ```python
  DEFAULT_BASE = r"C:/Your/Path/To/Percentiles"
  ```
* Run:

  ```bash
  streamlit run dashboard.py
  ```

---

### 5. Analyze a Specific Participant (API.py)

Run the CLI tool to analyze a participant against normative curves:

```bash
python API.py --participant-id P001 \
              --metadata metadata.json \
              --importance-folder importance/ \
              --percentiles-folder Percentiles/ \
              --top-regions 10 \
              --percentiles 1 5 10 25 50 75 90 95 99 \
              --smooth --smooth-window 5 \
              --output results.json --pretty
```

**Arguments:**

* `--participant-id, -pid` : Participant ID to analyze (**required**)
* `--metadata, -m` : Path to `metadata.json` file (**required**)
* `--importance-folder, -if` : Folder containing `before_40.json` and `after_40.json` (**required**)
* `--percentiles-folder, -pf` : Base folder containing percentile Excel files (**required**)
* `--top-regions, -tr` : Number of top regions to analyze (default: 10)
* `--percentiles, -p` : Percentiles to include (default: 1 5 10 25 50 75 90 95 99)
* `--smooth` : Apply smoothing to curves
* `--smooth-window` : Smoothing window size (default: 5)
* `--output, -o` : Output JSON file path (default: stdout)
* `--pretty` : Pretty-print JSON output

---

### 6. Plot Normative Curves (API\_plot.py)

Use the results from `API.py` to generate plots:

```bash
python API_plot.py --results results.json --output-dir plots/ --summary-only
```

**Arguments:**

* `--results, -r` : Path to `results.json` (**required**)
* `--output-dir, -o` : Directory to save plots (default: current folder)
* `--summary-only, -s` : Only create summary plot (skip individual plots)

---

