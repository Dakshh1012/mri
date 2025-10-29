# MRI Brain Analysis Frontend

A React-based frontend application for MRI brain analysis with two main features:
1. **Brain Age Prediction** - Upload MRI scans to predict brain age
2. **Normative Modeling** - Compare brain volumes against population norms with interactive charts

## Features

### Brain Age Prediction Page
- Upload NIfTI files (.nii or .nii.gz)
- Input age and gender
- Display predicted brain age and brain age gap
- Show volumetric analysis results
- Real-time processing status

### Normative Modeling Page
- Upload NIfTI files (.nii or .nii.gz)
- Input age and gender
- Interactive charts showing:
  - Percentile scores by brain region
  - Z-scores by brain region
  - Outlier detection and highlighting
- Summary statistics and outlier regions

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- MRI Backend API running on `http://localhost:8000`

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

1. Make sure your MRI Backend API is running on port 8000
2. Start the React development server:
```bash
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   └── Navigation.js       # Navigation bar component
│   ├── pages/
│   │   ├── BrainAge.js        # Brain age prediction page
│   │   └── NormativeModeling.js # Normative modeling with charts
│   ├── App.js                 # Main app component with routing
│   ├── index.js              # React entry point
│   └── index.css             # Global styles
└── package.json              # Dependencies and scripts
```

## Dependencies

- **React** - Frontend framework
- **React Router** - Navigation between pages
- **Axios** - HTTP requests to backend API
- **Recharts** - Interactive charts for normative modeling
- **Material-UI** - UI components and icons

## API Integration

The frontend communicates with the backend API endpoints:
- `POST /brain-age` - Brain age prediction
- `POST /normative` - Normative modeling analysis

Both endpoints expect:
- `nifti_file`: NIfTI file upload
- `age`: Age in years
- `gender`: M or F

## Chart Features

### Percentile Chart
- Shows brain region volumes as percentiles (0-100%)
- Red bars indicate outlier regions
- Reference lines at 5th, 50th, and 95th percentiles

### Z-Score Chart
- Shows standard deviations from population mean
- Red bars for outliers (beyond ±2 standard deviations)
- Reference lines at mean and ±2 SD

## File Upload Support

- Accepts `.nii` and `.nii.gz` files
- File size validation and display
- Real-time file validation

## Error Handling

- Network error handling
- File type validation
- API error message display
- Loading states with spinners

## Responsive Design

- Mobile-friendly responsive layout
- Grid-based result displays
- Collapsible detailed data sections

## Build for Production

```bash
npm run build
```

This creates a `build` folder with production-ready files.