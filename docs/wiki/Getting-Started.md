# Getting Started

This guide will help you get Soccer Film Analysis up and running quickly.

## Prerequisites

- **Python 3.10+**
- **Git**
- **PostgreSQL 14+** (optional, for database features)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/governedchaos/soccer-film-analysis.git
cd soccer-film-analysis
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m src.gui.main_window
```

The first time you run the app, YOLOv8 models (~25MB) will be downloaded automatically.

## First Analysis

1. **Load a Video**: Click "Open Video" and select a game recording
2. **Set Team Colors** (optional): Use the color picker to set jersey colors
3. **Start Analysis**: Click "Start Analysis" button
4. **View Results**: Watch real-time detection and check the stats panel

## Database Setup (Optional)

To save analysis across sessions:

1. Install PostgreSQL
2. Create database: `CREATE DATABASE soccer_analysis;`
3. Copy `.env.example` to `.env` and configure database credentials
4. Run: `python scripts/setup_database.py`

## Next Steps

- Read about [Features](Features) to understand what the app can do
- Learn about the [Detection System](Detection-System) for accuracy tips
- Check [Analytics Modules](Analytics-Modules) for advanced analysis
