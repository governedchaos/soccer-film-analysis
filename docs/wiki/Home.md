# Soccer Film Analysis Wiki

Welcome to the Soccer Film Analysis wiki! This application provides comprehensive analysis of soccer game footage using computer vision and machine learning.

## Quick Links

- [Getting Started](Getting-Started)
- [Features Overview](Features)
- [Detection System](Detection-System)
- [Analytics Modules](Analytics-Modules)
- [GUI Guide](GUI-Guide)
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)

## About This Project

Soccer Film Analysis is a desktop application designed for high school soccer coaches and analysts. It processes game video to automatically:

- **Detect players** and track them throughout the match
- **Classify teams** by jersey color
- **Track the ball** and determine possession
- **Identify referees** separately from players
- **Analyze formations** and tactical patterns
- **Calculate metrics** like xG, pressing intensity, and more

### Key Features

- **Runs Offline**: Uses local YOLOv8 models - no API keys or internet required
- **Modern GUI**: PyQt6 interface with real-time visualization
- **Comprehensive Analytics**: Formation detection, xG model, space analysis
- **Database Storage**: PostgreSQL for persistent game/player data
- **Export Options**: PDF reports, data export

## Technology

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 + ByteTrack |
| GUI | PyQt6 |
| Database | PostgreSQL |
| Analytics | NumPy, SciPy, scikit-learn |

## Getting Help

- Check the [Troubleshooting](Troubleshooting) page for common issues
- Review the [FAQ](FAQ) for frequently asked questions
- Open an issue on GitHub for bugs or feature requests

## Contributing

We welcome contributions! See the main README for contribution guidelines.
