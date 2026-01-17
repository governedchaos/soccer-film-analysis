#!/bin/bash
# ============================================================
# Soccer Film Analysis - Run Application
# ============================================================

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate 2>/dev/null || {
        echo "Error: Virtual environment not found."
        echo "Run ./deploy.sh first to set up the project."
        exit 1
    }
fi

# Check what to run
case "${1:-gui}" in
    gui|--gui|-g)
        echo "Starting GUI application..."
        python -m src.gui.main_window
        ;;
    cli|--cli|-c)
        shift
        if [ -z "$1" ]; then
            echo "Usage: ./run.sh cli VIDEO_PATH [--depth quick|standard|deep]"
            exit 1
        fi
        echo "Running CLI analysis..."
        python scripts/run_analysis.py "$@"
        ;;
    test|--test|-t)
        echo "Running tests..."
        python -c "
from config.settings import settings
print(f'Database: {settings.db_name}')
print(f'Device: {settings.get_device()}')
print('Imports OK')
"
        ;;
    db|--db|-d)
        echo "Setting up database..."
        python scripts/setup_database.py
        ;;
    *)
        echo "Soccer Film Analysis - Run Script"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  gui      Launch GUI application (default)"
        echo "  cli      Run CLI analysis: ./run.sh cli VIDEO_PATH"
        echo "  test     Run quick test"
        echo "  db       Setup/reset database"
        echo ""
        ;;
esac
