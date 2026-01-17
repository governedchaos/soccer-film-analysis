#!/usr/bin/env python3
"""
Soccer Film Analysis - Command Line Analysis Tool
Run video analysis from the command line
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from config import settings, AnalysisDepth
from src.core.video_processor import VideoProcessor, AnalysisProgress


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze soccer game video from command line"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--depth",
        type=str,
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Analysis depth level (default: standard)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame number"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End frame number (default: end of video)"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't save results to database"
    )
    parser.add_argument(
        "--home-color",
        type=str,
        help="Home team jersey color as R,G,B (e.g., '255,255,0' for yellow)"
    )
    parser.add_argument(
        "--away-color",
        type=str,
        help="Away team jersey color as R,G,B (e.g., '255,0,0' for red)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for reports"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    # Verify video exists
    video_path = Path(args.video)
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        sys.exit(1)
    
    # Parse team colors if provided
    home_color = None
    away_color = None
    if args.home_color:
        try:
            home_color = tuple(map(int, args.home_color.split(',')))
        except ValueError:
            console.print("[red]Error: Invalid home-color format. Use R,G,B (e.g., '255,255,0')[/red]")
            sys.exit(1)
    
    if args.away_color:
        try:
            away_color = tuple(map(int, args.away_color.split(',')))
        except ValueError:
            console.print("[red]Error: Invalid away-color format. Use R,G,B (e.g., '255,0,0')[/red]")
            sys.exit(1)
    
    console.print("\n[bold blue]Soccer Film Analysis[/bold blue]")
    console.print("=" * 50)
    console.print(f"Video: [green]{video_path.name}[/green]")
    console.print(f"Depth: [yellow]{args.depth}[/yellow]")
    console.print("=" * 50 + "\n")
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Load video
    console.print("[bold]Loading video...[/bold]")
    video_info = processor.load_video(video_path)
    console.print(f"  Resolution: {video_info.width}x{video_info.height}")
    console.print(f"  FPS: {video_info.fps:.1f}")
    console.print(f"  Duration: {video_info.duration_seconds/60:.1f} minutes")
    console.print(f"  Frames: {video_info.total_frames:,}")
    
    # Calibrate team colors
    console.print("\n[bold]Calibrating team colors...[/bold]")
    if home_color and away_color:
        processor.team_classifier.set_team_colors(home_color, away_color)
        console.print(f"  Using provided colors: Home={home_color}, Away={away_color}")
    else:
        processor.calibrate_teams()
        if processor.team_classifier.team_colors is not None:
            colors = processor.team_classifier.team_colors
            console.print(f"  Detected colors: Home={tuple(colors[0])}, Away={tuple(colors[1])}")
    
    # Run analysis with progress bar
    console.print(f"\n[bold]Starting {args.depth} analysis...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing...", total=100)
        
        def on_progress(p: AnalysisProgress):
            progress.update(task, completed=p.percentage, description=p.current_phase)
        
        result = processor.process_video(
            analysis_depth=AnalysisDepth(args.depth),
            start_frame=args.start,
            end_frame=args.end,
            progress_callback=on_progress,
            save_to_db=not args.no_db
        )
    
    # Print results
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print("=" * 50)
    console.print(f"Processed frames: {result.get('processed_frames', 0):,}")
    console.print(f"Processing time: {result.get('elapsed_seconds', 0)/60:.1f} minutes")
    
    if result.get('session_id'):
        console.print(f"Session ID: {result['session_id']}")
    
    console.print("\n[dim]Run with --output to generate reports[/dim]")


if __name__ == "__main__":
    main()
