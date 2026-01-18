"""
Soccer Film Analysis - PDF Report Generation
Generate comprehensive PDF reports with charts, heatmaps, and statistics
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether, Flowable
    )
    from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF generation will be unavailable.")

    # Stub classes for when ReportLab is not available
    class Flowable:
        """Stub Flowable class when ReportLab not installed."""
        def __init__(self, *args, **kwargs):
            pass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class ReportConfig:
    """Configuration for PDF report"""
    title: str = "Match Analysis Report"
    home_team: str = "Home"
    away_team: str = "Away"
    competition: str = ""
    venue: str = ""
    date: str = ""
    author: str = "Soccer Film Analysis"

    # Content options
    include_summary: bool = True
    include_possession: bool = True
    include_shots: bool = True
    include_passes: bool = True
    include_heatmaps: bool = True
    include_formations: bool = True
    include_events: bool = True
    include_player_stats: bool = True

    # Styling
    primary_color: Tuple[int, int, int] = (33, 150, 243)  # Blue
    home_color: Tuple[int, int, int] = (255, 193, 7)  # Yellow
    away_color: Tuple[int, int, int] = (244, 67, 54)  # Red
    page_size: str = "letter"  # letter or A4


class PitchDrawing(Flowable):
    """Custom flowable for drawing a soccer pitch"""

    def __init__(self, width=400, height=260, positions=None, title=""):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.positions = positions or []  # List of (x, y, team_id, label)
        self.title = title

    def draw(self):
        canvas = self.canv

        # Green background
        canvas.setFillColor(colors.Color(0.2, 0.5, 0.2))
        canvas.rect(0, 0, self.width, self.height, fill=1)

        # White lines
        canvas.setStrokeColor(colors.white)
        canvas.setLineWidth(1)

        # Outer boundary
        margin = 10
        canvas.rect(margin, margin, self.width - 2*margin, self.height - 2*margin)

        # Center line
        mid_x = self.width / 2
        canvas.line(mid_x, margin, mid_x, self.height - margin)

        # Center circle
        canvas.circle(mid_x, self.height / 2, 30)

        # Penalty areas
        pa_width = 60
        pa_height = 100
        # Left
        canvas.rect(margin, (self.height - pa_height) / 2, pa_width, pa_height)
        # Right
        canvas.rect(self.width - margin - pa_width, (self.height - pa_height) / 2, pa_width, pa_height)

        # Goal areas
        ga_width = 20
        ga_height = 50
        canvas.rect(margin, (self.height - ga_height) / 2, ga_width, ga_height)
        canvas.rect(self.width - margin - ga_width, (self.height - ga_height) / 2, ga_width, ga_height)

        # Draw positions
        for x, y, team_id, label in self.positions:
            # Scale positions to pitch
            px = margin + x * (self.width - 2*margin)
            py = margin + y * (self.height - 2*margin)

            # Team colors
            if team_id == 0:
                canvas.setFillColor(colors.Color(1, 0.8, 0))  # Yellow
            else:
                canvas.setFillColor(colors.Color(0.9, 0.3, 0.2))  # Red

            canvas.circle(px, py, 8, fill=1)

            if label:
                canvas.setFillColor(colors.white)
                canvas.setFont("Helvetica-Bold", 6)
                canvas.drawCentredString(px, py - 2, str(label))

        # Title
        if self.title:
            canvas.setFillColor(colors.white)
            canvas.setFont("Helvetica-Bold", 10)
            canvas.drawCentredString(self.width / 2, self.height - 5, self.title)


class PDFReportGenerator:
    """
    Generates comprehensive PDF reports for match analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")

        from config import settings
        self.output_dir = output_dir or settings.get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.Color(0.13, 0.59, 0.95)
        ))
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=5
        ))
        self.styles.add(ParagraphStyle(
            name='StatValue',
            parent=self.styles['Normal'],
            fontSize=24,
            alignment=TA_CENTER,
            textColor=colors.Color(0.13, 0.59, 0.95)
        ))

    def generate_report(
        self,
        config: ReportConfig,
        game_stats: Dict,
        events: List[Dict],
        player_stats: Optional[List[Dict]] = None,
        heatmap_images: Optional[Dict[str, Path]] = None,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate a comprehensive PDF report.

        Args:
            config: Report configuration
            game_stats: Dict with game-level statistics
            events: List of events from the match
            player_stats: Optional list of player statistics
            heatmap_images: Optional dict mapping team/player to heatmap image paths
            output_filename: Optional custom filename

        Returns:
            Path to generated PDF
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_filename or f"match_report_{timestamp}.pdf"
        output_path = self.output_dir / filename

        page_size = letter if config.page_size == "letter" else A4
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=page_size,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        story = []

        # Title Page
        story.extend(self._create_title_page(config, game_stats))

        # Match Summary
        if config.include_summary:
            story.extend(self._create_summary_section(config, game_stats))

        # Possession Analysis
        if config.include_possession:
            story.extend(self._create_possession_section(game_stats))

        # Shots Analysis
        if config.include_shots:
            story.extend(self._create_shots_section(game_stats))

        # Passing Analysis
        if config.include_passes:
            story.extend(self._create_passing_section(game_stats))

        # Heatmaps
        if config.include_heatmaps and heatmap_images:
            story.extend(self._create_heatmaps_section(heatmap_images))

        # Formations
        if config.include_formations and 'formations' in game_stats:
            story.extend(self._create_formations_section(config, game_stats))

        # Events Timeline
        if config.include_events and events:
            story.extend(self._create_events_section(events))

        # Player Statistics
        if config.include_player_stats and player_stats:
            story.extend(self._create_player_stats_section(player_stats))

        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
        return output_path

    def _create_title_page(self, config: ReportConfig, game_stats: Dict) -> List:
        """Create title page content"""
        elements = []

        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph(config.title, self.styles['ReportTitle']))
        elements.append(Spacer(1, 0.5*inch))

        # Match info
        score_text = f"{config.home_team}  {game_stats.get('home_score', 0)} - {game_stats.get('away_score', 0)}  {config.away_team}"
        elements.append(Paragraph(score_text, ParagraphStyle(
            'ScoreLine', parent=self.styles['Normal'],
            fontSize=20, alignment=TA_CENTER, spaceAfter=20
        )))

        if config.competition:
            elements.append(Paragraph(config.competition, ParagraphStyle(
                'CompLine', parent=self.styles['Normal'],
                fontSize=14, alignment=TA_CENTER
            )))

        if config.venue:
            elements.append(Paragraph(config.venue, ParagraphStyle(
                'VenueLine', parent=self.styles['Normal'],
                fontSize=12, alignment=TA_CENTER, textColor=colors.gray
            )))

        if config.date:
            elements.append(Paragraph(config.date, ParagraphStyle(
                'DateLine', parent=self.styles['Normal'],
                fontSize=12, alignment=TA_CENTER, textColor=colors.gray
            )))

        elements.append(Spacer(1, 1*inch))
        elements.append(PageBreak())

        return elements

    def _create_summary_section(self, config: ReportConfig, game_stats: Dict) -> List:
        """Create match summary section"""
        elements = []
        elements.append(Paragraph("Match Summary", self.styles['SectionTitle']))

        # Key stats table
        data = [
            ['Statistic', config.home_team, config.away_team],
            ['Possession', f"{game_stats.get('home_possession', 50):.0f}%",
             f"{game_stats.get('away_possession', 50):.0f}%"],
            ['Shots', str(game_stats.get('home_shots', 0)),
             str(game_stats.get('away_shots', 0))],
            ['Shots on Target', str(game_stats.get('home_shots_on_target', 0)),
             str(game_stats.get('away_shots_on_target', 0))],
            ['xG', f"{game_stats.get('home_xg', 0):.2f}",
             f"{game_stats.get('away_xg', 0):.2f}"],
            ['Passes', str(game_stats.get('home_passes', 0)),
             str(game_stats.get('away_passes', 0))],
            ['Pass Accuracy', f"{game_stats.get('home_pass_accuracy', 0):.0f}%",
             f"{game_stats.get('away_pass_accuracy', 0):.0f}%"],
            ['Corners', str(game_stats.get('home_corners', 0)),
             str(game_stats.get('away_corners', 0))],
            ['Fouls', str(game_stats.get('home_fouls', 0)),
             str(game_stats.get('away_fouls', 0))],
        ]

        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.13, 0.59, 0.95)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_possession_section(self, game_stats: Dict) -> List:
        """Create possession analysis section"""
        elements = []
        elements.append(Paragraph("Possession Analysis", self.styles['SectionTitle']))

        home_poss = game_stats.get('home_possession', 50)
        away_poss = game_stats.get('away_possession', 50)

        # Create bar chart
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh([''], [home_poss], color='#FFC107', label='Home')
            ax.barh([''], [away_poss], left=[home_poss], color='#F44336', label='Away')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Possession %')
            ax.legend(loc='upper right')
            ax.text(home_poss/2, 0, f"{home_poss:.0f}%", ha='center', va='center', color='black', fontweight='bold')
            ax.text(home_poss + away_poss/2, 0, f"{away_poss:.0f}%", ha='center', va='center', color='white', fontweight='bold')

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)

            elements.append(Image(img_buffer, width=5*inch, height=1.5*inch))

        elements.append(Spacer(1, 0.2*inch))
        return elements

    def _create_shots_section(self, game_stats: Dict) -> List:
        """Create shots analysis section"""
        elements = []
        elements.append(Paragraph("Shots Analysis", self.styles['SectionTitle']))

        # Shot locations visualization
        shot_positions = game_stats.get('shot_positions', [])
        pitch = PitchDrawing(
            width=400, height=260,
            positions=[(s['x'], s['y'], s['team_id'], 'G' if s.get('goal') else '') for s in shot_positions],
            title="Shot Locations"
        )
        elements.append(pitch)

        # xG summary
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Expected Goals (xG)", self.styles['SubSection']))

        xg_data = [
            ['Team', 'Goals', 'xG', 'Difference'],
            ['Home', str(game_stats.get('home_score', 0)),
             f"{game_stats.get('home_xg', 0):.2f}",
             f"{game_stats.get('home_score', 0) - game_stats.get('home_xg', 0):+.2f}"],
            ['Away', str(game_stats.get('away_score', 0)),
             f"{game_stats.get('away_xg', 0):.2f}",
             f"{game_stats.get('away_score', 0) - game_stats.get('away_xg', 0):+.2f}"],
        ]

        xg_table = Table(xg_data, colWidths=[1.5*inch]*4)
        xg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.3, 0.3, 0.3)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.gray),
        ]))
        elements.append(xg_table)

        elements.append(Spacer(1, 0.3*inch))
        return elements

    def _create_passing_section(self, game_stats: Dict) -> List:
        """Create passing analysis section"""
        elements = []
        elements.append(Paragraph("Passing Analysis", self.styles['SectionTitle']))

        if MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

            # Pass counts
            teams = ['Home', 'Away']
            passes = [game_stats.get('home_passes', 0), game_stats.get('away_passes', 0)]
            ax1.bar(teams, passes, color=['#FFC107', '#F44336'])
            ax1.set_ylabel('Total Passes')
            ax1.set_title('Pass Volume')

            # Pass accuracy
            accuracy = [game_stats.get('home_pass_accuracy', 0), game_stats.get('away_pass_accuracy', 0)]
            ax2.bar(teams, accuracy, color=['#FFC107', '#F44336'])
            ax2.set_ylabel('Accuracy %')
            ax2.set_title('Pass Accuracy')
            ax2.set_ylim(0, 100)

            plt.tight_layout()

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)

            elements.append(Image(img_buffer, width=6*inch, height=2.5*inch))

        elements.append(Spacer(1, 0.3*inch))
        return elements

    def _create_heatmaps_section(self, heatmap_images: Dict[str, Path]) -> List:
        """Create heatmaps section"""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("Position Heatmaps", self.styles['SectionTitle']))

        for label, img_path in heatmap_images.items():
            if Path(img_path).exists():
                elements.append(Paragraph(label, self.styles['SubSection']))
                elements.append(Image(str(img_path), width=5*inch, height=3.5*inch))
                elements.append(Spacer(1, 0.2*inch))

        return elements

    def _create_formations_section(self, config: ReportConfig, game_stats: Dict) -> List:
        """Create formations section"""
        elements = []
        elements.append(Paragraph("Formations", self.styles['SectionTitle']))

        formations = game_stats.get('formations', {})

        for team_id, formation_data in formations.items():
            team_name = config.home_team if team_id == 0 else config.away_team
            formation = formation_data.get('formation', 'Unknown')
            positions = formation_data.get('positions', [])

            elements.append(Paragraph(f"{team_name}: {formation}", self.styles['SubSection']))

            pitch_positions = [
                (p['x'], p['y'], team_id, p.get('jersey', ''))
                for p in positions
            ]
            pitch = PitchDrawing(width=350, height=230, positions=pitch_positions)
            elements.append(pitch)
            elements.append(Spacer(1, 0.2*inch))

        return elements

    def _create_events_section(self, events: List[Dict]) -> List:
        """Create events timeline section"""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("Match Events", self.styles['SectionTitle']))

        # Filter to key events
        key_events = [e for e in events if e.get('event_type') in
                      ['goal', 'shot', 'save', 'corner', 'foul', 'kickoff',
                       'halftime_start', 'second_half', 'game_end']]

        if not key_events:
            key_events = events[:30]  # Show first 30 if no key events

        data = [['Time', 'Event', 'Team', 'Details']]
        for event in key_events[:40]:  # Limit to 40 events
            data.append([
                event.get('timestamp', event.get('time', '')),
                event.get('event_type', '').replace('_', ' ').title(),
                'Home' if event.get('team_id') == 0 else 'Away' if event.get('team_id') == 1 else '-',
                event.get('description', '')[:40]
            ])

        table = Table(data, colWidths=[1*inch, 1.5*inch, 1*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))

        elements.append(table)
        return elements

    def _create_player_stats_section(self, player_stats: List[Dict]) -> List:
        """Create player statistics section"""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("Player Statistics", self.styles['SectionTitle']))

        # Sort by team and then by some metric
        home_players = [p for p in player_stats if p.get('team_id') == 0]
        away_players = [p for p in player_stats if p.get('team_id') == 1]

        for team_label, players in [('Home Team', home_players), ('Away Team', away_players)]:
            if not players:
                continue

            elements.append(Paragraph(team_label, self.styles['SubSection']))

            data = [['#', 'Name', 'Min', 'Goals', 'Assists', 'Passes', 'Distance']]
            for p in sorted(players, key=lambda x: x.get('jersey_number', 99)):
                data.append([
                    str(p.get('jersey_number', '')),
                    p.get('name', 'Unknown')[:20],
                    str(p.get('minutes_played', 0)),
                    str(p.get('goals', 0)),
                    str(p.get('assists', 0)),
                    str(p.get('passes', 0)),
                    f"{p.get('distance_km', 0):.1f}km"
                ])

            table = Table(data, colWidths=[0.4*inch, 1.8*inch, 0.5*inch, 0.6*inch, 0.6*inch, 0.7*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.3, 0.3, 0.3)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (1, 1), (1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))

        return elements


def generate_quick_report(
    game_stats: Dict,
    events: List[Dict],
    home_team: str = "Home",
    away_team: str = "Away",
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Quick function to generate a basic PDF report.

    Args:
        game_stats: Dictionary with game statistics
        events: List of match events
        home_team: Home team name
        away_team: Away team name
        output_path: Optional output path

    Returns:
        Path to generated PDF or None if failed
    """
    if not REPORTLAB_AVAILABLE:
        logger.error("ReportLab not available for PDF generation")
        return None

    try:
        generator = PDFReportGenerator()
        config = ReportConfig(
            home_team=home_team,
            away_team=away_team,
            date=datetime.now().strftime("%Y-%m-%d")
        )
        return generator.generate_report(
            config=config,
            game_stats=game_stats,
            events=events,
            output_filename=str(output_path.name) if output_path else None
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return None
