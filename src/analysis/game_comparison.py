"""
Soccer Film Analysis - Multi-Game Comparison
Compare statistics across multiple games
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class GameStats:
    """Statistics for a single game"""
    game_id: str
    date: Optional[datetime] = None
    home_team: str = ""
    away_team: str = ""
    home_score: int = 0
    away_score: int = 0

    # Possession
    home_possession: float = 50.0
    away_possession: float = 50.0

    # Shots
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0

    # xG
    home_xg: float = 0.0
    away_xg: float = 0.0

    # Passes
    home_passes: int = 0
    away_passes: int = 0
    home_pass_accuracy: float = 0.0
    away_pass_accuracy: float = 0.0

    # Other stats
    home_corners: int = 0
    away_corners: int = 0
    home_fouls: int = 0
    away_fouls: int = 0

    # Player-level stats
    player_stats: Dict = field(default_factory=dict)


class GameComparisonAnalyzer:
    """
    Analyzes and compares statistics across multiple games.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        from config import settings
        self.data_dir = data_dir or settings.get_output_dir()
        self.games: List[GameStats] = []

    def add_game(self, stats: GameStats):
        """Add a game's statistics"""
        self.games.append(stats)
        logger.info(f"Added game: {stats.game_id}")

    def load_game_from_file(self, filepath: Path) -> Optional[GameStats]:
        """Load game statistics from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stats = GameStats(
                game_id=data.get('game_id', filepath.stem),
                date=datetime.fromisoformat(data['date']) if 'date' in data else None,
                home_team=data.get('home_team', ''),
                away_team=data.get('away_team', ''),
                home_score=data.get('home_score', 0),
                away_score=data.get('away_score', 0),
                home_possession=data.get('possession', {}).get('home', 50),
                away_possession=data.get('possession', {}).get('away', 50),
                home_shots=data.get('shots', {}).get('home', 0),
                away_shots=data.get('shots', {}).get('away', 0),
                home_xg=data.get('xg', {}).get('home', 0),
                away_xg=data.get('xg', {}).get('away', 0),
                player_stats=data.get('player_stats', {})
            )

            self.add_game(stats)
            return stats

        except Exception as e:
            logger.error(f"Failed to load game from {filepath}: {e}")
            return None

    def load_all_games(self, pattern: str = "analysis_*.json"):
        """Load all game files matching pattern"""
        for filepath in self.data_dir.glob(pattern):
            self.load_game_from_file(filepath)

        logger.info(f"Loaded {len(self.games)} games")

    def get_team_trends(self, team_name: str) -> Dict:
        """
        Get trends for a specific team across all games.
        """
        team_games = []

        for game in self.games:
            if game.home_team == team_name:
                team_games.append({
                    'game_id': game.game_id,
                    'date': game.date,
                    'opponent': game.away_team,
                    'home_away': 'home',
                    'goals_for': game.home_score,
                    'goals_against': game.away_score,
                    'possession': game.home_possession,
                    'shots': game.home_shots,
                    'xg': game.home_xg,
                    'passes': game.home_passes
                })
            elif game.away_team == team_name:
                team_games.append({
                    'game_id': game.game_id,
                    'date': game.date,
                    'opponent': game.home_team,
                    'home_away': 'away',
                    'goals_for': game.away_score,
                    'goals_against': game.home_score,
                    'possession': game.away_possession,
                    'shots': game.away_shots,
                    'xg': game.away_xg,
                    'passes': game.away_passes
                })

        if not team_games:
            return {'error': f'No games found for team: {team_name}'}

        # Calculate averages
        avg_possession = sum(g['possession'] for g in team_games) / len(team_games)
        avg_shots = sum(g['shots'] for g in team_games) / len(team_games)
        avg_xg = sum(g['xg'] for g in team_games) / len(team_games)
        avg_goals_for = sum(g['goals_for'] for g in team_games) / len(team_games)
        avg_goals_against = sum(g['goals_against'] for g in team_games) / len(team_games)

        wins = sum(1 for g in team_games if g['goals_for'] > g['goals_against'])
        draws = sum(1 for g in team_games if g['goals_for'] == g['goals_against'])
        losses = sum(1 for g in team_games if g['goals_for'] < g['goals_against'])

        return {
            'team': team_name,
            'games_played': len(team_games),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'avg_possession': round(avg_possession, 1),
            'avg_shots': round(avg_shots, 1),
            'avg_xg': round(avg_xg, 2),
            'avg_goals_for': round(avg_goals_for, 2),
            'avg_goals_against': round(avg_goals_against, 2),
            'games': team_games
        }

    def get_player_trends(self, player_id: int, team_name: str) -> Dict:
        """
        Get trends for a specific player across all games.
        """
        player_data = []

        for game in self.games:
            if game.home_team == team_name or game.away_team == team_name:
                player_stats = game.player_stats.get(str(player_id), {})
                if player_stats:
                    player_data.append({
                        'game_id': game.game_id,
                        'date': game.date,
                        **player_stats
                    })

        if not player_data:
            return {'error': f'No data found for player {player_id}'}

        # Calculate averages for common stats
        stat_keys = set()
        for pd in player_data:
            stat_keys.update(pd.keys())
        stat_keys -= {'game_id', 'date'}

        averages = {}
        for key in stat_keys:
            values = [pd.get(key, 0) for pd in player_data if isinstance(pd.get(key), (int, float))]
            if values:
                averages[f'avg_{key}'] = round(sum(values) / len(values), 2)

        return {
            'player_id': player_id,
            'games_played': len(player_data),
            'averages': averages,
            'games': player_data
        }

    def compare_games(
        self,
        game_ids: List[str]
    ) -> Dict:
        """
        Compare specific games side by side.
        """
        comparison = {
            'games': [],
            'metrics': {}
        }

        selected_games = [g for g in self.games if g.game_id in game_ids]

        for game in selected_games:
            comparison['games'].append({
                'id': game.game_id,
                'matchup': f"{game.home_team} vs {game.away_team}",
                'score': f"{game.home_score}-{game.away_score}",
                'possession': f"{game.home_possession:.0f}%-{game.away_possession:.0f}%",
                'shots': f"{game.home_shots}-{game.away_shots}",
                'xg': f"{game.home_xg:.2f}-{game.away_xg:.2f}"
            })

        # Calculate metric comparisons
        metrics = ['home_possession', 'home_shots', 'home_xg', 'home_passes']
        for metric in metrics:
            values = [getattr(g, metric, 0) for g in selected_games]
            comparison['metrics'][metric] = {
                'values': values,
                'avg': sum(values) / len(values) if values else 0,
                'max': max(values) if values else 0,
                'min': min(values) if values else 0
            }

        return comparison

    def plot_trends(
        self,
        team_name: str,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot trends for a team over time.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for plotting")
            return None

        trends = self.get_team_trends(team_name)
        if 'error' in trends:
            logger.error(trends['error'])
            return None

        games = trends['games']
        if not games:
            return None

        metrics = metrics or ['possession', 'shots', 'xg']

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        x = range(len(games))
        labels = [g.get('opponent', f"Game {i+1}") for i, g in enumerate(games)]

        for ax, metric in zip(axes, metrics):
            values = [g.get(metric, 0) for g in games]
            ax.bar(x, values, color='steelblue', alpha=0.7)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.axhline(y=sum(values)/len(values), color='red',
                      linestyle='--', label='Average')
            ax.legend()

        fig.suptitle(f'{team_name} - Performance Trends')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Trend chart saved to: {save_path}")

        return fig

    def generate_comparison_report(self) -> str:
        """
        Generate a text report comparing all loaded games.
        """
        if not self.games:
            return "No games loaded for comparison."

        lines = [
            "=" * 60,
            "MULTI-GAME COMPARISON REPORT",
            "=" * 60,
            f"Total games analyzed: {len(self.games)}",
            ""
        ]

        # List all games
        lines.append("-" * 60)
        lines.append("GAMES")
        lines.append("-" * 60)

        for game in self.games:
            date_str = game.date.strftime("%Y-%m-%d") if game.date else "Unknown"
            lines.append(
                f"{date_str}: {game.home_team} {game.home_score}-{game.away_score} {game.away_team}"
            )

        # Aggregate stats
        lines.extend(["", "-" * 60, "AGGREGATE STATISTICS", "-" * 60])

        total_goals = sum(g.home_score + g.away_score for g in self.games)
        avg_possession_home = sum(g.home_possession for g in self.games) / len(self.games)
        total_shots = sum(g.home_shots + g.away_shots for g in self.games)

        lines.append(f"Total goals scored: {total_goals}")
        lines.append(f"Average goals per game: {total_goals / len(self.games):.1f}")
        lines.append(f"Average home possession: {avg_possession_home:.1f}%")
        lines.append(f"Total shots: {total_shots}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def export_to_csv(self, filepath: Path):
        """Export all game stats to CSV"""
        if not PANDAS_AVAILABLE:
            logger.error("Pandas required for CSV export")
            return

        data = []
        for game in self.games:
            data.append({
                'game_id': game.game_id,
                'date': game.date,
                'home_team': game.home_team,
                'away_team': game.away_team,
                'home_score': game.home_score,
                'away_score': game.away_score,
                'home_possession': game.home_possession,
                'away_possession': game.away_possession,
                'home_shots': game.home_shots,
                'away_shots': game.away_shots,
                'home_xg': game.home_xg,
                'away_xg': game.away_xg
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported to CSV: {filepath}")
