"""
Soccer Film Analysis - Team/Player Database
Persistent database of teams and players with historical stats
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from contextlib import contextmanager
from loguru import logger


@dataclass
class Player:
    """Represents a player in the database"""
    id: Optional[int] = None
    team_id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    jersey_number: Optional[int] = None
    position: str = ""  # GK, DEF, MID, FWD
    date_of_birth: Optional[str] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    preferred_foot: str = "right"  # left, right, both
    photo_path: Optional[str] = None
    notes: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Aggregated stats (calculated from games)
    total_games: int = 0
    total_minutes: int = 0
    total_goals: int = 0
    total_assists: int = 0
    total_shots: int = 0
    total_passes: int = 0
    total_distance_km: float = 0.0
    avg_speed_kmh: float = 0.0

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def display_name(self) -> str:
        if self.jersey_number:
            return f"#{self.jersey_number} {self.last_name}"
        return self.last_name or self.first_name


@dataclass
class Team:
    """Represents a team in the database"""
    id: Optional[int] = None
    name: str = ""
    short_name: str = ""
    primary_color: Tuple[int, int, int] = (255, 255, 255)
    secondary_color: Tuple[int, int, int] = (0, 0, 0)
    logo_path: Optional[str] = None
    coach_name: str = ""
    home_venue: str = ""
    formation: str = "4-4-2"
    notes: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Aggregated stats
    total_games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0


@dataclass
class GameRecord:
    """Represents a game record in the database"""
    id: Optional[int] = None
    home_team_id: int = 0
    away_team_id: int = 0
    game_date: str = ""
    competition: str = ""
    venue: str = ""
    home_score: int = 0
    away_score: int = 0
    video_path: Optional[str] = None
    analysis_path: Optional[str] = None
    notes: str = ""
    created_at: Optional[str] = None

    # Game stats
    home_possession: float = 50.0
    away_possession: float = 50.0
    home_shots: int = 0
    away_shots: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0


@dataclass
class PlayerGameStats:
    """Stats for a player in a specific game"""
    id: Optional[int] = None
    player_id: int = 0
    game_id: int = 0
    minutes_played: int = 0
    goals: int = 0
    assists: int = 0
    shots: int = 0
    shots_on_target: int = 0
    passes: int = 0
    pass_accuracy: float = 0.0
    tackles: int = 0
    interceptions: int = 0
    distance_km: float = 0.0
    sprints: int = 0
    max_speed_kmh: float = 0.0
    heatmap_data: Optional[str] = None  # JSON serialized


class TeamDatabase:
    """
    SQLite-based database for teams, players, and game records.
    """

    def __init__(self, db_path: Optional[Path] = None):
        from config import settings
        self.db_path = db_path or (settings.get_output_dir() / "team_database.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Teams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    short_name TEXT,
                    primary_color TEXT,
                    secondary_color TEXT,
                    logo_path TEXT,
                    coach_name TEXT,
                    home_venue TEXT,
                    formation TEXT DEFAULT '4-4-2',
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER,
                    first_name TEXT,
                    last_name TEXT NOT NULL,
                    jersey_number INTEGER,
                    position TEXT,
                    date_of_birth TEXT,
                    height_cm INTEGER,
                    weight_kg INTEGER,
                    preferred_foot TEXT DEFAULT 'right',
                    photo_path TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams(id)
                )
            """)

            # Games table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team_id INTEGER NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    game_date TEXT,
                    competition TEXT,
                    venue TEXT,
                    home_score INTEGER DEFAULT 0,
                    away_score INTEGER DEFAULT 0,
                    video_path TEXT,
                    analysis_path TEXT,
                    notes TEXT,
                    home_possession REAL DEFAULT 50.0,
                    away_possession REAL DEFAULT 50.0,
                    home_shots INTEGER DEFAULT 0,
                    away_shots INTEGER DEFAULT 0,
                    home_xg REAL DEFAULT 0.0,
                    away_xg REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (home_team_id) REFERENCES teams(id),
                    FOREIGN KEY (away_team_id) REFERENCES teams(id)
                )
            """)

            # Player game stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_game_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    game_id INTEGER NOT NULL,
                    minutes_played INTEGER DEFAULT 0,
                    goals INTEGER DEFAULT 0,
                    assists INTEGER DEFAULT 0,
                    shots INTEGER DEFAULT 0,
                    shots_on_target INTEGER DEFAULT 0,
                    passes INTEGER DEFAULT 0,
                    pass_accuracy REAL DEFAULT 0.0,
                    tackles INTEGER DEFAULT 0,
                    interceptions INTEGER DEFAULT 0,
                    distance_km REAL DEFAULT 0.0,
                    sprints INTEGER DEFAULT 0,
                    max_speed_kmh REAL DEFAULT 0.0,
                    heatmap_data TEXT,
                    FOREIGN KEY (player_id) REFERENCES players(id),
                    FOREIGN KEY (game_id) REFERENCES games(id),
                    UNIQUE(player_id, game_id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_game_stats(player_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id)")

        logger.info(f"Database initialized at: {self.db_path}")

    # ==================== TEAM OPERATIONS ====================

    def add_team(self, team: Team) -> int:
        """Add a new team and return its ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO teams (name, short_name, primary_color, secondary_color,
                                   logo_path, coach_name, home_venue, formation, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team.name, team.short_name,
                json.dumps(team.primary_color), json.dumps(team.secondary_color),
                team.logo_path, team.coach_name, team.home_venue,
                team.formation, team.notes
            ))
            team_id = cursor.lastrowid
            logger.info(f"Added team: {team.name} (ID: {team_id})")
            return team_id

    def get_team(self, team_id: int) -> Optional[Team]:
        """Get a team by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_team(row)
        return None

    def get_all_teams(self) -> List[Team]:
        """Get all teams"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams ORDER BY name")
            return [self._row_to_team(row) for row in cursor.fetchall()]

    def update_team(self, team: Team) -> bool:
        """Update an existing team"""
        if not team.id:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE teams SET
                    name = ?, short_name = ?, primary_color = ?, secondary_color = ?,
                    logo_path = ?, coach_name = ?, home_venue = ?, formation = ?,
                    notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                team.name, team.short_name,
                json.dumps(team.primary_color), json.dumps(team.secondary_color),
                team.logo_path, team.coach_name, team.home_venue,
                team.formation, team.notes, team.id
            ))
            return cursor.rowcount > 0

    def delete_team(self, team_id: int) -> bool:
        """Delete a team (also removes associated players)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM players WHERE team_id = ?", (team_id,))
            cursor.execute("DELETE FROM teams WHERE id = ?", (team_id,))
            return cursor.rowcount > 0

    def _row_to_team(self, row) -> Team:
        """Convert a database row to Team object"""
        return Team(
            id=row['id'],
            name=row['name'],
            short_name=row['short_name'] or "",
            primary_color=tuple(json.loads(row['primary_color'] or '[255,255,255]')),
            secondary_color=tuple(json.loads(row['secondary_color'] or '[0,0,0]')),
            logo_path=row['logo_path'],
            coach_name=row['coach_name'] or "",
            home_venue=row['home_venue'] or "",
            formation=row['formation'] or "4-4-2",
            notes=row['notes'] or "",
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    # ==================== PLAYER OPERATIONS ====================

    def add_player(self, player: Player) -> int:
        """Add a new player and return their ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO players (team_id, first_name, last_name, jersey_number,
                                     position, date_of_birth, height_cm, weight_kg,
                                     preferred_foot, photo_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player.team_id, player.first_name, player.last_name,
                player.jersey_number, player.position, player.date_of_birth,
                player.height_cm, player.weight_kg, player.preferred_foot,
                player.photo_path, player.notes
            ))
            player_id = cursor.lastrowid
            logger.info(f"Added player: {player.full_name} (ID: {player_id})")
            return player_id

    def get_player(self, player_id: int) -> Optional[Player]:
        """Get a player by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM players WHERE id = ?", (player_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_player(row)
        return None

    def get_players_by_team(self, team_id: int) -> List[Player]:
        """Get all players for a team"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM players WHERE team_id = ?
                ORDER BY jersey_number, last_name
            """, (team_id,))
            return [self._row_to_player(row) for row in cursor.fetchall()]

    def get_player_by_jersey(self, team_id: int, jersey_number: int) -> Optional[Player]:
        """Get a player by team and jersey number"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM players WHERE team_id = ? AND jersey_number = ?
            """, (team_id, jersey_number))
            row = cursor.fetchone()
            if row:
                return self._row_to_player(row)
        return None

    def update_player(self, player: Player) -> bool:
        """Update an existing player"""
        if not player.id:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE players SET
                    team_id = ?, first_name = ?, last_name = ?, jersey_number = ?,
                    position = ?, date_of_birth = ?, height_cm = ?, weight_kg = ?,
                    preferred_foot = ?, photo_path = ?, notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                player.team_id, player.first_name, player.last_name,
                player.jersey_number, player.position, player.date_of_birth,
                player.height_cm, player.weight_kg, player.preferred_foot,
                player.photo_path, player.notes, player.id
            ))
            return cursor.rowcount > 0

    def delete_player(self, player_id: int) -> bool:
        """Delete a player"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM player_game_stats WHERE player_id = ?", (player_id,))
            cursor.execute("DELETE FROM players WHERE id = ?", (player_id,))
            return cursor.rowcount > 0

    def _row_to_player(self, row) -> Player:
        """Convert a database row to Player object"""
        return Player(
            id=row['id'],
            team_id=row['team_id'],
            first_name=row['first_name'] or "",
            last_name=row['last_name'],
            jersey_number=row['jersey_number'],
            position=row['position'] or "",
            date_of_birth=row['date_of_birth'],
            height_cm=row['height_cm'],
            weight_kg=row['weight_kg'],
            preferred_foot=row['preferred_foot'] or "right",
            photo_path=row['photo_path'],
            notes=row['notes'] or "",
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    # ==================== GAME OPERATIONS ====================

    def add_game(self, game: GameRecord) -> int:
        """Add a new game record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO games (home_team_id, away_team_id, game_date, competition,
                                   venue, home_score, away_score, video_path,
                                   analysis_path, notes, home_possession, away_possession,
                                   home_shots, away_shots, home_xg, away_xg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game.home_team_id, game.away_team_id, game.game_date,
                game.competition, game.venue, game.home_score, game.away_score,
                game.video_path, game.analysis_path, game.notes,
                game.home_possession, game.away_possession,
                game.home_shots, game.away_shots, game.home_xg, game.away_xg
            ))
            game_id = cursor.lastrowid
            logger.info(f"Added game record (ID: {game_id})")
            return game_id

    def get_game(self, game_id: int) -> Optional[GameRecord]:
        """Get a game by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_game(row)
        return None

    def get_games_by_team(self, team_id: int, limit: int = 50) -> List[GameRecord]:
        """Get games involving a team"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM games
                WHERE home_team_id = ? OR away_team_id = ?
                ORDER BY game_date DESC
                LIMIT ?
            """, (team_id, team_id, limit))
            return [self._row_to_game(row) for row in cursor.fetchall()]

    def get_recent_games(self, limit: int = 20) -> List[GameRecord]:
        """Get most recent games"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM games ORDER BY game_date DESC LIMIT ?
            """, (limit,))
            return [self._row_to_game(row) for row in cursor.fetchall()]

    def _row_to_game(self, row) -> GameRecord:
        """Convert a database row to GameRecord object"""
        return GameRecord(
            id=row['id'],
            home_team_id=row['home_team_id'],
            away_team_id=row['away_team_id'],
            game_date=row['game_date'] or "",
            competition=row['competition'] or "",
            venue=row['venue'] or "",
            home_score=row['home_score'] or 0,
            away_score=row['away_score'] or 0,
            video_path=row['video_path'],
            analysis_path=row['analysis_path'],
            notes=row['notes'] or "",
            created_at=row['created_at'],
            home_possession=row['home_possession'] or 50.0,
            away_possession=row['away_possession'] or 50.0,
            home_shots=row['home_shots'] or 0,
            away_shots=row['away_shots'] or 0,
            home_xg=row['home_xg'] or 0.0,
            away_xg=row['away_xg'] or 0.0
        )

    # ==================== PLAYER STATS OPERATIONS ====================

    def add_player_game_stats(self, stats: PlayerGameStats) -> int:
        """Add player stats for a game"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO player_game_stats
                (player_id, game_id, minutes_played, goals, assists, shots,
                 shots_on_target, passes, pass_accuracy, tackles, interceptions,
                 distance_km, sprints, max_speed_kmh, heatmap_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.player_id, stats.game_id, stats.minutes_played,
                stats.goals, stats.assists, stats.shots, stats.shots_on_target,
                stats.passes, stats.pass_accuracy, stats.tackles,
                stats.interceptions, stats.distance_km, stats.sprints,
                stats.max_speed_kmh, stats.heatmap_data
            ))
            return cursor.lastrowid

    def get_player_career_stats(self, player_id: int) -> Dict:
        """Get aggregated career stats for a player"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as games,
                    SUM(minutes_played) as total_minutes,
                    SUM(goals) as total_goals,
                    SUM(assists) as total_assists,
                    SUM(shots) as total_shots,
                    SUM(passes) as total_passes,
                    AVG(pass_accuracy) as avg_pass_accuracy,
                    SUM(distance_km) as total_distance,
                    AVG(max_speed_kmh) as avg_max_speed,
                    SUM(sprints) as total_sprints
                FROM player_game_stats
                WHERE player_id = ?
            """, (player_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_team_stats(self, team_id: int) -> Dict:
        """Get aggregated stats for a team"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_games,
                    SUM(CASE WHEN home_team_id = ? AND home_score > away_score THEN 1
                             WHEN away_team_id = ? AND away_score > home_score THEN 1
                             ELSE 0 END) as wins,
                    SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN home_team_id = ? AND home_score < away_score THEN 1
                             WHEN away_team_id = ? AND away_score < home_score THEN 1
                             ELSE 0 END) as losses,
                    SUM(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as goals_for,
                    SUM(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as goals_against,
                    AVG(CASE WHEN home_team_id = ? THEN home_possession ELSE away_possession END) as avg_possession,
                    AVG(CASE WHEN home_team_id = ? THEN home_xg ELSE away_xg END) as avg_xg
                FROM games
                WHERE home_team_id = ? OR away_team_id = ?
            """, (team_id,) * 10)
            row = cursor.fetchone()
            return dict(row) if row else {}

    # ==================== IMPORT/EXPORT ====================

    def import_roster_csv(self, csv_path: Path, team_id: int) -> int:
        """Import players from a CSV file"""
        import csv
        count = 0
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                player = Player(
                    team_id=team_id,
                    first_name=row.get('first_name', row.get('First Name', '')),
                    last_name=row.get('last_name', row.get('Last Name', '')),
                    jersey_number=int(row.get('number', row.get('Number', row.get('#', 0))) or 0),
                    position=row.get('position', row.get('Position', '')),
                )
                self.add_player(player)
                count += 1
        logger.info(f"Imported {count} players from {csv_path}")
        return count

    def export_to_json(self, filepath: Path):
        """Export entire database to JSON"""
        data = {
            'teams': [asdict(t) for t in self.get_all_teams()],
            'players': [],
            'games': [asdict(g) for g in self.get_recent_games(1000)],
        }
        for team in self.get_all_teams():
            for player in self.get_players_by_team(team.id):
                player_dict = asdict(player)
                player_dict['career_stats'] = self.get_player_career_stats(player.id)
                data['players'].append(player_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported database to {filepath}")
