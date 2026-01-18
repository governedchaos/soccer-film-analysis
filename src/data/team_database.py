"""
Soccer Film Analysis - Team/Player Database
PostgreSQL-based persistent database of teams and players with historical stats
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from contextlib import contextmanager
from loguru import logger

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")


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
    PostgreSQL-based database for teams, players, and game records.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "soccer_analysis",
        user: str = "postgres",
        password: str = "",
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize database connection.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 is required. Install with: pip install psycopg2-binary")

        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

        # Try to load from config if available
        try:
            from config import settings
            if hasattr(settings, 'database'):
                db_config = settings.database
                self.connection_params.update({
                    'host': getattr(db_config, 'host', host),
                    'port': getattr(db_config, 'port', port),
                    'database': getattr(db_config, 'name', database),
                    'user': getattr(db_config, 'user', user),
                    'password': getattr(db_config, 'password', password)
                })
        except (ImportError, AttributeError):
            pass

        # Create connection pool
        try:
            self.pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                **self.connection_params
            )
            logger.info(f"Connected to PostgreSQL: {self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}")
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.info("Attempting to create database...")
            self._create_database()
            self.pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                **self.connection_params
            )

        self._init_database()

    def _create_database(self):
        """Create database if it doesn't exist"""
        params = self.connection_params.copy()
        db_name = params.pop('database')

        # Connect to default 'postgres' database
        params['database'] = 'postgres'
        conn = psycopg2.connect(**params)
        conn.autocommit = True

        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if not cursor.fetchone():
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Created database: {db_name}")
        finally:
            conn.close()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections from pool"""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.putconn(conn)

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Teams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    short_name VARCHAR(50),
                    primary_color JSONB DEFAULT '[255, 255, 255]',
                    secondary_color JSONB DEFAULT '[0, 0, 0]',
                    logo_path TEXT,
                    coach_name VARCHAR(255),
                    home_venue VARCHAR(255),
                    formation VARCHAR(20) DEFAULT '4-4-2',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id SERIAL PRIMARY KEY,
                    team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
                    first_name VARCHAR(100),
                    last_name VARCHAR(100) NOT NULL,
                    jersey_number INTEGER,
                    position VARCHAR(20),
                    date_of_birth DATE,
                    height_cm INTEGER,
                    weight_kg INTEGER,
                    preferred_foot VARCHAR(10) DEFAULT 'right',
                    photo_path TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Games table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id SERIAL PRIMARY KEY,
                    home_team_id INTEGER NOT NULL REFERENCES teams(id),
                    away_team_id INTEGER NOT NULL REFERENCES teams(id),
                    game_date DATE,
                    competition VARCHAR(255),
                    venue VARCHAR(255),
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Player game stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_game_stats (
                    id SERIAL PRIMARY KEY,
                    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
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
                    heatmap_data JSONB,
                    UNIQUE(player_id, game_id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_game_stats(player_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id)")

            # Create updated_at trigger function
            cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)

            # Add triggers for updated_at
            for table in ['teams', 'players']:
                cursor.execute(f"""
                    DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
                    CREATE TRIGGER update_{table}_updated_at
                        BEFORE UPDATE ON {table}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                """)

        logger.info("Database schema initialized")

    def close(self):
        """Close all database connections"""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            logger.info("Database connections closed")

    # ==================== TEAM OPERATIONS ====================

    def add_team(self, team: Team) -> int:
        """Add a new team and return its ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO teams (name, short_name, primary_color, secondary_color,
                                   logo_path, coach_name, home_venue, formation, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                team.name, team.short_name,
                json.dumps(list(team.primary_color)),
                json.dumps(list(team.secondary_color)),
                team.logo_path, team.coach_name, team.home_venue,
                team.formation, team.notes
            ))
            team_id = cursor.fetchone()[0]
            logger.info(f"Added team: {team.name} (ID: {team_id})")
            return team_id

    def get_team(self, team_id: int) -> Optional[Team]:
        """Get a team by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM teams WHERE id = %s", (team_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_team(row)
        return None

    def get_team_by_name(self, name: str) -> Optional[Team]:
        """Get a team by name"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM teams WHERE name ILIKE %s", (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_team(row)
        return None

    def get_all_teams(self) -> List[Team]:
        """Get all teams"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
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
                    name = %s, short_name = %s, primary_color = %s, secondary_color = %s,
                    logo_path = %s, coach_name = %s, home_venue = %s, formation = %s,
                    notes = %s
                WHERE id = %s
            """, (
                team.name, team.short_name,
                json.dumps(list(team.primary_color)),
                json.dumps(list(team.secondary_color)),
                team.logo_path, team.coach_name, team.home_venue,
                team.formation, team.notes, team.id
            ))
            return cursor.rowcount > 0

    def delete_team(self, team_id: int) -> bool:
        """Delete a team (also removes associated players)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM players WHERE team_id = %s", (team_id,))
            cursor.execute("DELETE FROM teams WHERE id = %s", (team_id,))
            return cursor.rowcount > 0

    def _row_to_team(self, row: dict) -> Team:
        """Convert a database row to Team object"""
        primary_color = row.get('primary_color', [255, 255, 255])
        if isinstance(primary_color, str):
            primary_color = json.loads(primary_color)

        secondary_color = row.get('secondary_color', [0, 0, 0])
        if isinstance(secondary_color, str):
            secondary_color = json.loads(secondary_color)

        return Team(
            id=row['id'],
            name=row['name'],
            short_name=row.get('short_name') or "",
            primary_color=tuple(primary_color),
            secondary_color=tuple(secondary_color),
            logo_path=row.get('logo_path'),
            coach_name=row.get('coach_name') or "",
            home_venue=row.get('home_venue') or "",
            formation=row.get('formation') or "4-4-2",
            notes=row.get('notes') or "",
            created_at=str(row.get('created_at')) if row.get('created_at') else None,
            updated_at=str(row.get('updated_at')) if row.get('updated_at') else None
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                player.team_id, player.first_name, player.last_name,
                player.jersey_number, player.position, player.date_of_birth,
                player.height_cm, player.weight_kg, player.preferred_foot,
                player.photo_path, player.notes
            ))
            player_id = cursor.fetchone()[0]
            logger.info(f"Added player: {player.full_name} (ID: {player_id})")
            return player_id

    def get_player(self, player_id: int) -> Optional[Player]:
        """Get a player by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM players WHERE id = %s", (player_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_player(row)
        return None

    def get_players_by_team(self, team_id: int) -> List[Player]:
        """Get all players for a team"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM players WHERE team_id = %s
                ORDER BY jersey_number, last_name
            """, (team_id,))
            return [self._row_to_player(row) for row in cursor.fetchall()]

    def get_player_by_jersey(self, team_id: int, jersey_number: int) -> Optional[Player]:
        """Get a player by team and jersey number"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM players WHERE team_id = %s AND jersey_number = %s
            """, (team_id, jersey_number))
            row = cursor.fetchone()
            if row:
                return self._row_to_player(row)
        return None

    def search_players(self, query: str, team_id: Optional[int] = None) -> List[Player]:
        """Search players by name"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if team_id:
                cursor.execute("""
                    SELECT * FROM players
                    WHERE team_id = %s AND (
                        first_name ILIKE %s OR last_name ILIKE %s
                    )
                    ORDER BY last_name
                """, (team_id, f"%{query}%", f"%{query}%"))
            else:
                cursor.execute("""
                    SELECT * FROM players
                    WHERE first_name ILIKE %s OR last_name ILIKE %s
                    ORDER BY last_name
                """, (f"%{query}%", f"%{query}%"))
            return [self._row_to_player(row) for row in cursor.fetchall()]

    def update_player(self, player: Player) -> bool:
        """Update an existing player"""
        if not player.id:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE players SET
                    team_id = %s, first_name = %s, last_name = %s, jersey_number = %s,
                    position = %s, date_of_birth = %s, height_cm = %s, weight_kg = %s,
                    preferred_foot = %s, photo_path = %s, notes = %s
                WHERE id = %s
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
            cursor.execute("DELETE FROM player_game_stats WHERE player_id = %s", (player_id,))
            cursor.execute("DELETE FROM players WHERE id = %s", (player_id,))
            return cursor.rowcount > 0

    def _row_to_player(self, row: dict) -> Player:
        """Convert a database row to Player object"""
        return Player(
            id=row['id'],
            team_id=row.get('team_id'),
            first_name=row.get('first_name') or "",
            last_name=row['last_name'],
            jersey_number=row.get('jersey_number'),
            position=row.get('position') or "",
            date_of_birth=str(row['date_of_birth']) if row.get('date_of_birth') else None,
            height_cm=row.get('height_cm'),
            weight_kg=row.get('weight_kg'),
            preferred_foot=row.get('preferred_foot') or "right",
            photo_path=row.get('photo_path'),
            notes=row.get('notes') or "",
            created_at=str(row.get('created_at')) if row.get('created_at') else None,
            updated_at=str(row.get('updated_at')) if row.get('updated_at') else None
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                game.home_team_id, game.away_team_id, game.game_date,
                game.competition, game.venue, game.home_score, game.away_score,
                game.video_path, game.analysis_path, game.notes,
                game.home_possession, game.away_possession,
                game.home_shots, game.away_shots, game.home_xg, game.away_xg
            ))
            game_id = cursor.fetchone()[0]
            logger.info(f"Added game record (ID: {game_id})")
            return game_id

    def get_game(self, game_id: int) -> Optional[GameRecord]:
        """Get a game by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM games WHERE id = %s", (game_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_game(row)
        return None

    def get_games_by_team(self, team_id: int, limit: int = 50) -> List[GameRecord]:
        """Get games involving a team"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM games
                WHERE home_team_id = %s OR away_team_id = %s
                ORDER BY game_date DESC
                LIMIT %s
            """, (team_id, team_id, limit))
            return [self._row_to_game(row) for row in cursor.fetchall()]

    def get_recent_games(self, limit: int = 20) -> List[GameRecord]:
        """Get most recent games"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM games ORDER BY game_date DESC LIMIT %s
            """, (limit,))
            return [self._row_to_game(row) for row in cursor.fetchall()]

    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> List[GameRecord]:
        """Get head-to-head games between two teams"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM games
                WHERE (home_team_id = %s AND away_team_id = %s)
                   OR (home_team_id = %s AND away_team_id = %s)
                ORDER BY game_date DESC
                LIMIT %s
            """, (team1_id, team2_id, team2_id, team1_id, limit))
            return [self._row_to_game(row) for row in cursor.fetchall()]

    def _row_to_game(self, row: dict) -> GameRecord:
        """Convert a database row to GameRecord object"""
        return GameRecord(
            id=row['id'],
            home_team_id=row['home_team_id'],
            away_team_id=row['away_team_id'],
            game_date=str(row['game_date']) if row.get('game_date') else "",
            competition=row.get('competition') or "",
            venue=row.get('venue') or "",
            home_score=row.get('home_score') or 0,
            away_score=row.get('away_score') or 0,
            video_path=row.get('video_path'),
            analysis_path=row.get('analysis_path'),
            notes=row.get('notes') or "",
            created_at=str(row.get('created_at')) if row.get('created_at') else None,
            home_possession=row.get('home_possession') or 50.0,
            away_possession=row.get('away_possession') or 50.0,
            home_shots=row.get('home_shots') or 0,
            away_shots=row.get('away_shots') or 0,
            home_xg=row.get('home_xg') or 0.0,
            away_xg=row.get('away_xg') or 0.0
        )

    # ==================== PLAYER STATS OPERATIONS ====================

    def add_player_game_stats(self, stats: PlayerGameStats) -> int:
        """Add or update player stats for a game"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO player_game_stats
                (player_id, game_id, minutes_played, goals, assists, shots,
                 shots_on_target, passes, pass_accuracy, tackles, interceptions,
                 distance_km, sprints, max_speed_kmh, heatmap_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, game_id) DO UPDATE SET
                    minutes_played = EXCLUDED.minutes_played,
                    goals = EXCLUDED.goals,
                    assists = EXCLUDED.assists,
                    shots = EXCLUDED.shots,
                    shots_on_target = EXCLUDED.shots_on_target,
                    passes = EXCLUDED.passes,
                    pass_accuracy = EXCLUDED.pass_accuracy,
                    tackles = EXCLUDED.tackles,
                    interceptions = EXCLUDED.interceptions,
                    distance_km = EXCLUDED.distance_km,
                    sprints = EXCLUDED.sprints,
                    max_speed_kmh = EXCLUDED.max_speed_kmh,
                    heatmap_data = EXCLUDED.heatmap_data
                RETURNING id
            """, (
                stats.player_id, stats.game_id, stats.minutes_played,
                stats.goals, stats.assists, stats.shots, stats.shots_on_target,
                stats.passes, stats.pass_accuracy, stats.tackles,
                stats.interceptions, stats.distance_km, stats.sprints,
                stats.max_speed_kmh, stats.heatmap_data
            ))
            return cursor.fetchone()[0]

    def get_player_game_stats(self, player_id: int, game_id: int) -> Optional[PlayerGameStats]:
        """Get stats for a player in a specific game"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM player_game_stats
                WHERE player_id = %s AND game_id = %s
            """, (player_id, game_id))
            row = cursor.fetchone()
            if row:
                return self._row_to_player_stats(row)
        return None

    def get_player_career_stats(self, player_id: int) -> Dict:
        """Get aggregated career stats for a player"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT
                    COUNT(*) as games,
                    COALESCE(SUM(minutes_played), 0) as total_minutes,
                    COALESCE(SUM(goals), 0) as total_goals,
                    COALESCE(SUM(assists), 0) as total_assists,
                    COALESCE(SUM(shots), 0) as total_shots,
                    COALESCE(SUM(passes), 0) as total_passes,
                    COALESCE(AVG(pass_accuracy), 0) as avg_pass_accuracy,
                    COALESCE(SUM(distance_km), 0) as total_distance,
                    COALESCE(AVG(max_speed_kmh), 0) as avg_max_speed,
                    COALESCE(SUM(sprints), 0) as total_sprints,
                    COALESCE(SUM(tackles), 0) as total_tackles,
                    COALESCE(SUM(interceptions), 0) as total_interceptions
                FROM player_game_stats
                WHERE player_id = %s
            """, (player_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_team_stats(self, team_id: int) -> Dict:
        """Get aggregated stats for a team"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT
                    COUNT(*) as total_games,
                    SUM(CASE WHEN home_team_id = %s AND home_score > away_score THEN 1
                             WHEN away_team_id = %s AND away_score > home_score THEN 1
                             ELSE 0 END) as wins,
                    SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN home_team_id = %s AND home_score < away_score THEN 1
                             WHEN away_team_id = %s AND away_score < home_score THEN 1
                             ELSE 0 END) as losses,
                    SUM(CASE WHEN home_team_id = %s THEN home_score ELSE away_score END) as goals_for,
                    SUM(CASE WHEN home_team_id = %s THEN away_score ELSE home_score END) as goals_against,
                    AVG(CASE WHEN home_team_id = %s THEN home_possession ELSE away_possession END) as avg_possession,
                    AVG(CASE WHEN home_team_id = %s THEN home_xg ELSE away_xg END) as avg_xg
                FROM games
                WHERE home_team_id = %s OR away_team_id = %s
            """, (team_id,) * 10)
            row = cursor.fetchone()
            return dict(row) if row else {}

    def _row_to_player_stats(self, row: dict) -> PlayerGameStats:
        """Convert a database row to PlayerGameStats object"""
        return PlayerGameStats(
            id=row['id'],
            player_id=row['player_id'],
            game_id=row['game_id'],
            minutes_played=row.get('minutes_played') or 0,
            goals=row.get('goals') or 0,
            assists=row.get('assists') or 0,
            shots=row.get('shots') or 0,
            shots_on_target=row.get('shots_on_target') or 0,
            passes=row.get('passes') or 0,
            pass_accuracy=row.get('pass_accuracy') or 0.0,
            tackles=row.get('tackles') or 0,
            interceptions=row.get('interceptions') or 0,
            distance_km=row.get('distance_km') or 0.0,
            sprints=row.get('sprints') or 0,
            max_speed_kmh=row.get('max_speed_kmh') or 0.0,
            heatmap_data=row.get('heatmap_data')
        )

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
            'teams': [],
            'players': [],
            'games': [asdict(g) for g in self.get_recent_games(1000)],
        }

        for team in self.get_all_teams():
            team_dict = asdict(team)
            team_dict['stats'] = self.get_team_stats(team.id)
            data['teams'].append(team_dict)

            for player in self.get_players_by_team(team.id):
                player_dict = asdict(player)
                player_dict['career_stats'] = self.get_player_career_stats(player.id)
                data['players'].append(player_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported database to {filepath}")
