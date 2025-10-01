
"""

# ============================================================================
# PART 1: CONFIGURATION, ENUMS, AND BASE DATA CLASSES
# ============================================================================
# This part contains the foundational configuration and data structures
# Libraries imported here will be used throughout all parts

import pandas as pd
import numpy as np
import pulp
import json
import hashlib
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION CLASS
# ============================================================================

class OptimizerConfig:
    """
    Enhanced configuration with improved ownership projection and optimization modes.

    Key improvements:
    - Dynamic ownership calculation based on multiple factors
    - Field-specific configurations for different tournament types
    - Optimization modes for different strategies
    """

    # Core DraftKings Showdown constraints
    SALARY_CAP = 50000
    MIN_SALARY = 3000
    MAX_SALARY = 12000
    CAPTAIN_MULTIPLIER = 1.5
    ROSTER_SIZE = 6
    FLEX_SPOTS = 5

    # DraftKings Showdown specific rules
    MIN_TEAMS_REQUIRED = 2  # Must have players from both teams
    MAX_PLAYERS_PER_TEAM = 5  # Max 5 from one team (leaving 1 for opponent)

    # Performance settings
    MAX_ITERATIONS = 1000
    OPTIMIZATION_TIMEOUT = 30  # seconds per lineup
    MAX_PARALLEL_THREADS = 4  # Default, can be adjusted
    MAX_HISTORY_ENTRIES = 50
    CACHE_SIZE = 100

    # Enhanced ownership projection system
    # Base ownership by position with salary correlation
    OWNERSHIP_BY_POSITION = {
        'QB': {'base': 15, 'salary_factor': 0.002, 'scarcity_multiplier': 1.2},
        'RB': {'base': 12, 'salary_factor': 0.0015, 'scarcity_multiplier': 1.0},
        'WR': {'base': 10, 'salary_factor': 0.0018, 'scarcity_multiplier': 0.95},
        'TE': {'base': 8, 'salary_factor': 0.001, 'scarcity_multiplier': 1.1},
        'DST': {'base': 5, 'salary_factor': 0.0005, 'scarcity_multiplier': 1.0},
        'K': {'base': 3, 'salary_factor': 0.0003, 'scarcity_multiplier': 0.9},
        'FLEX': {'base': 5, 'salary_factor': 0.001, 'scarcity_multiplier': 1.0}
    }

    @classmethod
    def get_default_ownership(cls, position: str, salary: float,
                            game_total: float = 47.0,
                            is_favorite: bool = False,
                            injury_news: bool = False) -> float:
        """
        Enhanced ownership projection with multiple factors.

        DFS Value: Better ownership estimates = better leverage opportunities

        Args:
            position: Player position
            salary: Player salary
            game_total: Total points in game (higher = more ownership)
            is_favorite: Whether player is on favored team
            injury_news: Whether there's recent injury news affecting player

        Returns:
            Projected ownership percentage (0-100)
        """
        pos_config = cls.OWNERSHIP_BY_POSITION.get(
            position,
            cls.OWNERSHIP_BY_POSITION['FLEX']
        )

        base = pos_config['base']
        salary_factor = pos_config['salary_factor']
        scarcity = pos_config['scarcity_multiplier']

        # Salary adjustment (higher salary = higher ownership generally)
        salary_adjustment = (salary - 5000) * salary_factor

        # Game total adjustment (high-scoring games = more ownership on all players)
        total_adjustment = (game_total - 47.0) * 0.15

        # Favorite adjustment (favored team gets small ownership boost)
        favorite_bonus = 2.0 if is_favorite else -1.0

        # Injury news creates ownership volatility
        injury_adjustment = np.random.uniform(-3.0, 5.0) if injury_news else 0

        # Calculate base ownership
        ownership = (base + salary_adjustment + total_adjustment +
                    favorite_bonus + injury_adjustment) * scarcity

        # Add randomness for variation (important for diverse lineups)
        random_factor = np.random.normal(1.0, 0.08)
        ownership *= random_factor

        # Bound between realistic limits
        return max(0.5, min(50.0, ownership))

    # Contest field sizes mapped to internal configurations
    FIELD_SIZES = {
        'Single Entry': 'small_field',
        '3-Max': 'small_field',
        '5-Max': 'small_field',
        '20-Max': 'medium_field',
        '150-Max': 'large_field',
        'Large GPP (1000+)': 'large_field_aggressive',
        'Milly Maker': 'milly_maker',
        'Showdown Special': 'large_field_aggressive'
    }

    # Optimization modes for different ceiling/floor strategies
    OPTIMIZATION_MODES = {
        'balanced': {'ceiling_weight': 0.5, 'floor_weight': 0.5},
        'ceiling': {'ceiling_weight': 0.8, 'floor_weight': 0.2},
        'floor': {'ceiling_weight': 0.2, 'floor_weight': 0.8},
        'boom_or_bust': {'ceiling_weight': 1.0, 'floor_weight': 0.0}
    }

    # GPP Ownership targets by field size
    # DFS Value: Different field sizes require different ownership strategies
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (70, 110),          # Higher ownership acceptable in small fields
        'medium_field': (60, 90),          # Balanced approach
        'large_field': (50, 80),           # Lower ownership for differentiation
        'large_field_aggressive': (40, 70), # Very low ownership strategy
        'milly_maker': (30, 60),           # Ultra contrarian for massive fields
        'super_contrarian': (20, 50)       # Extreme leverage plays
    }

    # Field-specific AI configurations
    # DFS Value: Each tournament type requires different strategic approach
    FIELD_SIZE_CONFIGS = {
        'small_field': {
            'max_exposure': 0.4,              # Can use same captain in 40% of lineups
            'min_unique_captains': 5,
            'max_chalk_players': 3,           # Can use more chalk in small fields
            'min_leverage_players': 1,
            'ownership_leverage_weight': 0.3,
            'correlation_weight': 0.4,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Moderate',
            'min_total_ownership': 70,
            'max_total_ownership': 110,
            'similarity_threshold': 0.7       # Allow more similar lineups
        },
        'medium_field': {
            'max_exposure': 0.3,
            'min_unique_captains': 10,
            'max_chalk_players': 2,
            'min_leverage_players': 2,
            'ownership_leverage_weight': 0.35,
            'correlation_weight': 0.35,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Strong',
            'min_total_ownership': 60,
            'max_total_ownership': 90,
            'similarity_threshold': 0.67
        },
        'large_field': {
            'max_exposure': 0.25,
            'min_unique_captains': 15,
            'max_chalk_players': 2,
            'min_leverage_players': 2,
            'ownership_leverage_weight': 0.4,
            'correlation_weight': 0.3,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Strong',
            'min_total_ownership': 50,
            'max_total_ownership': 80,
            'similarity_threshold': 0.67
        },
        'large_field_aggressive': {
            'max_exposure': 0.2,
            'min_unique_captains': 20,
            'max_chalk_players': 1,
            'min_leverage_players': 3,
            'ownership_leverage_weight': 0.45,
            'correlation_weight': 0.25,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Mandatory',
            'min_total_ownership': 40,
            'max_total_ownership': 70,
            'similarity_threshold': 0.6
        },
        'milly_maker': {
            'max_exposure': 0.15,
            'min_unique_captains': 30,
            'max_chalk_players': 1,
            'min_leverage_players': 4,
            'ownership_leverage_weight': 0.5,
            'correlation_weight': 0.2,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Mandatory',
            'min_total_ownership': 30,
            'max_total_ownership': 60,
            'similarity_threshold': 0.5
        }
    }

    # AI system weights for combining multiple AI recommendations
    AI_WEIGHTS = {
        'game_theory': 0.35,
        'correlation': 0.35,
        'contrarian': 0.30
    }

    # Sport-specific configurations (foundation for future expansion)
    SPORT_CONFIGS = {
        'NFL': {
            'roster_size': 6,
            'salary_cap': 50000,
            'positions': ['QB', 'RB', 'WR', 'TE', 'DST'],
            'scoring': 'DK_NFL'
        }
    }

# ============================================================================
# ENUM CLASSES
# ============================================================================

class AIStrategistType(Enum):
    """Types of AI strategists in the system"""
    GAME_THEORY = "Game Theory"
    CORRELATION = "Correlation"
    CONTRARIAN_NARRATIVE = "Contrarian Narrative"

class AIEnforcementLevel(Enum):
    """
    AI enforcement levels - how strictly to apply AI recommendations

    DFS Value: Different levels allow flexibility vs. control tradeoff
    """
    ADVISORY = "Advisory"      # AI suggestions only, no hard constraints
    MODERATE = "Moderate"      # Some constraints enforced
    STRONG = "Strong"          # Most constraints enforced
    MANDATORY = "Mandatory"    # All AI decisions enforced as hard constraints

class OptimizationMode(Enum):
    """Optimization modes for different strategies"""
    BALANCED = "balanced"
    CEILING = "ceiling"
    FLOOR = "floor"
    BOOM_OR_BUST = "boom_or_bust"

class StackType(Enum):
    """
    Enhanced stack types for different game scenarios

    DFS Value: Each stack type captures different correlated outcomes
    """
    QB_RECEIVER = "qb_receiver"      # Standard positive correlation
    ONSLAUGHT = "onslaught"          # 3+ players from winning team
    LEVERAGE = "leverage"            # Low-owned correlated plays
    BRING_BACK = "bring_back"        # Primary stack + opponent
    DEFENSIVE = "defensive"          # DST with negative game script
    HIDDEN = "hidden"                # Non-obvious correlations

class ConstraintPriority(Enum):
    """
    Constraint priority levels for tiered relaxation

    DFS Value: CRITICAL - enables three-tier constraint relaxation
    This allows maintaining DK rules while relaxing AI preferences
    """
    CRITICAL = 100              # DK rules, salary cap - NEVER relax
    AI_HIGH_CONFIDENCE = 90     # AI recommendations with >0.8 confidence
    AI_CONSENSUS = 85           # Multiple AIs agree
    AI_MODERATE = 70            # Single AI recommendation
    SOFT_PREFERENCE = 50        # Suggestions only

# ============================================================================
# BASE DATA CLASSES
# ============================================================================

@dataclass
class AIRecommendation:
    """
    Enhanced AI recommendation with validation and metadata

    DFS Value: Structured recommendations enable enforcement and tracking
    """
    captain_targets: List[str] = field(default_factory=list)
    must_play: List[str] = field(default_factory=list)
    never_play: List[str] = field(default_factory=list)
    stacks: List[Dict] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.5
    enforcement_rules: List[Dict] = field(default_factory=list)
    narrative: str = ""
    source_ai: Optional[AIStrategistType] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Enhanced attributes for tracking
    ownership_leverage: Dict = field(default_factory=dict)
    correlation_matrix: Dict = field(default_factory=dict)
    contrarian_angles: List[str] = field(default_factory=list)
    ceiling_plays: List[str] = field(default_factory=list)
    floor_plays: List[str] = field(default_factory=list)
    boom_bust_candidates: List[str] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Enhanced validation with detailed error reporting

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Check for empty recommendations
        if not self.captain_targets and not self.must_play:
            errors.append("No captain targets or must-play players specified")

        # Validate confidence bounds
        if not 0 <= self.confidence <= 1:
            errors.append(f"Invalid confidence score: {self.confidence}")
            self.confidence = max(0, min(1, self.confidence))

        # Check for conflicts between must_play and never_play
        conflicts = set(self.must_play) & set(self.never_play)
        if conflicts:
            errors.append(f"Conflicting players in must/never play: {conflicts}")

        # Validate stack structures
        for stack in self.stacks:
            if not isinstance(stack, dict):
                errors.append("Invalid stack format - must be dictionary")
            elif 'players' in stack and len(stack['players']) < 2:
                errors.append("Stack must have at least 2 players")
            elif 'player1' in stack and 'player2' not in stack:
                errors.append("Stack missing player2")

        # Validate enforcement rules
        for rule in self.enforcement_rules:
            if 'type' not in rule or 'constraint' not in rule:
                errors.append("Invalid enforcement rule format")
            if rule.get('type') not in ['hard', 'soft']:
                errors.append(f"Invalid rule type: {rule.get('type')}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'captain_targets': self.captain_targets,
            'must_play': self.must_play,
            'never_play': self.never_play,
            'stacks': self.stacks,
            'key_insights': self.key_insights,
            'confidence': self.confidence,
            'enforcement_rules': self.enforcement_rules,
            'narrative': self.narrative,
            'source_ai': self.source_ai.value if self.source_ai else None,
            'timestamp': self.timestamp.isoformat(),
            'ownership_leverage': self.ownership_leverage,
            'correlation_matrix': self.correlation_matrix,
            'contrarian_angles': self.contrarian_angles
        }

@dataclass
class LineupConstraints:
    """
    Enhanced constraints for lineup generation

    DFS Value: Flexible constraint system allows nuanced lineup construction
    """
    min_salary: int = 48000
    max_salary: int = 50000
    min_projection: float = 0
    max_ownership: float = 200
    min_ownership: float = 0
    required_positions: Dict[str, int] = field(default_factory=dict)
    banned_players: Set[str] = field(default_factory=set)
    locked_players: Set[str] = field(default_factory=set)

    # Enhanced constraints
    required_stacks: List[Dict] = field(default_factory=list)
    max_exposure: Dict[str, float] = field(default_factory=dict)
    team_limits: Dict[str, int] = field(default_factory=dict)
    correlation_requirements: List[Dict] = field(default_factory=list)
    ownership_buckets: Dict[str, List[str]] = field(default_factory=dict)

    def validate_lineup(self, lineup: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a lineup against constraints

        Args:
            lineup: Lineup dictionary with Captain, FLEX, metrics

        Returns:
            (is_valid, list of violations)
        """
        errors = []

        # Salary validation
        total_salary = lineup.get('Salary', 0)
        if total_salary < self.min_salary:
            errors.append(f"Salary too low: ${total_salary:,} < ${self.min_salary:,}")
        if total_salary > self.max_salary:
            errors.append(f"Salary too high: ${total_salary:,} > ${self.max_salary:,}")

        # Ownership validation
        total_ownership = lineup.get('Total_Ownership', 0)
        if total_ownership > self.max_ownership:
            errors.append(f"Ownership too high: {total_ownership:.1f}%")
        if total_ownership < self.min_ownership:
            errors.append(f"Ownership too low: {total_ownership:.1f}%")

        # Player validation
        all_players = [lineup.get('Captain')] + lineup.get('FLEX', [])

        for banned in self.banned_players:
            if banned in all_players:
                errors.append(f"Banned player in lineup: {banned}")

        for required in self.locked_players:
            if required not in all_players:
                errors.append(f"Required player missing: {required}")

        return len(errors) == 0, errors

@dataclass
class PerformanceMetrics:
    """
    Track optimizer performance metrics

    DFS Value: Understanding what works enables improvement over time
    """
    lineup_generation_time: float = 0
    total_iterations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    constraint_violations: int = 0
    successful_lineups: int = 0
    failed_lineups: int = 0

    # Enhanced metrics
    ai_api_calls: int = 0
    ai_cache_hits: int = 0
    average_confidence: float = 0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    stack_success_rate: float = 0
    ownership_accuracy: float = 0

    def calculate_efficiency(self) -> float:
        """Calculate lineup generation efficiency"""
        if self.total_iterations == 0:
            return 0
        return self.successful_lineups / self.total_iterations

    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache effectiveness"""
        total_cache_attempts = self.cache_hits + self.cache_misses
        if total_cache_attempts == 0:
            return 0
        return self.cache_hits / total_cache_attempts

    def get_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'efficiency': self.calculate_efficiency(),
            'avg_time_per_lineup': (
                self.lineup_generation_time / max(self.successful_lineups, 1)
            ),
            'cache_hit_rate': self.calculate_cache_hit_rate(),
            'success_rate': (
                self.successful_lineups /
                max(self.successful_lineups + self.failed_lineups, 1)
            ),
            'ai_metrics': {
                'api_calls': self.ai_api_calls,
                'cache_hits': self.ai_cache_hits,
                'cache_hit_rate': (
                    self.ai_cache_hits / max(self.ai_api_calls, 1)
                ),
                'avg_confidence': self.average_confidence
            },
            'strategy_distribution': self.strategy_distribution
        }

# ============================================================================
# END OF PART 1
# ============================================================================

# ============================================================================
# PART 2: SINGLETON UTILITIES & LOGGING INFRASTRUCTURE
# ============================================================================
# These are singleton classes for tracking, logging, and monitoring throughout
# the optimization process. They provide centralized tracking without coupling.

# ============================================================================
# GLOBAL SINGLETONS - Proper Singleton Implementation
# ============================================================================

_global_logger = None
_performance_monitor = None
_ai_tracker = None
_singleton_lock = threading.Lock()

def get_logger():
    """Thread-safe singleton logger"""
    global _global_logger
    if _global_logger is None:
        with _singleton_lock:
            if _global_logger is None:  # Double-check
                _global_logger = GlobalLogger()
    return _global_logger

def get_performance_monitor():
    """Thread-safe singleton performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        with _singleton_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_ai_tracker():
    """Thread-safe singleton AI decision tracker"""
    global _ai_tracker
    if _ai_tracker is None:
        with _singleton_lock:
            if _ai_tracker is None:
                _ai_tracker = AIDecisionTracker()
    return _ai_tracker

# ============================================================================
# GLOBAL LOGGER WITH ENHANCED ERROR TRACKING
# ============================================================================

class GlobalLogger:
    """
    Enhanced global logger with memory management and intelligent error tracking

    DFS Value: Understanding failures helps avoid them in future optimizations
    Key improvements:
    - Error pattern detection to identify recurring issues
    - Contextual error suggestions for faster debugging
    - Automatic memory cleanup
    - Better categorization of log types
    """

    # OPTIMIZATION: Pre-compiled regex patterns (Recommendation #10)
    _PATTERN_NUMBER = re.compile(r'\d+')
    _PATTERN_DOUBLE_QUOTE = re.compile(r'"[^"]*"')
    _PATTERN_SINGLE_QUOTE = re.compile(r"'[^']*'")

    def __init__(self):
        self.logs = deque(maxlen=OptimizerConfig.MAX_HISTORY_ENTRIES)
        self.error_logs = deque(maxlen=20)
        self.ai_decisions = deque(maxlen=50)
        self.optimization_events = deque(maxlen=30)
        self.performance_metrics = defaultdict(list)
        self._lock = threading.RLock()

        # Enhanced error tracking
        self.error_patterns = defaultdict(int)
        self.error_suggestions_cache = {}
        self.last_cleanup = datetime.now()

        # Track common failure modes
        self.failure_categories = {
            'constraint': 0,
            'salary': 0,
            'ownership': 0,
            'api': 0,
            'validation': 0,
            'timeout': 0,
            'other': 0
        }

    def log(self, message: str, level: str = "INFO", context: Dict = None):
        """
        Enhanced logging with context and pattern detection

        Args:
            message: Log message
            level: INFO, WARNING, ERROR, CRITICAL, DEBUG
            context: Additional context dictionary
        """
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'context': context or {}
            }

            self.logs.append(entry)

            if level in ["ERROR", "CRITICAL"]:
                self.error_logs.append(entry)

                # Track error patterns
                error_key = self._extract_error_pattern(message)
                self.error_patterns[error_key] += 1

                # Categorize failure
                self._categorize_failure(message)

            # Periodic cleanup (every 5 minutes)
            if (datetime.now() - self.last_cleanup).seconds > 300:
                self._cleanup()

    def log_exception(self, exception: Exception, context: str = "",
                     critical: bool = False):
        """
        Enhanced exception logging with helpful suggestions

        DFS Value: Faster debugging = more time optimizing lineups
        """
        with self._lock:
            error_msg = f"{context}: {str(exception)}" if context else str(exception)

            # Generate helpful suggestions based on error type
            suggestions = self._get_error_suggestions(exception, context)

            entry = {
                'timestamp': datetime.now(),
                'level': "CRITICAL" if critical else "ERROR",
                'message': error_msg,
                'exception_type': type(exception).__name__,
                'traceback': traceback.format_exc(),
                'suggestions': suggestions,
                'context': context
            }

            self.error_logs.append(entry)

            # Also log to main log with suggestion
            if suggestions:
                self.log(
                    f"Error: {error_msg}\nSuggestion: {suggestions[0]}",
                    "ERROR",
                    {'has_suggestion': True}
                )

    def _extract_error_pattern(self, message: str) -> str:
        """
        Extract error pattern for tracking common issues
        OPTIMIZED: Uses pre-compiled regex patterns (Recommendation #10)

        Replaces specific values with placeholders to identify patterns
        """
        pattern = self._PATTERN_NUMBER.sub('N', message)  # Replace numbers
        pattern = self._PATTERN_DOUBLE_QUOTE.sub('"X"', pattern)  # Replace quoted strings
        pattern = self._PATTERN_SINGLE_QUOTE.sub("'X'", pattern)  # Replace single-quoted
        return pattern[:100]  # Limit length

    def _categorize_failure(self, message: str):
        """Categorize failure type for analytics"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['constraint', 'infeasible', 'no solution']):
            self.failure_categories['constraint'] += 1
        elif any(word in message_lower for word in ['salary', 'cap', 'budget']):
            self.failure_categories['salary'] += 1
        elif 'ownership' in message_lower:
            self.failure_categories['ownership'] += 1
        elif any(word in message_lower for word in ['api', 'connection', 'timeout']):
            self.failure_categories['api'] += 1
        elif 'validation' in message_lower:
            self.failure_categories['validation'] += 1
        elif 'timeout' in message_lower:
            self.failure_categories['timeout'] += 1
        else:
            self.failure_categories['other'] += 1

    def _get_error_suggestions(self, exception: Exception, context: str) -> List[str]:
        """
        Provide helpful suggestions based on error type
        FIXED: LRU-style cache eviction to prevent memory leak (Recommendation #8)

        DFS Value: Contextual help reduces time spent debugging
        """
        suggestions = []
        error_str = str(exception).lower()
        exception_type = type(exception).__name__

        # Check cache first
        cache_key = f"{exception_type}_{context}"
        if cache_key in self.error_suggestions_cache:
            return self.error_suggestions_cache[cache_key]

        # KeyError suggestions
        if isinstance(exception, KeyError):
            suggestions.append("Check that all required columns are present in CSV")
            suggestions.append("Verify player names match exactly between data and AI recommendations")
            suggestions.append("Ensure DataFrame has been properly validated")

        # ValueError suggestions
        elif isinstance(exception, ValueError):
            if "salary" in error_str:
                suggestions.append("Check salary cap constraints - may be too restrictive")
                suggestions.append("Verify required players fit within salary cap")
            elif "ownership" in error_str:
                suggestions.append("Verify ownership projections are between 0-100")
                suggestions.append("Check for missing ownership data")
            else:
                suggestions.append("Check data types and value ranges")

        # PuLP solver errors
        elif "pulp" in exception_type.lower() or "solver" in error_str:
            suggestions.append("Optimization constraints may be infeasible")
            suggestions.append("Try relaxing AI enforcement level")
            suggestions.append("Check that required players can fit in salary cap")
            suggestions.append("Verify team diversity requirements can be met")

        # Timeout errors
        elif "timeout" in error_str:
            suggestions.append("Reduce number of lineups or increase timeout")
            suggestions.append("Try fewer hard constraints")
            suggestions.append("Consider using fewer parallel threads")

        # API errors
        elif "api" in error_str or "connection" in error_str:
            suggestions.append("Check API key is valid")
            suggestions.append("Verify internet connection")
            suggestions.append("API may be rate-limited - wait and retry")

        # DataFrame/Pandas errors
        elif isinstance(exception, (pd.errors.EmptyDataError, AttributeError)):
            suggestions.append("Ensure CSV file is not empty")
            suggestions.append("Check column names match expected format")
            suggestions.append("Verify data has been loaded correctly")

        # Generic suggestion if no specific match
        if not suggestions:
            suggestions.append("Check logs for more details")
            suggestions.append("Verify all input data is valid")

        # FIXED: LRU-style eviction instead of clearing all (Recommendation #8)
        if len(self.error_suggestions_cache) > 100:
            # Remove oldest 50% instead of clearing everything
            old_keys = list(self.error_suggestions_cache.keys())[:50]
            for key in old_keys:
                del self.error_suggestions_cache[key]

        self.error_suggestions_cache[cache_key] = suggestions

        return suggestions

    def _cleanup(self):
        """Memory cleanup to prevent unbounded growth"""
        cutoff = datetime.now() - timedelta(hours=1)

        # Clean old performance metrics
        for key in list(self.performance_metrics.keys()):
            self.performance_metrics[key] = [
                m for m in self.performance_metrics[key]
                if m.get('timestamp', datetime.now()) > cutoff
            ]

            # Remove empty keys
            if not self.performance_metrics[key]:
                del self.performance_metrics[key]

        # Clean old error patterns (keep only recent patterns)
        if len(self.error_patterns) > 50:
            # Keep top 30 most common patterns
            sorted_patterns = sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.error_patterns = dict(sorted_patterns[:30])

        self.last_cleanup = datetime.now()

    def log_ai_decision(self, decision_type: str, ai_source: str,
                       success: bool, details: Dict = None, confidence: float = 0):
        """Log AI decision for tracking"""
        with self._lock:
            self.ai_decisions.append({
                'timestamp': datetime.now(),
                'type': decision_type,
                'source': ai_source,
                'success': success,
                'confidence': confidence,
                'details': details or {}
            })

    def log_optimization_start(self, num_lineups: int, field_size: str, settings: Dict):
        """Log optimization start with settings"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'start',
                'num_lineups': num_lineups,
                'field_size': field_size,
                'settings': settings
            })

    def log_optimization_end(self, lineups_generated: int, time_taken: float,
                            success_rate: float):
        """Log optimization completion"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'complete',
                'lineups_generated': lineups_generated,
                'time_taken': time_taken,
                'success_rate': success_rate
            })

    def get_error_summary(self) -> Dict:
        """Get summary of errors for diagnostics"""
        with self._lock:
            return {
                'total_errors': len(self.error_logs),
                'error_categories': dict(self.failure_categories),
                'top_patterns': sorted(
                    self.error_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'recent_errors': list(self.error_logs)[-5:]
            }

# ============================================================================
# PERFORMANCE MONITOR WITH ENHANCED TRACKING
# ============================================================================

class PerformanceMonitor:
    """
    Enhanced performance monitoring with detailed metrics

    DFS Value: Identify bottlenecks to optimize generation speed
    """

    def __init__(self):
        self.timers = {}
        self.metrics = defaultdict(list)
        self._lock = threading.RLock()
        self.start_times = {}

        # Enhanced tracking
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.memory_snapshots = deque(maxlen=10)

        # Track specific optimization phases
        self.phase_times = {
            'data_load': [],
            'ai_analysis': [],
            'lineup_generation': [],
            'validation': [],
            'export': []
        }

    def start_timer(self, operation: str):
        """Start timing an operation"""
        with self._lock:
            self.start_times[operation] = time.time()
            self.operation_counts[operation] += 1

    def stop_timer(self, operation: str) -> float:
        """
        Stop timing and return elapsed time

        Returns:
            Elapsed time in seconds, or 0 if timer wasn't started
        """
        with self._lock:
            if operation not in self.start_times:
                return 0

            elapsed = time.time() - self.start_times[operation]
            del self.start_times[operation]

            # Store timing data
            self.operation_times[operation].append(elapsed)

            # Keep only recent timings (memory management)
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-50:]

            return elapsed

    def record_metric(self, metric_name: str, value: float, tags: Dict = None):
        """Record a metric with optional tags"""
        with self._lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': datetime.now(),
                'tags': tags or {}
            })

            # Cleanup old metrics (keep last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if m['timestamp'] > cutoff
            ]

    def record_phase_time(self, phase: str, duration: float):
        """Record time for specific optimization phase"""
        with self._lock:
            if phase in self.phase_times:
                self.phase_times[phase].append(duration)

                # Keep recent history
                if len(self.phase_times[phase]) > 20:
                    self.phase_times[phase] = self.phase_times[phase][-10:]

    def get_operation_stats(self, operation: str) -> Dict:
        """Get statistics for an operation"""
        with self._lock:
            times = self.operation_times.get(operation, [])
            if not times:
                return {}

            return {
                'count': self.operation_counts[operation],
                'avg_time': np.mean(times),
                'median_time': np.median(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
                'std_dev': np.std(times) if len(times) > 1 else 0
            }

    def get_phase_summary(self) -> Dict:
        """Get summary of optimization phases"""
        with self._lock:
            summary = {}
            for phase, times in self.phase_times.items():
                if times:
                    summary[phase] = {
                        'avg_time': np.mean(times),
                        'total_time': sum(times),
                        'count': len(times)
                    }
            return summary

    def get_bottlenecks(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify performance bottlenecks

        Returns:
            List of (operation, avg_time) sorted by time
        """
        with self._lock:
            bottlenecks = []
            for operation, times in self.operation_times.items():
                if times:
                    avg_time = np.mean(times)
                    bottlenecks.append((operation, avg_time))

            return sorted(bottlenecks, key=lambda x: x[1], reverse=True)[:top_n]

# ============================================================================
# AI DECISION TRACKER WITH LEARNING
# ============================================================================

class AIDecisionTracker:
    """
    Track AI decisions and learn from performance

    DFS Value: CRITICAL - Learning what works improves future recommendations
    """

    def __init__(self):
        self.decisions = deque(maxlen=OptimizerConfig.MAX_HISTORY_ENTRIES)
        self.performance_feedback = deque(maxlen=100)
        self.decision_patterns = defaultdict(list)
        self._lock = threading.RLock()

        # Learning components
        self.successful_patterns = defaultdict(float)
        self.failed_patterns = defaultdict(float)
        self.confidence_calibration = defaultdict(list)

        # Track which AI strategies work best
        self.strategy_performance = {
            'game_theory': {'wins': 0, 'attempts': 0, 'avg_score': 0},
            'correlation': {'wins': 0, 'attempts': 0, 'avg_score': 0},
            'contrarian': {'wins': 0, 'attempts': 0, 'avg_score': 0}
        }

    def track_decision(self, ai_type: AIStrategistType, decision: AIRecommendation,
                      context: Dict = None):
        """Track an AI decision"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'ai_type': ai_type,
                'captain_count': len(decision.captain_targets),
                'confidence': decision.confidence,
                'enforcement_rules': len(decision.enforcement_rules),
                'context': context or {}
            }

            self.decisions.append(entry)

            # Track patterns
            pattern_key = self._extract_pattern(decision)
            self.decision_patterns[pattern_key].append(entry)

    def _extract_pattern(self, decision: AIRecommendation) -> str:
        """Extract pattern from decision for learning"""
        pattern_elements = [
            f"conf_{int(decision.confidence*10)}",
            f"capt_{min(len(decision.captain_targets), 5)}",
            f"must_{min(len(decision.must_play), 3)}",
            f"stack_{min(len(decision.stacks), 3)}"
        ]
        return "_".join(pattern_elements)

    def record_performance(self, lineup: Dict, actual_score: Optional[float] = None):
        """
        Record lineup performance for learning

        DFS Value: Track what wins to do more of it
        """
        with self._lock:
            if actual_score is not None:
                projected = lineup.get('Projected', 0)
                accuracy = 1 - abs(actual_score - projected) / max(actual_score, 1)

                entry = {
                    'timestamp': datetime.now(),
                    'strategy': lineup.get('AI_Strategy', 'unknown'),
                    'projected': projected,
                    'actual': actual_score,
                    'accuracy': accuracy,
                    'success': actual_score > projected * 1.1,  # 10% better than projected
                    'ownership': lineup.get('Total_Ownership', 0),
                    'captain': lineup.get('Captain', '')
                }

                self.performance_feedback.append(entry)

                # Update pattern success rates
                strategy = lineup.get('AI_Strategy', 'unknown')
                ownership_tier = lineup.get('Ownership_Tier', 'unknown')
                pattern_key = f"{strategy}_{ownership_tier}"

                if entry['success']:
                    self.successful_patterns[pattern_key] += 1
                else:
                    self.failed_patterns[pattern_key] += 1

                # Update confidence calibration
                confidence = lineup.get('Confidence', 0.5)
                conf_bucket = int(confidence * 10)
                self.confidence_calibration[conf_bucket].append(accuracy)

                # Update strategy performance
                if strategy in self.strategy_performance:
                    self.strategy_performance[strategy]['attempts'] += 1
                    if entry['success']:
                        self.strategy_performance[strategy]['wins'] += 1

                    # Update running average score
                    current_avg = self.strategy_performance[strategy]['avg_score']
                    attempts = self.strategy_performance[strategy]['attempts']
                    self.strategy_performance[strategy]['avg_score'] = (
                        (current_avg * (attempts - 1) + actual_score) / attempts
                    )

    def get_learning_insights(self) -> Dict:
        """
        Get insights from tracked performance

        DFS Value: Actionable insights for future optimizations
        """
        with self._lock:
            insights = {
                'total_decisions': len(self.decisions),
                'avg_confidence': (
                    np.mean([d['confidence'] for d in self.decisions])
                    if self.decisions else 0
                )
            }

            # Calculate pattern success rates
            pattern_stats = {}
            for pattern in set(list(self.successful_patterns.keys()) +
                             list(self.failed_patterns.keys())):
                successes = self.successful_patterns.get(pattern, 0)
                failures = self.failed_patterns.get(pattern, 0)
                total = successes + failures

                if total >= 5:  # Only report patterns with enough data
                    pattern_stats[pattern] = {
                        'success_rate': successes / total,
                        'total': total,
                        'confidence': 'high' if total >= 10 else 'medium'
                    }

            insights['pattern_performance'] = pattern_stats

            # Confidence calibration
            calibration = {}
            for conf_level, accuracies in self.confidence_calibration.items():
                if accuracies:
                    calibration[conf_level / 10] = {
                        'actual_accuracy': np.mean(accuracies),
                        'sample_size': len(accuracies)
                    }

            insights['confidence_calibration'] = calibration

            # Strategy performance
            insights['strategy_performance'] = {
                strategy: {
                    'win_rate': (
                        stats['wins'] / stats['attempts']
                        if stats['attempts'] > 0 else 0
                    ),
                    'avg_score': stats['avg_score'],
                    'attempts': stats['attempts']
                }
                for strategy, stats in self.strategy_performance.items()
            }

            return insights

    def get_recommended_adjustments(self) -> Dict:
        """
        Get recommended adjustments based on learning

        DFS Value: Apply learned patterns to improve future builds
        """
        insights = self.get_learning_insights()
        adjustments = {}

        # Recommend confidence adjustments
        calibration = insights.get('confidence_calibration', {})
        for conf_level, stats in calibration.items():
            actual_accuracy = stats['actual_accuracy']
            sample_size = stats['sample_size']

            # Only adjust if we have enough data and significant difference
            if sample_size >= 5 and abs(conf_level - actual_accuracy) > 0.15:
                adjustments[f'confidence_{conf_level:.1f}'] = {
                    'current': conf_level,
                    'suggested': actual_accuracy,
                    'reason': f'Historical accuracy is {actual_accuracy:.1%} vs stated {conf_level:.1%}'
                }

        # Recommend pattern adjustments
        pattern_perf = insights.get('pattern_performance', {})
        for pattern, stats in pattern_perf.items():
            if stats['total'] >= 10:  # Enough data for confidence
                if stats['success_rate'] > 0.7:
                    adjustments[f'boost_{pattern}'] = {
                        'multiplier': 1.2,
                        'reason': f'Pattern shows {stats["success_rate"]:.1%} success rate'
                    }
                elif stats['success_rate'] < 0.3:
                    adjustments[f'reduce_{pattern}'] = {
                        'multiplier': 0.8,
                        'reason': f'Pattern shows only {stats["success_rate"]:.1%} success rate'
                    }

        # Strategy recommendations
        strategy_perf = insights.get('strategy_performance', {})
        best_strategy = max(
            strategy_perf.items(),
            key=lambda x: x[1]['win_rate'] if x[1]['attempts'] >= 5 else 0,
            default=(None, None)
        )

        if best_strategy[0] and best_strategy[1]['attempts'] >= 5:
            adjustments['preferred_strategy'] = {
                'strategy': best_strategy[0],
                'win_rate': best_strategy[1]['win_rate'],
                'reason': f'Best performing strategy with {best_strategy[1]["win_rate"]:.1%} win rate'
            }

        return adjustments

    def apply_learned_adjustments(self, df: pd.DataFrame,
                                  current_strategy: str) -> pd.DataFrame:
        """
        Apply learned adjustments to player projections

        DFS Value: Use historical performance to improve projections
        """
        adjustments = self.get_recommended_adjustments()
        df_adjusted = df.copy()

        # Apply pattern-based adjustments
        for key, adjustment in adjustments.items():
            if key.startswith('boost_') and current_strategy in key:
                # Boost projections slightly for successful patterns
                multiplier = adjustment['multiplier']
                df_adjusted['Projected_Points'] *= multiplier

            elif key.startswith('reduce_') and current_strategy in key:
                # Reduce projections for unsuccessful patterns
                multiplier = adjustment['multiplier']
                df_adjusted['Projected_Points'] *= multiplier

        return df_adjusted

# ============================================================================
# END OF PART 2
# ============================================================================

# ============================================================================
# PART 3: ENFORCEMENT, VALIDATION, AND SYNTHESIS COMPONENTS
# ============================================================================
# This part contains the rule engine that enforces AI recommendations,
# manages ownership bucketing, validates configurations, and synthesizes
# multiple AI perspectives into unified strategy.

# ============================================================================
# AI ENFORCEMENT ENGINE WITH THREE-TIER RELAXATION
# ============================================================================

class AIEnforcementEngine:
    """
    Enhanced enforcement engine with three-tier constraint relaxation

    DFS Value: CRITICAL - Three-tier system allows maintaining diversity
    while respecting AI recommendations. Strict -> Relaxed -> Minimal
    enables generating unique lineups without violating core DK rules.
    """

    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.STRONG):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Enhanced rule tracking
        self.applied_rules = deque(maxlen=100)
        self.rule_success_rate = defaultdict(float)
        self.violation_patterns = defaultdict(int)

        # Track which rules are most effective
        self.rule_effectiveness = defaultdict(lambda: {'applied': 0, 'success': 0})

    def create_enforcement_rules(self,
                                 recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """
        Create comprehensive enforcement rules from AI recommendations

        Returns dictionary with categorized rules by priority and type
        """
        self.logger.log(
            f"Creating enforcement rules at {self.enforcement_level.value} level",
            "INFO"
        )

        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        # Process recommendations based on enforcement level
        if self.enforcement_level == AIEnforcementLevel.MANDATORY:
            rules = self._create_mandatory_rules(recommendations)
        elif self.enforcement_level == AIEnforcementLevel.STRONG:
            rules = self._create_strong_rules(recommendations)
        elif self.enforcement_level == AIEnforcementLevel.MODERATE:
            rules = self._create_moderate_rules(recommendations)
        else:  # ADVISORY
            rules = self._create_advisory_rules(recommendations)

        # Add advanced stacking rules
        rules['stacking_rules'].extend(self._create_stacking_rules(recommendations))

        # Sort by priority within each category
        for rule_type in rules:
            if isinstance(rules[rule_type], list):
                rules[rule_type].sort(
                    key=lambda x: x.get('priority', 0),
                    reverse=True
                )

        total_rules = sum(len(v) for v in rules.values() if isinstance(v, list))
        self.logger.log(f"Created {total_rules} enforcement rules", "INFO")

        return rules

    def _create_mandatory_rules(self, recommendations: Dict) -> Dict:
        """
        Create mandatory enforcement rules (all AI decisions enforced as hard constraints)

        DFS Value: Maximum AI control for large-field GPPs
        """
        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(
                ai_type.value.lower().replace(' ', '_'),
                0.33
            )

            # Captain constraints (highest priority)
            if rec.captain_targets:
                rules['hard_constraints'].append({
                    'rule': 'captain_from_list',
                    'players': rec.captain_targets[:7],
                    'source': ai_type.value,
                    'priority': int(ConstraintPriority.AI_HIGH_CONFIDENCE.value *
                                   weight * rec.confidence),
                    'type': 'hard',
                    'relaxation_tier': 1  # Never relax
                })

            # Must play constraints
            for i, player in enumerate(rec.must_play[:3]):
                rules['hard_constraints'].append({
                    'rule': 'must_include',
                    'player': player,
                    'source': ai_type.value,
                    'priority': int((ConstraintPriority.AI_HIGH_CONFIDENCE.value - i * 5) *
                                   weight * rec.confidence),
                    'type': 'hard',
                    'relaxation_tier': 2  # Relax only on attempt 3
                })

            # Never play constraints
            for i, player in enumerate(rec.never_play[:3]):
                rules['hard_constraints'].append({
                    'rule': 'must_exclude',
                    'player': player,
                    'source': ai_type.value,
                    'priority': int((ConstraintPriority.AI_MODERATE.value - i * 5) *
                                   weight * rec.confidence),
                    'type': 'hard',
                    'relaxation_tier': 2
                })

            # Stack constraints
            for i, stack in enumerate(rec.stacks[:3]):
                rules['stacking_rules'].append({
                    'rule': 'must_stack',
                    'stack': stack,
                    'source': ai_type.value,
                    'priority': int((ConstraintPriority.AI_MODERATE.value - i * 5) *
                                   weight * rec.confidence),
                    'type': 'hard',
                    'relaxation_tier': 3  # Relax on attempt 2+
                })

        return rules

    def _create_strong_rules(self, recommendations: Dict) -> Dict:
        """
        Create strong enforcement rules (most AI decisions enforced)

        DFS Value: Balanced approach - strong AI guidance with some flexibility
        """
        # Start with moderate rules as base
        rules = self._create_moderate_rules(recommendations)

        # Upgrade high-confidence recommendations to hard constraints
        for ai_type, rec in recommendations.items():
            if rec.confidence > 0.75:  # High confidence threshold
                weight = OptimizerConfig.AI_WEIGHTS.get(
                    ai_type.value.lower().replace(' ', '_'),
                    0.33
                )

                if rec.captain_targets:
                    rules['hard_constraints'].append({
                        'rule': 'captain_selection',
                        'players': rec.captain_targets[:5],
                        'source': ai_type.value,
                        'priority': int(ConstraintPriority.AI_HIGH_CONFIDENCE.value *
                                       weight * rec.confidence),
                        'type': 'hard',
                        'relaxation_tier': 2
                    })

        return rules

    def _create_moderate_rules(self, recommendations: Dict) -> Dict:
        """
        Create moderate enforcement rules (balanced approach)

        DFS Value: Enforces consensus, suggests individual recommendations
        """
        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        # Find consensus recommendations (multiple AIs agree)
        captain_counts = defaultdict(int)
        must_play_counts = defaultdict(int)

        for rec in recommendations.values():
            for captain in rec.captain_targets:
                captain_counts[captain] += 1
            for player in rec.must_play:
                must_play_counts[player] += 1

        # Enforce consensus as hard constraints
        for captain, count in captain_counts.items():
            if count >= 2:  # At least 2 AIs agree
                rules['hard_constraints'].append({
                    'rule': 'consensus_captain',
                    'player': captain,
                    'agreement': count,
                    'priority': ConstraintPriority.AI_CONSENSUS.value + (count * 5),
                    'type': 'hard',
                    'relaxation_tier': 2
                })

        # Add soft constraints for single AI recommendations
        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(
                ai_type.value.lower().replace(' ', '_'),
                0.33
            )

            for player in rec.must_play[:3]:
                if must_play_counts[player] == 1:  # Only one AI suggested
                    rules['soft_constraints'].append({
                        'rule': 'prefer_player',
                        'player': player,
                        'source': ai_type.value,
                        'weight': weight * rec.confidence,
                        'priority': int(ConstraintPriority.SOFT_PREFERENCE.value *
                                       weight * rec.confidence),
                        'type': 'soft'
                    })

        return rules

    def _create_advisory_rules(self, recommendations: Dict) -> Dict:
        """
        Create advisory rules (suggestions only)

        DFS Value: Maximum flexibility for small-field contests
        """
        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        # All recommendations become soft constraints
        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(
                ai_type.value.lower().replace(' ', '_'),
                0.33
            )

            # Captain preferences
            for i, captain in enumerate(rec.captain_targets[:5]):
                rules['soft_constraints'].append({
                    'rule': 'prefer_captain',
                    'player': captain,
                    'source': ai_type.value,
                    'weight': weight * rec.confidence * (1 - i * 0.1),
                    'priority': int(ConstraintPriority.SOFT_PREFERENCE.value *
                                   weight * rec.confidence),
                    'type': 'soft'
                })

        return rules

    def _create_stacking_rules(self, recommendations: Dict) -> List[Dict]:
        """
        Create advanced stacking rules for all stack types

        DFS Value: Each stack type captures different game scripts
        - Onslaught: Blowout scenarios
        - Bring_back: High-scoring games
        - Leverage: Low-owned correlations
        - Defensive: Negative game scripts
        """
        stacking_rules = []

        for ai_type, rec in recommendations.items():
            for stack in rec.stacks:
                stack_type = stack.get('type', 'standard')

                if stack_type == 'onslaught':
                    # 3-4+ players from same team
                    stacking_rules.append({
                        'rule': 'onslaught_stack',
                        'players': stack.get('players', []),
                        'team': stack.get('team'),
                        'min_players': 3,
                        'max_players': 5,
                        'scenario': stack.get('scenario', 'blowout'),
                        'priority': ConstraintPriority.AI_MODERATE.value,
                        'source': ai_type.value,
                        'correlation_strength': 0.6,
                        'relaxation_tier': 3
                    })

                elif stack_type == 'bring_back':
                    # Primary stack with opposing player
                    stacking_rules.append({
                        'rule': 'bring_back_stack',
                        'primary_players': stack.get('primary_stack', []),
                        'bring_back_player': stack.get('bring_back'),
                        'game_total': stack.get('game_total', 45),
                        'priority': ConstraintPriority.AI_MODERATE.value,
                        'source': ai_type.value,
                        'correlation_strength': 0.5,
                        'relaxation_tier': 3
                    })

                elif stack_type == 'leverage':
                    # Low ownership correlated plays
                    stacking_rules.append({
                        'rule': 'leverage_stack',
                        'players': [stack.get('player1'), stack.get('player2')],
                        'combined_ownership_max': stack.get('combined_ownership', 20),
                        'leverage_score_min': 3.0,
                        'priority': ConstraintPriority.SOFT_PREFERENCE.value,
                        'source': ai_type.value,
                        'correlation_strength': 0.4,
                        'relaxation_tier': 3
                    })

                elif stack_type == 'defensive':
                    # DST with negative game script
                    stacking_rules.append({
                        'rule': 'defensive_stack',
                        'dst_team': stack.get('dst_team'),
                        'opposing_players_max': 1,
                        'scenario': 'defensive_game',
                        'priority': ConstraintPriority.SOFT_PREFERENCE.value,
                        'source': ai_type.value,
                        'relaxation_tier': 3
                    })

                else:
                    # Standard two-player stack
                    if 'player1' in stack and 'player2' in stack:
                        stacking_rules.append({
                            'rule': 'standard_stack',
                            'players': [stack['player1'], stack['player2']],
                            'correlation': stack.get('correlation', 0.5),
                            'priority': ConstraintPriority.SOFT_PREFERENCE.value,
                            'source': ai_type.value,
                            'relaxation_tier': 3
                        })

        # Remove duplicate stacks
        unique_stacks = []
        seen = set()

        for stack in stacking_rules:
            players = stack.get('players', [])
            if players:
                stack_id = "_".join(sorted(players[:2]))
                if stack_id not in seen:
                    seen.add(stack_id)
                    unique_stacks.append(stack)
            else:
                unique_stacks.append(stack)

        return unique_stacks

    def should_apply_constraint(self, constraint: Dict, attempt_num: int) -> bool:
        """
        Determine if constraint should be applied based on relaxation tier

        DFS Value: CRITICAL - Enables three-tier constraint relaxation

        Args:
            constraint: Constraint dictionary with relaxation_tier
            attempt_num: Current attempt (0-2 for three tiers)

        Returns:
            True if constraint should be applied
        """
        tier = constraint.get('relaxation_tier', 1)

        # Tier 1: Never relax (DK rules, salary cap)
        if tier == 1:
            return True

        # Tier 2: Relax only on attempt 3 (final)
        if tier == 2:
            return attempt_num < 2

        # Tier 3: Relax on attempt 2+ (soft preferences)
        if tier == 3:
            return attempt_num == 0

        return True

    def validate_lineup_against_ai(self, lineup: Dict,
                                   enforcement_rules: Dict) -> Tuple[bool, List[str]]:
        """
        Validate lineup against AI enforcement rules

        Returns:
            (is_valid, list of violations)
        """
        violations = []
        captain = lineup.get('Captain')
        flex = lineup.get('FLEX', [])
        all_players = [captain] + flex

        # Check hard constraints only
        for rule in enforcement_rules.get('hard_constraints', []):
            rule_type = rule.get('rule')

            if rule_type == 'captain_from_list':
                if captain not in rule.get('players', []):
                    violations.append(
                        f"Captain {captain} not in AI-recommended list: {rule['source']}"
                    )

            elif rule_type == 'must_include':
                if rule.get('player') not in all_players:
                    violations.append(
                        f"Missing required player: {rule['player']} ({rule['source']})"
                    )

            elif rule_type == 'must_exclude':
                if rule.get('player') in all_players:
                    violations.append(
                        f"Included banned player: {rule['player']} ({rule['source']})"
                    )

            elif rule_type == 'consensus_captain':
                if captain != rule.get('player'):
                    violations.append(
                        f"Not using consensus captain: {rule['player']} "
                        f"(agreement: {rule.get('agreement', 0)} AIs)"
                    )

        # Check stacking rules (if hard)
        for stack_rule in enforcement_rules.get('stacking_rules', []):
            if stack_rule.get('type') == 'hard':
                if not self._validate_stack_rule(all_players, stack_rule):
                    violations.append(
                        f"Stack rule violation: {stack_rule.get('rule')} "
                        f"({stack_rule.get('source')})"
                    )

        # Track violations for learning
        for violation in violations:
            self.violation_patterns[violation[:50]] += 1

        is_valid = len(violations) == 0

        # Record rule application
        self.applied_rules.append({
            'timestamp': datetime.now(),
            'lineup_num': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': len(violations)
        })

        # Update rule effectiveness
        for rule in enforcement_rules.get('hard_constraints', []):
            rule_key = f"{rule.get('rule')}_{rule.get('source')}"
            self.rule_effectiveness[rule_key]['applied'] += 1
            if is_valid:
                self.rule_effectiveness[rule_key]['success'] += 1

        return is_valid, violations

    def _validate_stack_rule(self, players: List[str], stack_rule: Dict) -> bool:
        """Validate a specific stack rule"""
        rule_type = stack_rule.get('rule')

        if rule_type == 'onslaught_stack':
            required_players = stack_rule.get('players', [])
            min_required = stack_rule.get('min_players', 3)
            count = sum(1 for p in required_players if p in players)
            return count >= min_required

        elif rule_type == 'bring_back_stack':
            primary = stack_rule.get('primary_players', [])
            bring_back = stack_rule.get('bring_back_player')
            has_primary = any(p in players for p in primary)
            has_bring_back = bring_back in players
            return has_primary and has_bring_back

        elif rule_type == 'standard_stack':
            required = stack_rule.get('players', [])
            return all(p in players for p in required[:2])

        return True

    def get_effectiveness_report(self) -> Dict:
        """Get report on rule effectiveness"""
        report = {}
        for rule_key, stats in self.rule_effectiveness.items():
            if stats['applied'] > 0:
                report[rule_key] = {
                    'success_rate': stats['success'] / stats['applied'],
                    'applied': stats['applied']
                }
        return report

# ============================================================================
# AI OWNERSHIP BUCKET MANAGER WITH DYNAMIC THRESHOLDS
# ============================================================================

class AIOwnershipBucketManager:
    """
    Enhanced ownership bucket management with dynamic threshold adjustment

    DFS Value: Dynamic thresholds adapt to slate characteristics
    Flat ownership slate = tighter buckets, polarized = wider buckets
    """

    def __init__(self, enforcement_engine: AIEnforcementEngine = None):
        self.enforcement_engine = enforcement_engine
        self.logger = get_logger()

        # Base bucket thresholds (will be adjusted dynamically)
        self.bucket_thresholds = {
            'mega_chalk': 35,
            'chalk': 20,
            'moderate': 15,
            'pivot': 10,
            'leverage': 5,
            'super_leverage': 2
        }

        # Store original thresholds for reference
        self.base_thresholds = self.bucket_thresholds.copy()

    def adjust_thresholds_for_slate(self, df: pd.DataFrame, field_size: str):
        """
        Dynamically adjust bucket thresholds based on slate characteristics

        DFS Value: CRITICAL - Adapts to ownership distribution
        Flat ownership = lower thresholds for differentiation
        Polarized ownership = higher thresholds to capture extremes
        """
        ownership_std = df['Ownership'].std()
        ownership_mean = df['Ownership'].mean()

        self.logger.log(
            f"Adjusting thresholds - Ownership std: {ownership_std:.1f}, "
            f"mean: {ownership_mean:.1f}",
            "DEBUG"
        )

        # Reset to base thresholds
        self.bucket_thresholds = self.base_thresholds.copy()

        # Adjust based on ownership concentration
        if ownership_std < 5:  # Flat ownership - everyone similar
            # Lower thresholds to create differentiation
            for key in self.bucket_thresholds:
                self.bucket_thresholds[key] *= 0.85
            self.logger.log("Flat ownership detected - lowering thresholds", "INFO")

        elif ownership_std > 15:  # Polarized ownership
            # Raise thresholds to avoid over-focusing on extremes
            for key in self.bucket_thresholds:
                self.bucket_thresholds[key] *= 1.15
            self.logger.log("Polarized ownership detected - raising thresholds", "INFO")

        # Field-size adjustments
        if field_size in ['large_field_aggressive', 'milly_maker']:
            # Shift all thresholds down for more aggressive leverage
            for key in self.bucket_thresholds:
                self.bucket_thresholds[key] *= 0.85
            self.logger.log(
                f"Large field ({field_size}) - increasing leverage sensitivity",
                "INFO"
            )

        # Adjust based on mean ownership
        if ownership_mean < 8:  # Low overall ownership
            # Market inefficiency - be more aggressive
            self.bucket_thresholds['chalk'] *= 0.9
            self.bucket_thresholds['leverage'] *= 1.1

        elif ownership_mean > 15:  # High overall ownership
            # Chalky slate - need more differentiation
            self.bucket_thresholds['leverage'] *= 0.9

    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize players into ownership buckets
        OPTIMIZED: Vectorized operations (Recommendation #3)

        Returns dict with bucket names as keys, player lists as values
        """
        # Vectorized comparisons instead of row-by-row iteration
        ownership = df['Ownership'].fillna(10)
        players = df['Player'].values

        # Get thresholds
        thresholds = self.bucket_thresholds

        # Vectorized bucketing - much faster than iterating
        buckets = {
            'mega_chalk': players[ownership >= thresholds['mega_chalk']].tolist(),
            'chalk': players[(ownership >= thresholds['chalk']) &
                            (ownership < thresholds['mega_chalk'])].tolist(),
            'moderate': players[(ownership >= thresholds['moderate']) &
                               (ownership < thresholds['chalk'])].tolist(),
            'pivot': players[(ownership >= thresholds['pivot']) &
                            (ownership < thresholds['moderate'])].tolist(),
            'leverage': players[(ownership >= thresholds['leverage']) &
                               (ownership < thresholds['pivot'])].tolist(),
            'super_leverage': players[ownership < thresholds['leverage']].tolist()
        }

        # Log distribution
        self.logger.log(
            f"Ownership buckets: " + ", ".join(
                f"{k}={len(v)}" for k, v in buckets.items()
            ),
            "DEBUG"
        )

        return buckets

    def calculate_gpp_leverage(self, players: List[str], df: pd.DataFrame) -> float:
        """
        Calculate GPP leverage score for lineup

        DFS Value: Quantifies tournament differentiation
        Higher leverage = lower ownership + high ceiling
        """
        if not players:
            return 0

        total_projection = 0
        total_ownership = 0
        leverage_bonus = 0

        for i, player in enumerate(players):
            player_data = df[df['Player'] == player]
            if player_data.empty:
                continue

            row = player_data.iloc[0]
            projection = row.get('Projected_Points', 0)
            ownership = row.get('Ownership', 10)

            # Captain gets 1.5x weight (assuming first player is captain)
            if i == 0:
                projection *= 1.5
                ownership *= 1.5

            total_projection += projection
            total_ownership += ownership

            # Bonus for leverage plays
            if ownership < self.bucket_thresholds['leverage']:
                leverage_bonus += 15
            elif ownership < self.bucket_thresholds['pivot']:
                leverage_bonus += 8
            elif ownership < self.bucket_thresholds['moderate']:
                leverage_bonus += 3

        # Calculate leverage score
        # Formula: (projection / ownership) + leverage_bonus
        if total_ownership > 0:
            avg_projection = total_projection / len(players)
            avg_ownership = total_ownership / len(players)
            base_leverage = avg_projection / (avg_ownership + 1)
            return base_leverage + leverage_bonus

        return 0

    def get_bucket_recommendations(self, field_size: str, num_lineups: int) -> Dict:
        """Get ownership bucket recommendations for field size"""
        recommendations = {
            'mega_chalk_limit': 1,
            'chalk_limit': 2,
            'min_leverage': 2,
            'target_ownership': (60, 90)
        }

        if field_size in ['large_field', 'large_field_aggressive']:
            recommendations['mega_chalk_limit'] = 0
            recommendations['chalk_limit'] = 1
            recommendations['min_leverage'] = 3
            recommendations['target_ownership'] = (40, 70)

        elif field_size == 'milly_maker':
            recommendations['mega_chalk_limit'] = 0
            recommendations['chalk_limit'] = 0
            recommendations['min_leverage'] = 4
            recommendations['target_ownership'] = (30, 60)

        return recommendations

# ============================================================================
# AI CONFIG VALIDATOR
# ============================================================================

class AIConfigValidator:
    """Enhanced validator with dynamic strategy selection"""

    @staticmethod
    def validate_ai_requirements(enforcement_rules: Dict,
                                df: pd.DataFrame) -> Dict:
        """
        Validate that AI requirements are feasible

        Returns validation result with errors, warnings, suggestions
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        available_players = set(df['Player'].values)

        # Check captain requirements
        captain_rules = [
            r for r in enforcement_rules.get('hard_constraints', [])
            if r.get('rule') in ['captain_from_list', 'captain_selection']
        ]

        for rule in captain_rules:
            valid_captains = [
                p for p in rule.get('players', [])
                if p in available_players
            ]

            if not valid_captains:
                validation_result['errors'].append(
                    "No valid captains from AI recommendations"
                )
                validation_result['is_valid'] = False
                validation_result['suggestions'].append(
                    "Relax captain constraints or check player pool"
                )
            elif len(valid_captains) < 3:
                validation_result['warnings'].append(
                    f"Only {len(valid_captains)} valid captains available"
                )
                validation_result['suggestions'].append(
                    "Consider expanding captain pool for diversity"
                )

        # Check must include players
        must_include = [
            r for r in enforcement_rules.get('hard_constraints', [])
            if r.get('rule') == 'must_include'
        ]

        for rule in must_include:
            if rule.get('player') not in available_players:
                validation_result['errors'].append(
                    f"Required player not available: {rule.get('player')}"
                )
                validation_result['is_valid'] = False

        # Check stack feasibility
        stacking_rules = enforcement_rules.get('stacking_rules', [])

        for stack in stacking_rules:
            if stack.get('rule') == 'onslaught_stack':
                players = stack.get('players', [])
                valid = [p for p in players if p in available_players]

                if len(valid) < stack.get('min_players', 3):
                    validation_result['warnings'].append(
                        f"Onslaught stack may not be feasible "
                        f"(only {len(valid)} valid players)"
                    )

        # Check salary feasibility
        hard_constraints = enforcement_rules.get('hard_constraints', [])
        required_players = [
            r.get('player') for r in hard_constraints
            if r.get('rule') == 'must_include' and r.get('player')
        ]

        if required_players:
            min_required_salary = df[
                df['Player'].isin(required_players)
            ]['Salary'].sum()

            if min_required_salary > OptimizerConfig.SALARY_CAP * 0.6:
                validation_result['warnings'].append(
                    f"Required players use {min_required_salary/OptimizerConfig.SALARY_CAP:.0%} "
                    "of salary cap"
                )
                validation_result['suggestions'].append(
                    "May have limited flexibility for other positions"
                )

        return validation_result

    @staticmethod
    def get_ai_strategy_distribution(field_size: str, num_lineups: int,
                                    consensus_level: str = 'mixed') -> Dict:
        """
        Dynamic strategy distribution based on consensus level

        DFS Value: Adapts lineup mix based on AI agreement
        High consensus = more balanced, Low consensus = more variety
        """
        # Base distribution by field size
        base_distributions = {
            'small_field': {
                'balanced': 0.5,
                'correlation_heavy': 0.3,
                'ownership_leverage': 0.2
            },
            'medium_field': {
                'balanced': 0.4,
                'correlation_heavy': 0.3,
                'ownership_leverage': 0.2,
                'contrarian': 0.1
            },
            'large_field': {
                'balanced': 0.3,
                'correlation_heavy': 0.2,
                'ownership_leverage': 0.3,
                'contrarian': 0.2
            },
            'large_field_aggressive': {
                'balanced': 0.2,
                'correlation_heavy': 0.2,
                'ownership_leverage': 0.3,
                'contrarian': 0.3
            },
            'milly_maker': {
                'balanced': 0.1,
                'correlation_heavy': 0.1,
                'ownership_leverage': 0.4,
                'contrarian': 0.4
            }
        }

        distribution = base_distributions.get(
            field_size,
            base_distributions['large_field']
        ).copy()

        # Adjust based on consensus level
        if consensus_level == 'high':
            # More balanced when AIs agree
            distribution['balanced'] = min(
                distribution.get('balanced', 0.3) * 1.3,
                0.5
            )
            distribution['contrarian'] = (
                distribution.get('contrarian', 0.2) * 0.7
            )
        elif consensus_level == 'low':
            # More variety when AIs disagree
            distribution['contrarian'] = min(
                distribution.get('contrarian', 0.2) * 1.3,
                0.4
            )
            distribution['balanced'] = (
                distribution.get('balanced', 0.3) * 0.7
            )

        # Normalize
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}

        # Convert to lineup counts
        lineup_distribution = {}
        allocated = 0

        for strategy, pct in distribution.items():
            count = int(num_lineups * pct)
            lineup_distribution[strategy] = count
            allocated += count

        # Allocate remaining lineups to balanced
        if allocated < num_lineups:
            lineup_distribution['balanced'] = (
                lineup_distribution.get('balanced', 0) +
                (num_lineups - allocated)
            )

        return lineup_distribution

# ============================================================================
# AI SYNTHESIS ENGINE WITH PATTERN ANALYSIS
# ============================================================================

class AISynthesisEngine:
    """Enhanced synthesis engine with pattern recognition"""

    def __init__(self):
        self.logger = get_logger()
        self.synthesis_history = deque(maxlen=20)

    def synthesize_recommendations(self,
                                   game_theory: AIRecommendation,
                                   correlation: AIRecommendation,
                                   contrarian: AIRecommendation) -> Dict:
        """
        Synthesize three AI perspectives into unified strategy

        DFS Value: Combines multiple strategic viewpoints for robust approach
        """
        self.logger.log("Synthesizing triple AI recommendations", "INFO")

        synthesis = {
            'captain_strategy': {},
            'player_rankings': {},
            'stacking_rules': [],
            'avoidance_rules': [],
            'enforcement_rules': [],
            'confidence': 0,
            'narrative': "",
            'patterns': []
        }

        # Captain synthesis with consensus tracking
        captain_votes = defaultdict(list)

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            for captain in rec.captain_targets[:5]:
                captain_votes[captain].append(ai_type.value)

        # Classify captains by consensus
        for captain, votes in captain_votes.items():
            if len(votes) == 3:
                synthesis['captain_strategy'][captain] = 'consensus'
            elif len(votes) == 2:
                synthesis['captain_strategy'][captain] = 'majority'
            else:
                synthesis['captain_strategy'][captain] = votes[0]

        # Player rankings synthesis with weighted scoring
        player_scores = defaultdict(float)

        weights = {
            AIStrategistType.GAME_THEORY:
                OptimizerConfig.AI_WEIGHTS.get('game_theory', 0.33),
            AIStrategistType.CORRELATION:
                OptimizerConfig.AI_WEIGHTS.get('correlation', 0.33),
            AIStrategistType.CONTRARIAN_NARRATIVE:
                OptimizerConfig.AI_WEIGHTS.get('contrarian', 0.34)
        }

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            weight = weights[ai_type]

            # Score must-play players
            for i, player in enumerate(rec.must_play[:5]):
                player_scores[player] += weight * rec.confidence * (1 - i * 0.1)

            # Negative scores for fades
            for player in rec.never_play[:3]:
                player_scores[player] -= weight * rec.confidence * 0.5

        # Normalize scores
        if player_scores:
            max_score = max(abs(score) for score in player_scores.values())
            if max_score > 0:
                synthesis['player_rankings'] = {
                    player: score / max_score
                    for player, score in player_scores.items()
                }

        # Stack synthesis - combine all unique stacks
        all_stacks = []
        stack_patterns = defaultdict(int)

        for rec in [game_theory, correlation, contrarian]:
            for stack in rec.stacks:
                all_stacks.append(stack)
                stack_type = stack.get('type', 'standard')
                stack_patterns[stack_type] += 1

        # Prioritize stacks
        synthesis['stacking_rules'] = self._prioritize_stacks(all_stacks)

        # Pattern analysis
        synthesis['patterns'] = self._analyze_patterns(
            game_theory, correlation, contrarian, stack_patterns
        )

        # Calculate overall confidence
        confidences = [
            game_theory.confidence * weights[AIStrategistType.GAME_THEORY],
            correlation.confidence * weights[AIStrategistType.CORRELATION],
            contrarian.confidence * weights[AIStrategistType.CONTRARIAN_NARRATIVE]
        ]
        synthesis['confidence'] = sum(confidences)

        # Create narrative
        narratives = []
        if game_theory.narrative:
            narratives.append(f"GT: {game_theory.narrative[:80]}")
        if correlation.narrative:
            narratives.append(f"Corr: {correlation.narrative[:80]}")
        if contrarian.narrative:
            narratives.append(f"Contra: {contrarian.narrative[:80]}")

        synthesis['narrative'] = " | ".join(narratives)

        # Store in history
        self.synthesis_history.append({
            'timestamp': datetime.now(),
            'confidence': synthesis['confidence'],
            'captain_count': len(synthesis['captain_strategy']),
            'patterns': synthesis['patterns']
        })

        return synthesis

    def _prioritize_stacks(self, all_stacks: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate stacks"""
        # Group similar stacks
        stack_groups = defaultdict(list)

        for stack in all_stacks:
            # Create a key for grouping
            if 'player1' in stack and 'player2' in stack:
                key = tuple(sorted([stack['player1'], stack['player2']]))
            elif 'players' in stack:
                key = tuple(sorted(stack['players'][:2]))
            else:
                key = stack.get('type', 'unknown')

            stack_groups[key].append(stack)

        # Select best from each group
        prioritized = []

        for group in stack_groups.values():
            if group:
                # Sort by correlation strength or confidence
                best = max(group, key=lambda s: s.get('correlation', 0.5))

                # Boost priority if multiple AIs suggested
                if len(group) > 1:
                    best['consensus'] = True
                    best['priority'] = (
                        best.get('priority', 50) + 10 * len(group)
                    )

                prioritized.append(best)

        # Sort by priority
        prioritized.sort(key=lambda s: s.get('priority', 50), reverse=True)

        return prioritized[:10]  # Top 10 stacks

    def _analyze_patterns(self,
                         game_theory: AIRecommendation,
                         correlation: AIRecommendation,
                         contrarian: AIRecommendation,
                         stack_patterns: Dict) -> List[str]:
        """Analyze patterns in AI recommendations"""
        patterns = []

        # Check for unanimous agreement
        captain_overlap = (
            set(game_theory.captain_targets) &
            set(correlation.captain_targets) &
            set(contrarian.captain_targets)
        )

        if captain_overlap:
            patterns.append(
                f"Strong consensus on {len(captain_overlap)} captains"
            )

        # Check for contrarian dominance
        confidences = {
            'game_theory': game_theory.confidence,
            'correlation': correlation.confidence,
            'contrarian': contrarian.confidence
        }

        max_conf = max(confidences.values())
        if confidences['contrarian'] == max_conf:
            patterns.append("Contrarian approach favored")

        # Stack pattern analysis
        if stack_patterns.get('onslaught', 0) > 1:
            patterns.append("Multiple onslaught stacks recommended")

        if stack_patterns.get('bring_back', 0) > 0:
            patterns.append("Bring-back correlation identified")

        return patterns

# ============================================================================
# END OF PART 3
# ============================================================================

# ============================================================================
# PART 4: BASE AI STRATEGIST & CLAUDE API MANAGER
# ============================================================================
# This part contains the foundation for AI strategists and the API manager
# with robust error handling, retry logic, and learning capabilities.

# ============================================================================
# BASE AI STRATEGIST WITH LEARNING AND ADAPTATION
# ============================================================================

class BaseAIStrategist:
    """
    Enhanced base class for all AI strategists with learning capabilities

    DFS Value: Learning from past performance improves future recommendations
    Key features:
    - Adaptive confidence based on historical accuracy
    - Slate profile analysis for context-aware recommendations
    - Statistical fallbacks when API unavailable
    - Performance tracking for continuous improvement
    """

    def __init__(self, api_manager=None, strategist_type: AIStrategistType = None):
        self.api_manager = api_manager
        self.strategist_type = strategist_type
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.response_cache = {}
        self.max_cache_size = 20
        self._cache_lock = threading.RLock()

        # Performance tracking for learning
        self.performance_history = deque(maxlen=100)
        self.successful_patterns = defaultdict(float)

        # Fallback confidence levels by AI type
        self.fallback_confidence = {
            AIStrategistType.GAME_THEORY: 0.55,
            AIStrategistType.CORRELATION: 0.60,
            AIStrategistType.CONTRARIAN_NARRATIVE: 0.50
        }

        # Adaptive confidence modifier (adjusted based on performance)
        self.adaptive_confidence_modifier = 1.0

        # Store DataFrame reference for fallback recommendations
        self.df = None

    def get_recommendation(self, df: pd.DataFrame, game_info: Dict,
                          field_size: str, use_api: bool = True) -> AIRecommendation:
        """
        Get AI recommendation with comprehensive error handling and learning

        Args:
            df: Player DataFrame
            game_info: Game context (total, spread, weather, etc.)
            field_size: Contest size configuration
            use_api: Whether to use API or manual input

        Returns:
            AIRecommendation object
        """
        try:
            # Store DataFrame for fallback use
            self.df = df

            # Validate inputs
            if df.empty:
                self.logger.log(
                    f"{self.strategist_type.value}: Empty DataFrame provided",
                    "ERROR"
                )
                return self._get_fallback_recommendation(df, field_size)

            # Analyze slate characteristics for dynamic adjustment
            slate_profile = self._analyze_slate_profile(df, game_info)

            # Generate cache key
            cache_key = self._generate_cache_key(df, game_info, field_size)

            # Check cache
            with self._cache_lock:
                if cache_key in self.response_cache:
                    self.logger.log(
                        f"{self.strategist_type.value}: Using cached recommendation",
                        "DEBUG"
                    )
                    cached_rec = self.response_cache[cache_key]
                    # Apply adaptive confidence
                    cached_rec.confidence *= self.adaptive_confidence_modifier
                    return cached_rec

            # Generate prompt with slate context
            prompt = self.generate_prompt(df, game_info, field_size, slate_profile)

            # Get response (API or manual)
            if use_api and self.api_manager and self.api_manager.client:
                response = self._get_api_response(prompt)
            else:
                response = self._get_fallback_response(
                    df, game_info, field_size, slate_profile
                )

            # Parse response into recommendation
            recommendation = self.parse_response(response, df, field_size)

            # Apply learned adjustments
            recommendation = self._apply_learned_adjustments(
                recommendation, slate_profile
            )

            # Validate recommendation
            is_valid, errors = recommendation.validate()
            if not is_valid:
                self.logger.log(
                    f"{self.strategist_type.value} validation errors: {errors}",
                    "WARNING"
                )
                recommendation = self._correct_recommendation(recommendation, df)

            # Add enforcement rules
            recommendation.enforcement_rules = self.create_enforcement_rules(
                recommendation, df, field_size, slate_profile
            )

            # Cache the result
            with self._cache_lock:
                self.response_cache[cache_key] = recommendation
                if len(self.response_cache) > self.max_cache_size:
                    # Remove oldest entries (FIFO)
                    for key in list(self.response_cache.keys())[:5]:
                        del self.response_cache[key]

            return recommendation

        except Exception as e:
            self.logger.log_exception(
                e,
                f"{self.strategist_type.value}.get_recommendation"
            )
            return self._get_fallback_recommendation(df, field_size)

    def _analyze_slate_profile(self, df: pd.DataFrame,
                               game_info: Dict) -> Dict:
        """
        Analyze slate characteristics for dynamic strategy adjustment

        DFS Value: Context-aware recommendations based on game environment
        """
        profile = {
            'player_count': len(df),
            'avg_salary': df['Salary'].mean(),
            'salary_range': df['Salary'].max() - df['Salary'].min(),
            'avg_ownership': df.get('Ownership', pd.Series([10])).mean(),
            'ownership_concentration': df.get('Ownership', pd.Series([10])).std(),
            'total': game_info.get('total', 45),
            'spread': abs(game_info.get('spread', 0)),
            'weather': game_info.get('weather', 'Clear'),
            'teams': df['Team'].nunique(),
            'positions': df['Position'].value_counts().to_dict(),
            'value_distribution': (
                df['Projected_Points'].std() / df['Projected_Points'].mean()
            ),
            'is_primetime': game_info.get('primetime', False),
            'injuries': game_info.get('injury_count', 0),
            'slate_type': self._determine_slate_type(df, game_info)
        }

        return profile

    def _determine_slate_type(self, df: pd.DataFrame, game_info: Dict) -> str:
        """
        Determine slate type for strategy adjustment

        Returns: 'shootout', 'low_scoring', 'blowout_risk', 'flat_pricing', 'standard'
        """
        total = game_info.get('total', 45)
        spread = abs(game_info.get('spread', 0))

        if total > 50 and spread < 3:
            return 'shootout'
        elif total < 40:
            return 'low_scoring'
        elif spread > 10:
            return 'blowout_risk'
        elif df['Salary'].std() < 1000:
            return 'flat_pricing'
        else:
            return 'standard'

    def _apply_learned_adjustments(self,
                                   recommendation: AIRecommendation,
                                   slate_profile: Dict) -> AIRecommendation:
        """
        Apply learned patterns and adjustments

        DFS Value: Historical performance improves future recommendations
        """
        # Adjust confidence based on slate type and historical performance
        slate_type = slate_profile.get('slate_type', 'standard')

        if slate_type in self.successful_patterns:
            confidence_boost = self.successful_patterns[slate_type] * 0.1
            recommendation.confidence = min(
                0.95,
                recommendation.confidence + confidence_boost
            )

        # Apply adaptive confidence modifier
        recommendation.confidence *= self.adaptive_confidence_modifier

        # Adjust captain targets based on slate profile
        if slate_type == 'shootout' and self.df is not None:
            # Prioritize pass catchers in shootouts
            qbs_and_receivers = []
            for player in recommendation.captain_targets:
                player_data = self.df[self.df['Player'] == player]
                if not player_data.empty:
                    if player_data.iloc[0]['Position'] in ['QB', 'WR', 'TE']:
                        qbs_and_receivers.append(player)

            if qbs_and_receivers:
                # Move QB/pass catchers to front
                recommendation.captain_targets = qbs_and_receivers + [
                    p for p in recommendation.captain_targets
                    if p not in qbs_and_receivers
                ]

        elif slate_type == 'blowout_risk' and self.df is not None:
            # Favor players from underdog team
            spread = slate_profile.get('spread', 0)
            if spread > 7:
                # Identify underdog team (this is simplified)
                teams = self.df['Team'].unique()
                # Adjust captain targets to favor underdog players
                pass

        return recommendation

    def track_performance(self, lineup: Dict, actual_points: float = None):
        """
        Track lineup performance for learning

        DFS Value: Continuous improvement through feedback
        """
        if actual_points is not None:
            performance_data = {
                'strategy': self.strategist_type.value,
                'projected': lineup.get('Projected', 0),
                'actual': actual_points,
                'accuracy': 1 - abs(
                    actual_points - lineup.get('Projected', 0)
                ) / max(actual_points, 1),
                'timestamp': datetime.now(),
                'slate_type': lineup.get('slate_type', 'standard')
            }

            self.performance_history.append(performance_data)

            # Update successful patterns by slate type
            slate_type = performance_data.get('slate_type', 'standard')
            if performance_data['accuracy'] > 0.8:
                self.successful_patterns[slate_type] += 1

            # Update adaptive confidence based on recent performance
            if len(self.performance_history) >= 10:
                recent_accuracy = np.mean([
                    p['accuracy']
                    for p in list(self.performance_history)[-10:]
                ])
                # Range: 0.5 to 1.5
                self.adaptive_confidence_modifier = 0.5 + recent_accuracy

    def _get_api_response(self, prompt: str) -> str:
        """Get response from API with error handling"""
        try:
            if not self.api_manager:
                return "{}"

            response = self.api_manager.get_ai_response(
                prompt,
                self.strategist_type
            )
            return response

        except Exception as e:
            self.logger.log_exception(e, f"{self.strategist_type.value} API call")
            return "{}"

    def _get_fallback_response(self, df: pd.DataFrame, game_info: Dict,
                               field_size: str, slate_profile: Dict) -> str:
        """
        Generate fallback response using statistical analysis

        DFS Value: CRITICAL - Enables operation without API
        Uses data-driven approach rather than empty response
        """
        # This will be implemented by child classes
        # Base implementation returns empty JSON
        return "{}"

    def _get_fallback_recommendation(self, df: pd.DataFrame,
                                    field_size: str) -> AIRecommendation:
        """
        Create fallback recommendation using statistical analysis
        OPTIMIZED: Reduced DataFrame copies (Recommendation #11)

        DFS Value: CRITICAL - Better than empty recommendation
        Uses actual data to make informed suggestions
        """
        if df.empty:
            return AIRecommendation(
                captain_targets=[],
                confidence=0.3,
                source_ai=self.strategist_type
            )

        # Use vectorized operations and filters instead of copies
        # Calculate leverage scores efficiently
        ownership = df['Ownership'].fillna(10)
        projected = df['Projected_Points']

        # Vectorized GPP score calculation
        gpp_scores = projected / (ownership + 5)

        # Adjust based on AI type
        if self.strategist_type == AIStrategistType.GAME_THEORY:
            # Focus on ownership leverage - use boolean masks
            low_own_mask = ownership < 15
            leverage_mask = ownership < 10
            high_own_mask = ownership > 35

            captains = df.loc[low_own_mask].nlargest(7, 'Projected_Points')['Player'].tolist()
            must_play = df.loc[leverage_mask].nlargest(3, 'Projected_Points')['Player'].tolist()
            never_play = df.loc[high_own_mask].nlargest(2, 'Ownership')['Player'].tolist()

        elif self.strategist_type == AIStrategistType.CORRELATION:
            # Focus on stacking opportunities - use position masks
            qb_mask = df['Position'] == 'QB'
            receiver_mask = df['Position'].isin(['WR', 'TE'])

            captains = df.loc[qb_mask].nlargest(3, 'Projected_Points')['Player'].tolist()
            captains += df.loc[receiver_mask].nlargest(4, 'Projected_Points')['Player'].tolist()

            must_play = []
            never_play = []

        else:  # CONTRARIAN_NARRATIVE
            # Focus on ultra-low ownership - use masks
            ultra_low_mask = ownership < 10
            very_low_mask = ownership < 5

            captains = df.loc[ultra_low_mask].nlargest(7, 'Projected_Points')['Player'].tolist()
            must_play = df.loc[very_low_mask].nlargest(2, 'Projected_Points')['Player'].tolist()
            never_play = df.nlargest(3, 'Ownership')['Player'].tolist()

        # Create basic stacks
        stacks = self._create_statistical_stacks(df)

        confidence = self.fallback_confidence.get(
            self.strategist_type,
            0.5
        )

        return AIRecommendation(
            captain_targets=captains[:7],
            must_play=must_play[:3],
            never_play=never_play[:3],
            stacks=stacks[:3],
            key_insights=[
                f"Statistical {self.strategist_type.value} fallback",
                f"Based on {len(df)} players in pool",
                "API unavailable - using data-driven approach"
            ],
            confidence=confidence,
            narrative=f"Data-driven {self.strategist_type.value} approach",
            source_ai=self.strategist_type
        )

    def _create_statistical_stacks(self, df: pd.DataFrame) -> List[Dict]:
        """Create stacks using statistical analysis"""
        stacks = []

        try:
            # QB stacks - use masks for efficiency
            qb_mask = df['Position'] == 'QB'
            qbs = df[qb_mask]

            for _, qb in qbs.iterrows():
                team = qb['Team']

                # Find pass catchers from same team
                teammates_mask = (df['Team'] == team) & df['Position'].isin(['WR', 'TE'])
                teammates = df[teammates_mask]

                if not teammates.empty:
                    top_teammate = teammates.nlargest(1, 'Projected_Points').iloc[0]

                    stacks.append({
                        'player1': qb['Player'],
                        'player2': top_teammate['Player'],
                        'type': 'QB_receiver',
                        'correlation': 0.6,
                        'priority': 70
                    })

        except Exception as e:
            self.logger.log(f"Error creating statistical stacks: {e}", "WARNING")

        return stacks

    def _correct_recommendation(self, recommendation: AIRecommendation,
                                df: pd.DataFrame) -> AIRecommendation:
        """Correct invalid recommendations"""
        available_players = set(df['Player'].values)

        # Filter captain targets to valid players
        recommendation.captain_targets = [
            p for p in recommendation.captain_targets
            if p in available_players
        ]

        # Add fallback captains if needed
        if len(recommendation.captain_targets) < 3:
            top_players = df.nlargest(
                5, 'Projected_Points'
            )['Player'].tolist()
            for player in top_players:
                if player not in recommendation.captain_targets:
                    recommendation.captain_targets.append(player)
                if len(recommendation.captain_targets) >= 5:
                    break

        # Filter must_play to valid players
        recommendation.must_play = [
            p for p in recommendation.must_play
            if p in available_players
        ]

        # Filter never_play to valid players
        recommendation.never_play = [
            p for p in recommendation.never_play
            if p in available_players
        ]

        return recommendation

    def _generate_cache_key(self, df: pd.DataFrame, game_info: Dict,
                           field_size: str) -> str:
        """Generate cache key for memoization"""
        # Create hash from key parameters
        key_components = [
            str(len(df)),
            str(df['Player'].iloc[0] if not df.empty else ''),
            str(game_info.get('total', 45)),
            str(game_info.get('spread', 0)),
            field_size,
            self.strategist_type.value
        ]

        key_string = "_".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    # Abstract methods to be implemented by child classes
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate AI prompt - must be implemented by child classes"""
        raise NotImplementedError

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse AI response - must be implemented by child classes"""
        raise NotImplementedError

    def create_enforcement_rules(self, recommendation: AIRecommendation,
                                df: pd.DataFrame, field_size: str,
                                slate_profile: Dict) -> List[Dict]:
        """Create enforcement rules - can be overridden by child classes"""
        rules = []

        # Validate players exist
        available_players = set(df['Player'].values)

        # Captain enforcement
        valid_captains = [
            c for c in recommendation.captain_targets
            if c in available_players
        ]

        if valid_captains:
            priority = int(recommendation.confidence * 100)

            if recommendation.confidence > 0.7:
                rules.append({
                    'type': 'hard',
                    'constraint': f'{self.strategist_type.value}_captain',
                    'players': valid_captains[:5],
                    'priority': priority,
                    'relaxation_tier': 2
                })
            else:
                rules.append({
                    'type': 'soft',
                    'constraint': f'{self.strategist_type.value}_captain_pref',
                    'players': valid_captains[:5],
                    'weight': recommendation.confidence,
                    'priority': int(priority * 0.7)
                })

        return rules

# ============================================================================
# CLAUDE API MANAGER WITH ROBUST ERROR HANDLING
# ============================================================================

class ClaudeAPIManager:
    """
    Enhanced Claude API manager with retry logic and comprehensive caching
    FIXED: Exponential backoff overflow (Recommendation #7)

    DFS Value: Reliable API access = consistent AI recommendations
    Key features:
    - Exponential backoff retry for transient failures
    - Thread-safe caching
    - Rate limit handling
    - Detailed usage statistics
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.cache = {}
        self.max_cache_size = 30
        self._cache_lock = threading.RLock()

        self.stats = {
            'requests': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_size': 0,
            'total_tokens': 0,
            'avg_response_time': 0,
            'by_ai': {
                AIStrategistType.GAME_THEORY: {
                    'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0
                },
                AIStrategistType.CORRELATION: {
                    'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0
                },
                AIStrategistType.CONTRARIAN_NARRATIVE: {
                    'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0
                }
            }
        }

        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Response time tracking
        self.response_times = deque(maxlen=100)

        self.initialize_client()

    def initialize_client(self):
        """Initialize Claude client with validation"""
        try:
            # Validate API key format
            if not self.api_key or not self.api_key.startswith('sk-'):
                raise ValueError("Invalid API key format")

            # Import and initialize client
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)

                # Test the connection
                self.validate_connection()

            except ImportError:
                self.logger.log(
                    "Anthropic library not installed. "
                    "Install with: pip install anthropic",
                    "ERROR"
                )
                self.client = None
                return

            self.logger.log("Claude API client initialized successfully", "INFO")

        except Exception as e:
            self.logger.log(f"Failed to initialize Claude API: {e}", "ERROR")
            self.client = None

    def get_ai_response(self, prompt: str,
                       ai_type: Optional[AIStrategistType] = None,
                       max_retries: int = 3) -> str:
        """
        Get response from Claude API with exponential backoff retry
        FIXED: Timeout overflow prevention (Recommendation #7)

        DFS Value: CRITICAL - Retry logic prevents failures from transient issues

        Args:
            prompt: Prompt for AI
            ai_type: Type of AI strategist (for tracking)
            max_retries: Maximum retry attempts

        Returns:
            AI response string (JSON format)
        """
        # Generate cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        with self._cache_lock:
            if prompt_hash in self.cache:
                self.stats['cache_hits'] += 1
                self.logger.log(
                    f"Cache hit for {ai_type.value if ai_type else 'unknown'}",
                    "DEBUG"
                )
                return self.cache[prompt_hash]

        # Update request statistics
        self.stats['requests'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['requests'] += 1

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                if not self.client:
                    raise Exception("API client not initialized")

                self.perf_monitor.start_timer("claude_api_call")
                start_time = time.time()

                # FIXED: Cap timeout to prevent overflow (Recommendation #7)
                timeout = min(30 * (1.5 ** attempt), 300)  # Max 5 minutes

                # Make API call
                message = self.client.messages.create(
                    model="claude-3-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=timeout,
                    system="""You are an expert Daily Fantasy Sports (DFS) strategist specializing in NFL tournament optimization.
                             You provide specific, actionable recommendations using exact player names and clear reasoning.
                             Always respond with valid JSON containing specific player recommendations.
                             Focus on game theory, correlations, and contrarian angles that win GPP tournaments.
                             Your recommendations must be enforceable as optimization constraints.""",
                    messages=[{"role": "user", "content": prompt}]
                )

                elapsed = self.perf_monitor.stop_timer("claude_api_call")

                # Extract response
                response = message.content[0].text if message.content else "{}"

                # Update statistics
                response_time = time.time() - start_time
                self.response_times.append(response_time)

                if ai_type:
                    self.stats['by_ai'][ai_type]['tokens'] += len(response) // 4
                    # Update average time
                    current_avg = self.stats['by_ai'][ai_type]['avg_time']
                    total_requests = self.stats['by_ai'][ai_type]['requests']
                    self.stats['by_ai'][ai_type]['avg_time'] = (
                        (current_avg * (total_requests - 1) + response_time) /
                        total_requests
                    )

                self.stats['total_tokens'] += len(response) // 4
                self.stats['avg_response_time'] = np.mean(
                    list(self.response_times)
                )

                # Cache response
                with self._cache_lock:
                    self.cache[prompt_hash] = response
                    self.stats['cache_size'] = len(self.cache)

                    # Manage cache size
                    if len(self.cache) > self.max_cache_size:
                        # Remove oldest entries (FIFO)
                        for key in list(self.cache.keys())[:10]:
                            del self.cache[key]

                self.logger.log(
                    f"AI response received for "
                    f"{ai_type.value if ai_type else 'unknown'} "
                    f"({len(response)} chars, {elapsed:.2f}s)",
                    "DEBUG"
                )

                return response

            except TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.log(
                        f"Timeout on attempt {attempt+1}/{max_retries}, "
                        f"retrying in {wait_time}s",
                        "WARNING"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.log(
                        "All API retries exhausted (timeout)",
                        "ERROR"
                    )
                    self.stats['errors'] += 1
                    if ai_type:
                        self.stats['by_ai'][ai_type]['errors'] += 1
                    return "{}"

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting
                if "rate_limit" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        self.logger.log(
                            f"Rate limited, waiting {wait_time}s",
                            "WARNING"
                        )
                        time.sleep(wait_time)
                        continue

                # Log error and potentially retry
                self.stats['errors'] += 1
                if ai_type:
                    self.stats['by_ai'][ai_type]['errors'] += 1

                if attempt < max_retries - 1:
                    self.logger.log(
                        f"API error on attempt {attempt+1}: {e}, retrying...",
                        "WARNING"
                    )
                    time.sleep(2 ** attempt)
                else:
                    self.logger.log(
                        f"API error for {ai_type.value if ai_type else 'unknown'}: {e}",
                        "ERROR"
                    )
                    return "{}"

        # Should not reach here, but return empty JSON as fallback
        return "{}"

    def validate_connection(self) -> bool:
        """
        Validate API connection is working

        Returns:
            True if connection successful
        """
        try:
            if not self.client:
                return False

            # Try a minimal test request
            test_prompt = "Respond with only the word: OK"

            message = self.client.messages.create(
                model="claude-3-sonnet-20241022",
                max_tokens=10,
                temperature=0,
                timeout=10,
                messages=[{"role": "user", "content": test_prompt}]
            )

            return bool(message.content)

        except Exception as e:
            self.logger.log(f"API validation failed: {e}", "ERROR")
            return False

    def get_stats(self) -> Dict:
        """Get comprehensive API usage statistics"""
        with self._cache_lock:
            return {
                'requests': self.stats['requests'],
                'errors': self.stats['errors'],
                'error_rate': (
                    self.stats['errors'] / max(self.stats['requests'], 1)
                ),
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': (
                    self.stats['cache_hits'] / max(self.stats['requests'], 1)
                ),
                'cache_size': len(self.cache),
                'total_tokens': self.stats['total_tokens'],
                'avg_response_time': self.stats['avg_response_time'],
                'by_ai': dict(self.stats['by_ai'])
            }

    def clear_cache(self):
        """Clear response cache"""
        with self._cache_lock:
            self.cache.clear()
            self.stats['cache_size'] = 0

        self.logger.log("API cache cleared", "INFO")

# ============================================================================
# END OF PART 4
# ============================================================================

# ============================================================================
# PART 5: INDIVIDUAL AI STRATEGISTS
# ============================================================================
# This part contains the three specialized AI strategists:
# 1. Game Theory - Ownership leverage and game theory
# 2. Correlation - Stacking and correlation analysis
# 3. Contrarian Narrative - Hidden angles and contrarian plays

# ============================================================================
# GPP GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist(BaseAIStrategist):
    """
    AI Strategist 1: Game Theory and Ownership Leverage

    DFS Value: Identifies ownership arbitrage opportunities
    Focus: Low-owned high-upside plays, field tendencies, leverage spots
    """

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate game theory focused prompt with slate context"""
        self.logger.log(
            f"Generating Game Theory prompt for {field_size}",
            "DEBUG"
        )

        if df.empty:
            return "Error: Empty player pool"

        # Prepare ownership analysis
        bucket_manager = AIOwnershipBucketManager()
        bucket_manager.adjust_thresholds_for_slate(df, field_size)
        buckets = bucket_manager.categorize_players(df)

        # Get low-owned high-upside plays using vectorized operations
        low_owned_mask = df['Ownership'] < 10
        low_owned_high_upside = df[low_owned_mask].nlargest(10, 'Projected_Points')

        # Field-specific strategies
        field_strategies = {
            'small_field': "Focus on slight differentiation while maintaining optimal plays",
            'medium_field': "Balance chalk with 2-3 strong leverage plays",
            'large_field': "Aggressive leverage with <15% owned captains",
            'large_field_aggressive': "Ultra-leverage approach with <10% captains",
            'milly_maker': "Maximum contrarian approach with <10% captains only"
        }

        # Slate adjustments
        slate_adjustments = {
            'shootout': "Prioritize ceiling over floor, embrace variance",
            'low_scoring': "Target TD-dependent players for leverage",
            'blowout_risk': "Fade favorites heavily, target garbage time",
            'flat_pricing': "Ownership becomes primary differentiator",
            'standard': "Balanced approach with calculated risks"
        }

        prompt = f"""You are an expert DFS game theory strategist. Create an ENFORCEABLE lineup strategy for {field_size} GPP tournaments.

GAME CONTEXT:
Teams: {game_info.get('teams', 'Unknown')}
Total: {game_info.get('total', 45)}
Spread: {game_info.get('spread', 0)}
Weather: {game_info.get('weather', 'Clear')}
Slate Type: {slate_profile.get('slate_type', 'standard')}

OWNERSHIP LANDSCAPE:
Mega Chalk (>35%): {len(buckets['mega_chalk'])} players - {buckets['mega_chalk'][:3] if buckets['mega_chalk'] else 'None'}
Chalk (20-35%): {len(buckets['chalk'])} players - {buckets['chalk'][:3] if buckets['chalk'] else 'None'}
Pivot (10-20%): {len(buckets['pivot'])} players - {buckets['pivot'][:3] if buckets['pivot'] else 'None'}
Leverage (5-10%): {len(buckets['leverage'])} players - {buckets['leverage'][:3] if buckets['leverage'] else 'None'}
Super Leverage (<5%): {len(buckets['super_leverage'])} players - {buckets['super_leverage'][:3] if buckets['super_leverage'] else 'None'}

HIGH LEVERAGE PLAYS (<10% ownership):
{low_owned_high_upside[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_string()}

FIELD STRATEGY:
{field_strategies.get(field_size, 'Standard GPP strategy')}
{slate_adjustments.get(slate_profile.get('slate_type', 'standard'), '')}

CRITICAL: Respond ONLY with valid JSON. DO NOT include markdown formatting or code blocks.

PROVIDE SPECIFIC, ENFORCEABLE RULES IN THIS EXACT JSON FORMAT:
{{
    "captain_rules": {{
        "must_be_one_of": ["Player Name 1", "Player Name 2", "Player Name 3"],
        "ownership_ceiling": 15,
        "min_projection": 15,
        "leverage_score_min": 3,
        "reasoning": "Brief explanation of why these captains"
    }},
    "lineup_rules": {{
        "must_include": ["Player Name"],
        "never_include": ["Player Name"],
        "ownership_sum_range": [60, 90],
        "min_leverage_players": 2,
        "max_chalk_players": 2
    }},
    "correlation_rules": {{
        "required_stacks": [{{"player1": "Player Name", "player2": "Player Name", "type": "leverage"}}],
        "avoid_negative_correlation": ["player_a", "player_b"]
    }},
    "game_theory_insights": {{
        "field_tendencies": "What most lineups will look like",
        "exploit_angle": "How to exploit these tendencies",
        "unique_construction": "Your differentiation strategy",
        "ownership_arbitrage": "Where ownership doesn't match equity"
    }},
    "confidence": 0.85,
    "key_insight": "The ONE game theory edge that wins"
}}

Use EXACT player names from the data provided. Focus on ownership leverage and exploitable patterns."""

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse game theory response with robust error handling"""
        try:
            # Clean response - remove markdown if present
            response = response.strip()
            response = response.replace('```json\n', '').replace('```\n', '').replace('```', '')

            # Try to parse JSON
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    self.logger.log(
                        f"JSON parse error: {e}, attempting text extraction",
                        "WARNING"
                    )
                    data = self._extract_from_text_response(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)

            # Extract captain rules
            captain_rules = data.get('captain_rules', {})
            captain_targets = captain_rules.get('must_be_one_of', [])
            valid_captains = [c for c in captain_targets if c in available_players]

            # If insufficient valid captains, use game theory selection
            if len(valid_captains) < 3:
                ownership_ceiling = captain_rules.get('ownership_ceiling', 15)
                min_proj = captain_rules.get('min_projection', 15)

                # Use vectorized filtering instead of multiple operations
                eligible_mask = (
                    (df['Ownership'] <= ownership_ceiling) &
                    (df['Projected_Points'] >= min_proj)
                )
                eligible = df[eligible_mask]

                if len(eligible) < 5:
                    # Relax constraints
                    eligible = df[df['Ownership'] <= ownership_ceiling * 1.5]

                if not eligible.empty:
                    # Calculate leverage score vectorized
                    max_proj = eligible['Projected_Points'].max()
                    leverage_scores = (
                        (eligible['Projected_Points'] / max_proj * 100) /
                        (eligible['Ownership'] + 5)
                    )

                    # Get top leverage plays
                    eligible = eligible.copy()
                    eligible['Leverage_Score'] = leverage_scores
                    leverage_captains = eligible.nlargest(5, 'Leverage_Score')['Player'].tolist()

                    for captain in leverage_captains:
                        if captain not in valid_captains:
                            valid_captains.append(captain)
                        if len(valid_captains) >= 5:
                            break

            # Extract lineup rules
            lineup_rules = data.get('lineup_rules', {})
            must_include = [
                p for p in lineup_rules.get('must_include', [])
                if p in available_players
            ]
            never_include = [
                p for p in lineup_rules.get('never_include', [])
                if p in available_players
            ]
            ownership_range = lineup_rules.get('ownership_sum_range', [60, 90])
            min_leverage = lineup_rules.get('min_leverage_players', 2)

            # Extract correlation rules and create stacks
            correlation_rules = data.get('correlation_rules', {})
            stacks = []

            for stack_data in correlation_rules.get('required_stacks', []):
                p1 = stack_data.get('player1')
                p2 = stack_data.get('player2')

                if p1 in available_players and p2 in available_players:
                    stacks.append({
                        'player1': p1,
                        'player2': p2,
                        'type': stack_data.get('type', 'leverage'),
                        'correlation': 0.5,
                        'leverage_based': True
                    })

            # Extract insights
            game_theory = data.get('game_theory_insights', {})
            key_insights = [
                data.get('key_insight', 'Ownership arbitrage opportunity'),
                game_theory.get('exploit_angle', ''),
                game_theory.get('unique_construction', ''),
                f"Target {ownership_range[0]}-{ownership_range[1]}% total ownership"
            ]
            key_insights = [i for i in key_insights if i]

            # Build enforcement rules
            enforcement_rules = self._build_game_theory_enforcement_rules(
                valid_captains, must_include, never_include,
                ownership_range, min_leverage, stacks
            )

            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))

            return AIRecommendation(
                captain_targets=valid_captains[:7],
                must_play=must_include[:5],
                never_play=never_include[:5],
                stacks=stacks[:5],
                key_insights=key_insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=game_theory.get('exploit_angle', 'Game theory optimization'),
                source_ai=AIStrategistType.GAME_THEORY,
                ownership_leverage={
                    'ownership_range': ownership_range,
                    'ownership_ceiling': captain_rules.get('ownership_ceiling', 15),
                    'min_leverage': min_leverage,
                    'max_chalk': lineup_rules.get('max_chalk_players', 2)
                }
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_game_theory_response")
            return self._get_fallback_recommendation(df, field_size)

    def _extract_from_text_response(self, response: str, df: pd.DataFrame) -> Dict:
        """Extract structured data from non-JSON response"""
        data = {
            'captain_rules': {'ownership_ceiling': 15},
            'lineup_rules': {'min_leverage_players': 2},
            'confidence': 0.6
        }

        lines = response.lower().split('\n')

        for line in lines:
            if 'captain' in line and any(char.isdigit() for char in line):
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    data['captain_rules']['ownership_ceiling'] = min(numbers)

            if 'leverage' in line and 'players' in line:
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    data['lineup_rules']['min_leverage_players'] = max(1, min(numbers))

        return data

    def _build_game_theory_enforcement_rules(self, captains, must_include,
                                            never_include, ownership_range,
                                            min_leverage, stacks):
        """Build game theory enforcement rules"""
        rules = []

        # Captain constraint (highest priority)
        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'game_theory_captain',
                'players': captains[:5],
                'priority': ConstraintPriority.AI_HIGH_CONFIDENCE.value,
                'relaxation_tier': 2,
                'description': 'Game theory optimal captains'
            })

        # Ownership sum constraint
        rules.append({
            'type': 'hard',
            'constraint': 'ownership_sum',
            'min': ownership_range[0],
            'max': ownership_range[1],
            'priority': ConstraintPriority.AI_MODERATE.value,
            'relaxation_tier': 2,
            'description': f'Total ownership {ownership_range[0]}-{ownership_range[1]}%'
        })

        # Must include players
        for i, player in enumerate(must_include[:3]):
            rules.append({
                'type': 'hard',
                'constraint': 'must_include',
                'player': player,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Must include {player}'
            })

        # Never include players
        for i, player in enumerate(never_include[:3]):
            rules.append({
                'type': 'hard',
                'constraint': 'must_exclude',
                'player': player,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Fade chalk: {player}'
            })

        return rules

# ============================================================================
# GPP CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist(BaseAIStrategist):
    """
    AI Strategist 2: Correlation and Stacking Patterns

    DFS Value: Maximizes ceiling through correlated outcomes
    Focus: QB stacks, game scripts, onslaught scenarios, bring-backs
    """

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CORRELATION)
        self.correlation_matrix = {}

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate correlation focused prompt"""
        self.logger.log(
            f"Generating Correlation prompt for {field_size}",
            "DEBUG"
        )

        if df.empty:
            return "Error: Empty player pool"

        # Team analysis using vectorized operations
        teams = df['Team'].unique()[:2]

        if len(teams) >= 2:
            team1, team2 = teams[0], teams[1]
            team1_df = df[df['Team'] == team1]
            team2_df = df[df['Team'] == team2]
        else:
            team1 = teams[0] if len(teams) > 0 else "Unknown"
            team2 = "Unknown"
            team1_df = df if len(teams) > 0 else pd.DataFrame()
            team2_df = pd.DataFrame()

        # Calculate correlation opportunities using vectorized operations
        qbs = df[df['Position'] == 'QB']['Player'].tolist()
        pass_catchers = df[df['Position'].isin(['WR', 'TE'])]['Player'].tolist()
        rbs = df[df['Position'] == 'RB']['Player'].tolist()

        # Determine game flow expectations
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        favorite = team1 if spread < 0 else team2
        underdog = team2 if favorite == team1 else team1

        prompt = f"""You are an expert DFS correlation strategist. Create SPECIFIC stacking rules for {field_size} GPP.

GAME ENVIRONMENT:
Total: {total} | Spread: {spread}
Favorite: {favorite} by {abs(spread)} points
Underdog: {underdog}
Slate Type: {slate_profile.get('slate_type', 'standard')}

TEAM 1 - {team1} ({'Favorite' if team1 == favorite else 'Underdog'}):
{team1_df[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team1_df.empty else 'No data'}

TEAM 2 - {team2} ({'Favorite' if team2 == favorite else 'Underdog'}):
{team2_df[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team2_df.empty else 'No data'}

CORRELATION TARGETS:
QBs: {qbs}
Pass Catchers: {pass_catchers[:8]}
RBs: {rbs}

CRITICAL: Respond ONLY with valid JSON. DO NOT include markdown formatting or code blocks.

CREATE ADVANCED CORRELATION RULES IN THIS EXACT JSON FORMAT:
{{
    "primary_stacks": [
        {{"type": "QB_WR1", "player1": "exact_qb_name", "player2": "exact_wr_name", "correlation": 0.7, "narrative": "why this stack"}}
    ],
    "onslaught_stacks": [
        {{"team": "team_name", "players": ["qb", "wr1", "wr2"], "scenario": "blowout correlation explanation"}}
    ],
    "bring_back_stacks": [
        {{"primary": ["qb", "wr"], "bring_back": "opposing_wr_name", "game_total": 50}}
    ],
    "negative_correlation": [
        {{"avoid_together": ["player1", "player2"], "reason": "why they're negatively correlated"}}
    ],
    "game_script_stacks": {{
        "shootout": ["player1", "player2", "player3"],
        "blowout": ["player1", "player2"]
    }},
    "captain_correlation": {{
        "best_captains_for_stacking": ["player_name_1", "player_name_2"],
        "stack_multipliers": {{"QB": 1.5, "WR1": 1.3}}
    }},
    "confidence": 0.8,
    "stack_narrative": "Primary correlation thesis"
}}

Use EXACT player names. Focus on correlations that maximize ceiling potential."""

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse correlation response with robust handling"""
        try:
            # Clean response
            response = response.strip()
            response = response.replace('```json\n', '').replace('```\n', '').replace('```', '')

            # Parse JSON
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log(
                        "JSON parse failed, using text extraction",
                        "WARNING"
                    )
                    data = self._extract_correlation_from_text(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)
            all_stacks = []

            # Process primary stacks
            for stack in data.get('primary_stacks', []):
                if self._validate_stack(stack, available_players):
                    stack['priority'] = ConstraintPriority.AI_MODERATE.value
                    stack['enforced'] = True
                    all_stacks.append(stack)

            # Process onslaught stacks (3+ players from winning team)
            for onslaught in data.get('onslaught_stacks', []):
                players = onslaught.get('players', [])
                valid_players = [p for p in players if p in available_players]

                if len(valid_players) >= 3:
                    all_stacks.append({
                        'type': 'onslaught',
                        'players': valid_players,
                        'team': onslaught.get('team', ''),
                        'scenario': onslaught.get('scenario', 'Blowout correlation'),
                        'priority': ConstraintPriority.AI_MODERATE.value,
                        'correlation': 0.6
                    })

            # Process bring-back stacks
            for bring_back in data.get('bring_back_stacks', []):
                primary = bring_back.get('primary', [])
                opponent = bring_back.get('bring_back', '')

                valid_primary = [p for p in primary if p in available_players]

                if valid_primary and opponent in available_players:
                    all_stacks.append({
                        'type': 'bring_back',
                        'primary_stack': valid_primary,
                        'bring_back': opponent,
                        'game_total': bring_back.get('game_total', 45),
                        'priority': ConstraintPriority.AI_MODERATE.value,
                        'correlation': 0.5
                    })

            # If no valid stacks from AI, create statistical ones
            if len(all_stacks) < 2:
                all_stacks.extend(self._create_statistical_stacks(df))

            # Extract captain correlation
            captain_rules = data.get('captain_correlation', {})
            captain_targets = captain_rules.get('best_captains_for_stacking', [])
            valid_captains = [c for c in captain_targets if c in available_players]

            # If no valid captains, use correlation-based selection
            if len(valid_captains) < 3:
                valid_captains.extend(
                    self._get_correlation_captains(df, all_stacks)
                )
                valid_captains = list(set(valid_captains))[:7]

            # Process negative correlations
            avoid_pairs = []
            for neg_corr in data.get('negative_correlation', []):
                players = neg_corr.get('avoid_together', [])
                if (len(players) >= 2 and
                    all(p in available_players for p in players[:2])):
                    avoid_pairs.append({
                        'players': players,
                        'reason': neg_corr.get('reason', 'negative correlation')
                    })

            # Build enforcement rules
            enforcement_rules = self._build_correlation_enforcement_rules(
                all_stacks, avoid_pairs, valid_captains
            )

            # Build correlation matrix
            self.correlation_matrix = self._build_correlation_matrix(
                all_stacks, avoid_pairs
            )

            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))

            # Extract insights
            key_insights = [
                data.get('stack_narrative', 'Correlation-based construction'),
                f"Primary focus: {all_stacks[0]['type'] if all_stacks else 'standard'} stacks",
                f"{len(all_stacks)} correlation plays identified"
            ]

            return AIRecommendation(
                captain_targets=valid_captains,
                must_play=[],
                never_play=[],
                stacks=all_stacks[:10],
                key_insights=key_insights,
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('stack_narrative', 'Correlation optimization'),
                source_ai=AIStrategistType.CORRELATION,
                correlation_matrix=self.correlation_matrix
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_correlation_response")
            return self._get_fallback_recommendation(df, field_size)

    def _validate_stack(self, stack: Dict, available_players: Set[str]) -> bool:
        """Validate stack players exist"""
        player1 = stack.get('player1')
        player2 = stack.get('player2')
        return (player1 and player2 and
                player1 in available_players and
                player2 in available_players)

    def _get_correlation_captains(self, df: pd.DataFrame,
                                 stacks: List[Dict]) -> List[str]:
        """Get captain targets based on correlation analysis"""
        captains = []

        # Prioritize QBs involved in stacks
        for stack in stacks:
            if stack.get('type') in ['QB_WR', 'QB_TE', 'primary', 'QB_WR1']:
                player1 = stack.get('player1')
                if player1:
                    # Use vectorized filtering
                    player_match = df[df['Player'] == player1]
                    if not player_match.empty and player_match.iloc[0]['Position'] == 'QB':
                        captains.append(player1)

        # Add primary pass catchers
        for stack in stacks:
            if 'player2' in stack:
                player2 = stack['player2']
                if player2 not in captains:
                    captains.append(player2)

        # Add bring-back targets
        for stack in stacks:
            if stack.get('type') == 'bring_back':
                bring_back = stack.get('bring_back')
                if bring_back and bring_back not in captains:
                    captains.append(bring_back)

        return captains[:7]

    def _build_correlation_enforcement_rules(self, stacks: List[Dict],
                                            avoid_pairs: List[Dict],
                                            captains: List[str]) -> List[Dict]:
        """Build correlation-specific enforcement rules"""
        rules = []

        # Captain rules
        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'correlation_captain',
                'players': captains[:5],
                'priority': ConstraintPriority.AI_HIGH_CONFIDENCE.value,
                'relaxation_tier': 2,
                'description': 'Correlation-optimized captains'
            })

        # High priority stacks
        high_priority_stacks = [
            s for s in stacks if s.get('priority', 0) >= 70
        ][:3]

        for i, stack in enumerate(high_priority_stacks):
            if stack.get('type') == 'onslaught':
                rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': 'onslaught_stack',
                    'players': stack['players'][:4],
                    'min_players': 3,
                    'weight': 0.9 if i > 0 else 1.0,
                    'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                    'relaxation_tier': 3,
                    'description': f"Onslaught: {stack.get('team', 'team')}"
                })
            elif stack.get('type') == 'bring_back':
                rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': 'bring_back_stack',
                    'primary': stack.get('primary_stack', []),
                    'bring_back': stack.get('bring_back'),
                    'weight': 0.8 if i > 0 else 1.0,
                    'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                    'relaxation_tier': 3,
                    'description': 'Bring-back correlation'
                })
            else:
                players = []
                if 'player1' in stack and 'player2' in stack:
                    players = [stack['player1'], stack['player2']]

                if players:
                    rules.append({
                        'type': 'hard' if i == 0 else 'soft',
                        'constraint': 'correlation_stack',
                        'players': players,
                        'correlation': stack.get('correlation', 0.5),
                        'weight': 0.8 if i > 0 else 1.0,
                        'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                        'relaxation_tier': 3,
                        'description': f"Stack: {stack.get('type', 'correlation')}"
                    })

        return rules

    def _build_correlation_matrix(self, stacks: List[Dict],
                                  avoid_pairs: List[Dict]) -> Dict:
        """Build correlation matrix for reference"""
        matrix = {}

        # Positive correlations from stacks
        for stack in stacks:
            if 'player1' in stack and 'player2' in stack:
                key = f"{stack['player1']}_{stack['player2']}"
                matrix[key] = stack.get('correlation', 0.5)

        # Negative correlations
        for avoid in avoid_pairs:
            players = avoid['players']
            if len(players) >= 2:
                key = f"{players[0]}_{players[1]}"
                matrix[key] = -0.5

        return matrix

    def _extract_correlation_from_text(self, response: str,
                                      df: pd.DataFrame) -> Dict:
        """Extract correlation data from text response"""
        data = {
            'primary_stacks': [],
            'confidence': 0.6
        }

        # Look for QB-receiver pairs mentioned using vectorized operations
        qb_mask = df['Position'] == 'QB'
        receiver_mask = df['Position'].isin(['WR', 'TE'])

        qbs = df[qb_mask]['Player'].tolist()
        receivers = df[receiver_mask]['Player'].tolist()

        for qb in qbs:
            for receiver in receivers:
                if qb in response and receiver in response:
                    # Check if same team using vectorized filtering
                    qb_team = df[df['Player'] == qb]['Team'].values
                    rec_team = df[df['Player'] == receiver]['Team'].values

                    if (len(qb_team) > 0 and len(rec_team) > 0 and
                        qb_team[0] == rec_team[0]):
                        data['primary_stacks'].append({
                            'player1': qb,
                            'player2': receiver,
                            'type': 'QB_stack',
                            'correlation': 0.6
                        })
                        break

        return data

# ============================================================================
# GPP CONTRARIAN NARRATIVE STRATEGIST
# ============================================================================

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """
    AI Strategist 3: Contrarian Narratives and Hidden Angles

    DFS Value: Finds non-obvious tournament-winning angles
    Focus: Chalk fades, hidden correlations, contrarian captains, leverage scenarios
    """

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate contrarian narrative focused prompt"""
        self.logger.log(
            f"Generating Contrarian Narrative prompt for {field_size}",
            "DEBUG"
        )

        if df.empty:
            return "Error: Empty player pool"

        # Calculate contrarian opportunities using vectorized operations
        ownership = df['Ownership'].fillna(10)
        projected = df['Projected_Points']
        salary = df['Salary']

        # Vectorized calculations
        value = projected / (salary / 1000)
        contrarian_score = (projected / projected.max()) / (ownership / 100 + 0.1)

        # Add computed columns efficiently
        df_analysis = df.copy()
        df_analysis['Value'] = value
        df_analysis['Contrarian_Score'] = contrarian_score

        # Find contrarian plays using vectorized filtering
        low_owned_high_ceiling = df_analysis[ownership < 10].nlargest(10, 'Projected_Points')
        hidden_value = df_analysis[ownership < 15].nlargest(10, 'Value')
        contrarian_captains = df_analysis.nlargest(10, 'Contrarian_Score')

        # Identify chalk to fade
        chalk_plays = df_analysis[ownership > 30].nlargest(5, 'Ownership')

        # Teams
        teams = df['Team'].unique()[:2]
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)

        prompt = f"""You are a contrarian DFS strategist who finds NON-OBVIOUS narratives that win GPP tournaments.

GAME SETUP:
{teams[0] if len(teams) > 0 else 'Team1'} vs {teams[1] if len(teams) > 1 else 'Team2'}
Total: {total} | Spread: {spread}
Slate Type: {slate_profile.get('slate_type', 'standard')}

CONTRARIAN OPPORTUNITIES:

LOW-OWNED HIGH CEILING (<10% owned):
{low_owned_high_ceiling[['Player', 'Position', 'Team', 'Projected_Points', 'Ownership']].to_string()}

HIDDEN VALUE PLAYS:
{hidden_value[['Player', 'Position', 'Salary', 'Value', 'Ownership']].to_string()}

CONTRARIAN CAPTAIN SCORES:
{contrarian_captains[['Player', 'Contrarian_Score', 'Ownership']].to_string()}

CHALK TO FADE (>30% owned):
{chalk_plays[['Player', 'Position', 'Ownership', 'Salary']].to_string() if not chalk_plays.empty else 'No major chalk'}

CRITICAL: Respond ONLY with valid JSON. DO NOT include markdown formatting or code blocks.

CREATE CONTRARIAN TOURNAMENT-WINNING NARRATIVES IN THIS EXACT JSON FORMAT:
{{
    "primary_narrative": "The ONE scenario that creates a unique winning lineup",
    "contrarian_captains": [
        {{"player": "exact_name", "narrative": "Why this 5% captain wins", "ceiling_path": "How they hit 30+ points"}}
    ],
    "hidden_correlations": [
        {{"player1": "name1", "player2": "name2", "narrative": "Non-obvious connection"}}
    ],
    "fade_the_chalk": [
        {{"player": "chalk_name", "ownership": 35, "fade_reason": "Specific bust risk", "pivot_to": "alternative_name"}}
    ],
    "leverage_scenarios": [
        {{"scenario": "Game script", "beneficiaries": ["player1", "player2"], "probability": "Low but high reward"}}
    ],
    "contrarian_game_theory": {{
        "what_field_expects": "Common narrative",
        "fatal_flaw": "Why the field is wrong",
        "exploit_angle": "How to capitalize"
    }},
    "tournament_winner": {{
        "captain": "exact_contrarian_captain",
        "core": ["player1", "player2"],
        "differentiators": ["unique1", "unique2"],
        "total_ownership": 65,
        "win_condition": "What needs to happen"
    }},
    "confidence": 0.7
}}

Use EXACT player names. Find the narrative that makes sub-5% plays optimal."""

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse contrarian narrative response"""
        try:
            # Clean response
            response = response.strip()
            response = response.replace('```json\n', '').replace('```\n', '').replace('```', '')

            # Parse JSON
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log(
                        "JSON parse failed, using narrative extraction",
                        "WARNING"
                    )
                    data = self._extract_narrative_from_text(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)

            # Extract contrarian captains
            contrarian_captains = []
            captain_narratives = {}

            for captain_data in data.get('contrarian_captains', []):
                player = captain_data.get('player')
                if player and player in available_players:
                    contrarian_captains.append(player)
                    # Use vectorized lookup
                    player_own = df[df['Player'] == player]['Ownership'].values
                    captain_narratives[player] = {
                        'narrative': captain_data.get('narrative', ''),
                        'ceiling_path': captain_data.get('ceiling_path', ''),
                        'ownership': player_own[0] if len(player_own) > 0 else 10
                    }

            # If insufficient contrarian captains, find statistically
            if len(contrarian_captains) < 3:
                contrarian_captains.extend(
                    self._find_statistical_contrarian_captains(
                        df, contrarian_captains
                    )
                )
                contrarian_captains = contrarian_captains[:7]

            # Extract tournament winner lineup
            tournament_winner = data.get('tournament_winner', {})
            tw_captain = tournament_winner.get('captain')
            tw_core = tournament_winner.get('core', [])
            tw_differentiators = tournament_winner.get('differentiators', [])

            must_play = []

            # Add tournament winner captain
            if tw_captain and tw_captain in available_players:
                if tw_captain not in contrarian_captains:
                    contrarian_captains.insert(0, tw_captain)

            # Add core players
            for player in tw_core:
                if player in available_players:
                    must_play.append(player)

            # Add differentiators
            for player in tw_differentiators:
                if player in available_players and player not in must_play:
                    must_play.append(player)

            # Extract fades and pivots
            fades = []
            pivots = {}

            for fade_data in data.get('fade_the_chalk', []):
                fade_player = fade_data.get('player')
                pivot_player = fade_data.get('pivot_to')

                if fade_player and fade_player in available_players:
                    # Verify it's actually chalk using vectorized lookup
                    player_ownership = df[df['Player'] == fade_player]['Ownership'].values
                    if len(player_ownership) > 0 and player_ownership[0] > 20:
                        fades.append(fade_player)

                        if pivot_player and pivot_player in available_players:
                            pivots[fade_player] = pivot_player
                            if pivot_player not in must_play:
                                must_play.append(pivot_player)

            # Extract hidden correlations
            hidden_stacks = []

            for corr in data.get('hidden_correlations', []):
                p1 = corr.get('player1')
                p2 = corr.get('player2')

                if p1 in available_players and p2 in available_players:
                    hidden_stacks.append({
                        'player1': p1,
                        'player2': p2,
                        'type': 'hidden',
                        'narrative': corr.get('narrative', 'Non-obvious correlation'),
                        'correlation': 0.4
                    })

            # Build enforcement rules
            enforcement_rules = self._build_contrarian_enforcement_rules(
                contrarian_captains, must_play, fades, hidden_stacks,
                captain_narratives
            )

            # Extract contrarian angles
            game_theory = data.get('contrarian_game_theory', {})
            contrarian_angles = [
                game_theory.get('fatal_flaw', ''),
                game_theory.get('exploit_angle', ''),
                data.get('primary_narrative', '')
            ]
            contrarian_angles = [a for a in contrarian_angles if a]

            # Extract insights
            key_insights = [
                data.get('primary_narrative', 'Contrarian approach'),
                f"Fade {len(fades)} chalk plays",
                f"{len(contrarian_captains)} contrarian captains identified"
            ]

            confidence = data.get('confidence', 0.7)
            confidence = max(0.0, min(1.0, confidence))

            return AIRecommendation(
                captain_targets=contrarian_captains,
                must_play=must_play[:5],
                never_play=fades[:5],
                stacks=hidden_stacks[:5],
                key_insights=key_insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('primary_narrative', 'Contrarian strategy'),
                source_ai=AIStrategistType.CONTRARIAN_NARRATIVE,
                contrarian_angles=contrarian_angles[:3]
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_contrarian_response")
            return self._get_fallback_recommendation(df, field_size)

    def _find_statistical_contrarian_captains(self, df: pd.DataFrame,
                                             existing: List[str]) -> List[str]:
        """
        Find contrarian captains using statistical analysis
        OPTIMIZED: Use vectorized operations without DataFrame copy (Recommendation #11)
        """
        # Use boolean masks for efficient filtering
        existing_set = set(existing)
        existing_mask = ~df['Player'].isin(existing_set)
        low_own_mask = df['Ownership'] < 15
        combined_mask = existing_mask & low_own_mask

        eligible = df[combined_mask]

        if eligible.empty:
            return []

        # Calculate contrarian score vectorized
        max_proj = eligible['Projected_Points'].max()
        contrarian_scores = (
            (eligible['Projected_Points'] / max_proj) /
            (eligible['Ownership'] / 100 + 0.1)
        )

        # Get top contrarian plays
        eligible = eligible.copy()  # Only copy the filtered subset
        eligible['Contrarian_Score'] = contrarian_scores
        captains = eligible.nlargest(5, 'Contrarian_Score')['Player'].tolist()

        return captains

    def _build_contrarian_enforcement_rules(self, captains: List[str],
                                           must_play: List[str],
                                           fades: List[str],
                                           hidden_stacks: List[Dict],
                                           captain_narratives: Dict) -> List[Dict]:
        """Build contrarian-specific enforcement rules"""
        rules = []

        # Ultra-contrarian captain rule
        ultra_contrarian = []
        moderate_contrarian = []

        for captain in captains[:5]:
            ownership = captain_narratives.get(captain, {}).get('ownership', 10)
            if ownership < 5:
                ultra_contrarian.append(captain)
            else:
                moderate_contrarian.append(captain)

        if ultra_contrarian:
            rules.append({
                'type': 'hard',
                'constraint': 'ultra_contrarian_captain',
                'players': ultra_contrarian,
                'priority': ConstraintPriority.AI_HIGH_CONFIDENCE.value,
                'relaxation_tier': 2,
                'description': 'Ultra-contrarian captain (<5% owned)'
            })

        if moderate_contrarian:
            rules.append({
                'type': 'soft',
                'constraint': 'contrarian_captain',
                'players': moderate_contrarian,
                'weight': 0.8,
                'priority': ConstraintPriority.AI_MODERATE.value,
                'description': 'Contrarian captain options'
            })

        # Tournament core rules
        for i, player in enumerate(must_play[:3]):
            rules.append({
                'type': 'hard' if i == 0 else 'soft',
                'constraint': f'tournament_core_{player}',
                'player': player,
                'weight': 0.9 - (i * 0.1) if i > 0 else 1.0,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Tournament core: {player}'
            })

        # Fade rules
        for i, fade in enumerate(fades[:3]):
            rules.append({
                'type': 'hard' if i == 0 else 'soft',
                'constraint': f'fade_chalk_{fade}',
                'player': fade,
                'exclude': True,
                'weight': 0.8 - (i * 0.1) if i > 0 else 1.0,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Fade chalk: {fade}'
            })

        return rules

    def _extract_narrative_from_text(self, response: str,
                                    df: pd.DataFrame) -> Dict:
        """Extract contrarian narrative from text response"""
        data = {
            'contrarian_captains': [],
            'fade_the_chalk': [],
            'confidence': 0.6
        }

        # Use vectorized filtering
        low_owned_mask = df['Ownership'] < 10
        high_owned_mask = df['Ownership'] > 30

        low_owned = df[low_owned_mask]['Player'].tolist()
        high_owned = df[high_owned_mask]['Player'].tolist()

        response_lower = response.lower()

        for player in low_owned:
            if player.lower() in response_lower:
                data['contrarian_captains'].append({
                    'player': player,
                    'narrative': 'Low ownership leverage'
                })

        for player in high_owned:
            if (f"fade {player.lower()}" in response_lower or
                f"avoid {player.lower()}" in response_lower):
                player_own = df[df['Player'] == player]['Ownership'].values
                data['fade_the_chalk'].append({
                    'player': player,
                    'ownership': player_own[0] if len(player_own) > 0 else 30
                })

        return data

# ============================================================================
# END OF PART 5
# ============================================================================

# ============================================================================
# PART 6: MAIN OPTIMIZER ENGINE WITH THREE-TIER CONSTRAINT RELAXATION
# ============================================================================
# This is the core optimization engine that generates lineups using AI
# recommendations with intelligent constraint relaxation.

# ============================================================================
# AI-DRIVEN GPP OPTIMIZER
# ============================================================================

class AIChefGPPOptimizer:
    """
    Main optimizer where AI is the chef and optimization is execution
    OPTIMIZED: Multiple performance improvements (Recommendations #1, #2, #5, #6)

    DFS Value: CRITICAL - Three-tier relaxation enables lineup diversity
    while maintaining DK rules and AI strategic intent

    Key Features:
    - Three-tier constraint relaxation (Strict -> Relaxed -> Minimal)
    - Thread-safe parallel generation
    - Lineup similarity detection
    - Dynamic strategy distribution
    - Performance tracking
    """

    def __init__(self, df: pd.DataFrame, game_info: Dict,
                 field_size: str = 'large_field', api_manager=None):
        # Validate inputs
        self._validate_inputs(df, game_info, field_size)

        self.df = df.copy()
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager

        # Initialize AI strategists
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.contrarian_ai = GPPContrarianNarrativeStrategist(api_manager)

        # Initialize core components
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Get field configuration
        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size,
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )

        # AI enforcement and synthesis
        enforcement_level_str = field_config.get('ai_enforcement', 'Strong')
        enforcement_level = AIEnforcementLevel[enforcement_level_str.upper()]

        self.enforcement_engine = AIEnforcementEngine(enforcement_level)
        self.synthesis_engine = AISynthesisEngine()

        # Supporting components
        self.bucket_manager = AIOwnershipBucketManager(self.enforcement_engine)

        # Tracking
        self.ai_decisions_log = []
        self.optimization_log = []
        self.generated_lineups = []

        self.lineup_generation_stats = {
            'attempts': 0,
            'successes': 0,
            'failures_by_reason': defaultdict(int)
        }

        # Threading for parallel generation
        self.max_workers = min(OptimizerConfig.MAX_PARALLEL_THREADS, 4)
        self.generation_timeout = OptimizerConfig.OPTIMIZATION_TIMEOUT

        # OPTIMIZATION: Similarity tracking with frozensets (Recommendation #5)
        self._lineup_signatures = []

        # OPTIMIZATION: Model cache for reuse (Recommendation #2)
        self._model_cache = {}

        # Prepare data
        self._prepare_data()

    def _validate_inputs(self, df: pd.DataFrame, game_info: Dict,
                        field_size: str):
        """
        Validate all inputs before initialization
        IMPROVED: Better error messages (Recommendation #12)
        """
        if df is None or df.empty:
            raise ValueError(
                "Player pool DataFrame cannot be empty.\n"
                "Suggestion: Upload a CSV file with player data."
            )

        required_columns = ['Player', 'Position', 'Team', 'Salary',
                          'Projected_Points']
        missing_columns = [col for col in required_columns
                          if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Suggestion: Ensure CSV has these column names (case-insensitive): "
                f"{', '.join(required_columns)}"
            )

        if field_size not in OptimizerConfig.FIELD_SIZE_CONFIGS:
            self.logger.log(
                f"Unknown field size {field_size}, using large_field",
                "WARNING"
            )

    def _prepare_data(self):
        """
        Prepare data with additional calculations
        OPTIMIZED: Create indexed lookups for O(1) access (Recommendation #1)
        """
        # Add ownership if missing
        if 'Ownership' not in self.df.columns:
            self.df['Ownership'] = self.df.apply(
                lambda row: OptimizerConfig.get_default_ownership(
                    row['Position'],
                    row['Salary'],
                    self.game_info.get('total', 47.0)
                ),
                axis=1
            )

        # Calculate value metrics
        self.df['Value'] = (
            self.df['Projected_Points'] / (self.df['Salary'] / 1000)
        )
        self.df['GPP_Score'] = (
            self.df['Value'] * (30 / (self.df['Ownership'] + 10))
        )

        # Add team counts for validation
        self.team_counts = self.df['Team'].value_counts().to_dict()

        # OPTIMIZATION: Create O(1) indexed lookups (Recommendation #1)
        # This eliminates expensive df[df['Player'] == player] operations
        self.df_indexed = self.df.set_index('Player')
        self.player_lookup = {
            'position': self.df_indexed['Position'].to_dict(),
            'team': self.df_indexed['Team'].to_dict(),
            'salary': self.df_indexed['Salary'].to_dict(),
            'points': self.df_indexed['Projected_Points'].to_dict(),
            'ownership': self.df_indexed['Ownership'].to_dict(),
            'value': self.df_indexed['Value'].to_dict(),
            'gpp_score': self.df_indexed['GPP_Score'].to_dict()
        }

    def get_triple_ai_strategies(self,
                                use_api: bool = True) -> Dict[AIStrategistType, AIRecommendation]:
        """
        Get strategies from all three AIs with parallel execution

        Returns:
            Dictionary mapping AI type to recommendation
        """
        self.logger.log("Getting strategies from three AI strategists", "INFO")
        self.perf_monitor.start_timer("get_ai_strategies")

        recommendations = {}

        if use_api and self.api_manager and self.api_manager.client:
            # API mode - parallel execution
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        self.game_theory_ai.get_recommendation,
                        self.df, self.game_info, self.field_size, use_api
                    ): AIStrategistType.GAME_THEORY,
                    executor.submit(
                        self.correlation_ai.get_recommendation,
                        self.df, self.game_info, self.field_size, use_api
                    ): AIStrategistType.CORRELATION,
                    executor.submit(
                        self.contrarian_ai.get_recommendation,
                        self.df, self.game_info, self.field_size, use_api
                    ): AIStrategistType.CONTRARIAN_NARRATIVE
                }

                for future in as_completed(futures):
                    ai_type = futures[future]
                    try:
                        recommendation = future.result(timeout=60)
                        recommendations[ai_type] = recommendation
                        self.logger.log(
                            f"{ai_type.value} recommendation received",
                            "INFO"
                        )
                    except Exception as e:
                        self.logger.log(
                            f"{ai_type.value} failed: {e}",
                            "ERROR"
                        )
                        # Use fallback
                        if ai_type == AIStrategistType.GAME_THEORY:
                            recommendations[ai_type] = (
                                self.game_theory_ai._get_fallback_recommendation(
                                    self.df, self.field_size
                                )
                            )
                        elif ai_type == AIStrategistType.CORRELATION:
                            recommendations[ai_type] = (
                                self.correlation_ai._get_fallback_recommendation(
                                    self.df, self.field_size
                                )
                            )
                        else:
                            recommendations[ai_type] = (
                                self.contrarian_ai._get_fallback_recommendation(
                                    self.df, self.field_size
                                )
                            )
        else:
            # Manual mode - sequential execution
            recommendations[AIStrategistType.GAME_THEORY] = (
                self.game_theory_ai.get_recommendation(
                    self.df, self.game_info, self.field_size, False
                )
            )
            recommendations[AIStrategistType.CORRELATION] = (
                self.correlation_ai.get_recommendation(
                    self.df, self.game_info, self.field_size, False
                )
            )
            recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = (
                self.contrarian_ai.get_recommendation(
                    self.df, self.game_info, self.field_size, False
                )
            )

        # Validate we have all recommendations
        for ai_type in AIStrategistType:
            if ai_type not in recommendations:
                self.logger.log(
                    f"Missing recommendation for {ai_type.value}",
                    "WARNING"
                )
                # Create minimal fallback
                recommendations[ai_type] = AIRecommendation(
                    captain_targets=[],
                    confidence=0.3,
                    source_ai=ai_type
                )

        # Log AI decisions
        for ai_type, rec in recommendations.items():
            self.logger.log_ai_decision(
                "strategy_received",
                ai_type.value,
                True,
                {
                    'captains': len(rec.captain_targets),
                    'confidence': rec.confidence,
                    'stacks': len(rec.stacks)
                },
                rec.confidence
            )

        elapsed = self.perf_monitor.stop_timer("get_ai_strategies")
        self.logger.log(f"AI strategies obtained in {elapsed:.2f}s", "INFO")

        return recommendations

    def synthesize_ai_strategies(self,
                                recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Synthesize three AI perspectives into unified strategy"""
        self.logger.log("Synthesizing triple AI strategies", "INFO")

        try:
            # Use synthesis engine
            synthesis = self.synthesis_engine.synthesize_recommendations(
                recommendations.get(AIStrategistType.GAME_THEORY),
                recommendations.get(AIStrategistType.CORRELATION),
                recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE)
            )

            # Create enforcement rules
            enforcement_rules = (
                self.enforcement_engine.create_enforcement_rules(recommendations)
            )

            # Validate rules are feasible
            validation = AIConfigValidator.validate_ai_requirements(
                enforcement_rules, self.df
            )

            if not validation['is_valid']:
                self.logger.log(
                    f"AI requirements validation failed: {validation['errors']}",
                    "WARNING"
                )

            return {
                'synthesis': synthesis,
                'enforcement_rules': enforcement_rules,
                'validation': validation,
                'recommendations': recommendations
            }

        except Exception as e:
            self.logger.log_exception(e, "synthesize_ai_strategies")
            return {
                'synthesis': {
                    'captain_strategy': {},
                    'confidence': 0.4,
                    'narrative': "Using fallback synthesis"
                },
                'enforcement_rules': {
                    'hard_constraints': [],
                    'soft_constraints': []
                },
                'validation': {'is_valid': True, 'errors': [], 'warnings': []},
                'recommendations': recommendations
            }

    def generate_ai_driven_lineups(self, num_lineups: int,
                                   ai_strategy: Dict) -> pd.DataFrame:
        """
        Generate lineups with three-tier constraint relaxation

        DFS Value: CRITICAL - Maintains diversity while respecting AI intent
        """
        self.perf_monitor.start_timer("total_optimization")
        start_time = time.time()

        self.logger.log_optimization_start(num_lineups, self.field_size, {
            'mode': 'AI-as-Chef',
            'enforcement': self.enforcement_engine.enforcement_level.value,
            'parallel': self.max_workers > 1
        })

        # Extract components
        synthesis = ai_strategy.get('synthesis', {})
        enforcement_rules = ai_strategy.get('enforcement_rules', {})

        # Prepare data structures using indexed lookups
        players = self.df['Player'].tolist()
        positions = self.player_lookup['position']
        teams = self.player_lookup['team']
        salaries = self.player_lookup['salary']
        points = self.player_lookup['points']
        ownership = self.player_lookup['ownership']

        # Apply AI modifications to projections
        ai_adjusted_points = self._apply_ai_adjustments(points, synthesis)

        # Get strategy distribution
        consensus_level = self._determine_consensus_level(synthesis)
        strategy_distribution = AIConfigValidator.get_ai_strategy_distribution(
            self.field_size, num_lineups, consensus_level
        )

        self.logger.log(
            f"AI Strategy distribution: {strategy_distribution}",
            "INFO"
        )

        all_lineups = []
        used_captains = set()

        # Generate lineups by strategy
        lineup_tasks = []
        for strategy, count in strategy_distribution.items():
            strategy_name = (
                strategy if isinstance(strategy, str) else strategy.value
            )
            for i in range(count):
                lineup_tasks.append((len(lineup_tasks) + 1, strategy_name))

        # Use parallel generation for better performance
        if self.max_workers > 1 and len(lineup_tasks) > 5:
            all_lineups = self._generate_lineups_parallel(
                lineup_tasks, players, salaries, ai_adjusted_points,
                ownership, positions, teams, enforcement_rules,
                synthesis, used_captains
            )
        else:
            # Sequential generation
            for i, (lineup_num, strategy_name) in enumerate(lineup_tasks):
                lineup = self._build_ai_enforced_lineup(
                    lineup_num=lineup_num,
                    strategy=strategy_name,
                    players=players,
                    salaries=salaries,
                    points=ai_adjusted_points,
                    ownership=ownership,
                    positions=positions,
                    teams=teams,
                    enforcement_rules=enforcement_rules,
                    synthesis=synthesis,
                    used_captains=used_captains
                )

                if lineup:
                    is_valid, violations = (
                        self.enforcement_engine.validate_lineup_against_ai(
                            lineup, enforcement_rules
                        )
                    )

                    if is_valid or len(all_lineups) < num_lineups * 0.5:
                        # Accept if valid or we need more lineups
                        all_lineups.append(lineup)
                        used_captains.add(lineup['Captain'])

        # Calculate metrics
        total_time = time.time() - start_time
        success_rate = len(all_lineups) / max(num_lineups, 1)

        self.logger.log_optimization_end(
            len(all_lineups), total_time, success_rate
        )

        # Store generated lineups
        self.generated_lineups = all_lineups

        return pd.DataFrame(all_lineups)

    def _generate_lineups_parallel(self, lineup_tasks, players, salaries, points,
                                   ownership, positions, teams, enforcement_rules,
                                   synthesis, used_captains):
        """
        Thread-safe parallel lineup generation
        FIXED: Race condition in captain selection (Recommendation #6)

        DFS Value: Speed without sacrificing correctness
        """
        all_lineups = []
        captain_lock = threading.Lock()

        def generate_single_lineup(task_data):
            lineup_num, strategy_name = task_data

            # Don't pass used_captains to avoid TOCTOU bug
            lineup = self._build_ai_enforced_lineup(
                lineup_num=lineup_num,
                strategy=strategy_name,
                players=players,
                salaries=salaries,
                points=points,
                ownership=ownership,
                positions=positions,
                teams=teams,
                enforcement_rules=enforcement_rules,
                synthesis=synthesis,
                used_captains=set()  # Empty set to avoid conflicts
            )

            if lineup:
                is_valid, violations = (
                    self.enforcement_engine.validate_lineup_against_ai(
                        lineup, enforcement_rules
                    )
                )

                if is_valid:
                    # FIXED: Atomic captain check and add (Recommendation #6)
                    with captain_lock:
                        if lineup['Captain'] not in used_captains:
                            used_captains.add(lineup['Captain'])
                            return lineup
                        else:
                            return None  # Captain already used

            return None

        # Use ThreadPoolExecutor for I/O-bound tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(generate_single_lineup, task)
                for task in lineup_tasks
            ]

            for future in as_completed(futures):
                try:
                    lineup = future.result(timeout=15)
                    if lineup:
                        all_lineups.append(lineup)
                except Exception as e:
                    self.logger.log(
                        f"Parallel generation error: {e}",
                        "DEBUG"
                    )

        return all_lineups

    def _build_ai_enforced_lineup(self, lineup_num: int, strategy: str,
                                  players: List[str], salaries: Dict,
                                  points: Dict, ownership: Dict,
                                  positions: Dict, teams: Dict,
                                  enforcement_rules: Dict,
                                  synthesis: Dict,
                                  used_captains: Set[str]) -> Optional[Dict]:
        """
        Build lineup with three-tier constraint relaxation

        DFS Value: CRITICAL - Three attempts with graduated relaxation
        Attempt 1: Strict (all AI constraints)
        Attempt 2: Relaxed (relax soft constraints)
        Attempt 3: Minimal (only DK rules + captain diversity)
        """
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                self.lineup_generation_stats['attempts'] += 1

                # Create optimization model
                model = pulp.LpProblem(
                    f"AI_Lineup_{lineup_num}_{strategy}_a{attempt}",
                    pulp.LpMaximize
                )

                # Decision variables
                flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')

                # AI-modified objective function
                player_weights = synthesis.get('player_rankings', {})

                objective = pulp.lpSum([
                    points[p] * player_weights.get(p, 1.0) * flex[p] +
                    1.5 * points[p] * player_weights.get(p, 1.0) * captain[p]
                    for p in players
                ])

                model += objective

                # TIER 1 CONSTRAINTS - NEVER RELAX (DK Rules)
                self._add_dk_constraints(
                    model, flex, captain, players, salaries, teams
                )

                # TIER 2 CONSTRAINTS - Relax on attempt 3
                if attempt < 2:
                    self._add_tier2_constraints(
                        model, flex, captain, enforcement_rules,
                        players, used_captains, attempt
                    )

                # TIER 3 CONSTRAINTS - Relax on attempt 2+
                if attempt == 0:
                    self._add_tier3_constraints(
                        model, flex, captain, enforcement_rules, players
                    )

                # Solve with timeout
                timeout = 5 + (attempt * 5)
                model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))

                if pulp.LpStatus[model.status] == 'Optimal':
                    lineup = self._extract_lineup_from_solution(
                        flex, captain, players, salaries, points, ownership,
                        lineup_num, strategy, synthesis, teams
                    )

                    if lineup and self._verify_dk_requirements(lineup, teams):
                        # Check similarity to existing lineups
                        if not self._is_too_similar(lineup):
                            self.lineup_generation_stats['successes'] += 1
                            if attempt > 0:
                                self.logger.log(
                                    f"Lineup {lineup_num} succeeded on "
                                    f"attempt {attempt + 1}",
                                    "DEBUG"
                                )
                            return lineup
                        else:
                            self.lineup_generation_stats['failures_by_reason']['too_similar'] += 1
                    elif lineup:
                        self.lineup_generation_stats['failures_by_reason']['dk_requirements'] += 1
                else:
                    self.lineup_generation_stats['failures_by_reason']['no_solution'] += 1

            except Exception as e:
                self.lineup_generation_stats['failures_by_reason']['exception'] += 1
                self.logger.log(
                    f"Lineup {lineup_num} attempt {attempt + 1} error: {str(e)}",
                    "DEBUG"
                )

        return None

    def _add_dk_constraints(self, model, flex, captain, players,
                           salaries, teams):
        """
        Add DraftKings Showdown constraints - NEVER RELAXED

        DFS Value: CRITICAL - These ensure valid DK submissions
        """
        # Exactly 1 captain
        model += pulp.lpSum(captain.values()) == 1, "One_Captain"

        # Exactly 5 FLEX
        model += pulp.lpSum(flex.values()) == 5, "Five_Flex"

        # Player can't be both captain and FLEX
        for p in players:
            model += flex[p] + captain[p] <= 1, f"Exclusive_{p}"

        # Salary cap
        model += pulp.lpSum([
            salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
            for p in players
        ]) <= OptimizerConfig.SALARY_CAP, "Salary_Cap"

        # DraftKings team diversity requirements
        unique_teams = list(set(teams.values()))

        if len(unique_teams) >= 2:
            team1, team2 = unique_teams[0], unique_teams[1]
            team1_players = [p for p in players if teams[p] == team1]
            team2_players = [p for p in players if teams[p] == team2]

            # Must have AT LEAST 1 from each team
            if team1_players:
                model += pulp.lpSum([
                    flex[p] + captain[p] for p in team1_players
                ]) >= 1, "Min_Team1"

            if team2_players:
                model += pulp.lpSum([
                    flex[p] + captain[p] for p in team2_players
                ]) >= 1, "Min_Team2"

            # Max 5 from one team
            if team1_players:
                model += pulp.lpSum([
                    flex[p] + captain[p] for p in team1_players
                ]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM, "Max_Team1"

            if team2_players:
                model += pulp.lpSum([
                    flex[p] + captain[p] for p in team2_players
                ]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM, "Max_Team2"

    def _add_tier2_constraints(self, model, flex, captain, enforcement_rules,
                               players, used_captains, attempt):
        """Add Tier 2 constraints (AI high-confidence) - Relax on attempt 3"""
        for constraint in enforcement_rules.get('hard_constraints', []):
            if not self.enforcement_engine.should_apply_constraint(
                constraint, attempt
            ):
                continue

            rule = constraint.get('rule')

            if rule in ['captain_from_list', 'captain_selection',
                       'game_theory_captain', 'correlation_captain',
                       'ultra_contrarian_captain']:
                valid_captains = [
                    p for p in constraint.get('players', [])
                    if p in players and p not in used_captains
                ]

                if valid_captains:
                    model += pulp.lpSum([
                        captain[c] for c in valid_captains
                    ]) == 1, f"AI_Captain_{rule}"

            elif rule == 'must_include':
                player = constraint.get('player')
                if player and player in players:
                    model += flex[player] + captain[player] >= 1, \
                        f"Must_Include_{player}"

            elif rule == 'must_exclude':
                player = constraint.get('player')
                if player and player in players:
                    model += flex[player] + captain[player] == 0, \
                        f"Must_Exclude_{player}"

    def _add_tier3_constraints(self, model, flex, captain, enforcement_rules,
                               players):
        """Add Tier 3 constraints (soft preferences) - Only on attempt 1"""
        # Stack constraints
        for stack_rule in enforcement_rules.get('stacking_rules', []):
            if stack_rule.get('rule') == 'correlation_stack':
                stack_players = [
                    p for p in stack_rule.get('players', [])
                    if p in players
                ]
                if len(stack_players) >= 2:
                    # Encourage but don't require
                    model += pulp.lpSum([
                        flex[p] + captain[p] for p in stack_players
                    ]) >= 1, f"Stack_{stack_players[0]}"

    def _extract_lineup_from_solution(self, flex, captain, players, salaries,
                                     points, ownership, lineup_num, strategy,
                                     synthesis, teams):
        """
        Extract lineup from solved model
        OPTIMIZED: Use indexed lookups instead of DataFrame filtering (Recommendation #1)
        """
        captain_pick = None
        flex_picks = []

        for p in players:
            if captain[p].value() == 1:
                captain_pick = p
            if flex[p].value() == 1:
                flex_picks.append(p)

        if captain_pick and len(flex_picks) == 5:
            # Use O(1) dictionary lookups instead of DataFrame filtering
            total_salary = (
                sum(salaries[p] for p in flex_picks) +
                1.5 * salaries[captain_pick]
            )
            total_proj = (
                sum(points[p] for p in flex_picks) +
                1.5 * points[captain_pick]
            )
            total_ownership = (
                ownership.get(captain_pick, 5) * 1.5 +
                sum(ownership.get(p, 5) for p in flex_picks)
            )

            # Calculate leverage score
            all_players = [captain_pick] + flex_picks
            leverage_score = self.bucket_manager.calculate_gpp_leverage(
                all_players, self.df
            )

            # Determine ownership tier
            if total_ownership < 60:
                ownership_tier = 'Elite Contrarian'
            elif total_ownership < 80:
                ownership_tier = 'Optimal'
            elif total_ownership < 100:
                ownership_tier = 'Balanced'
            else:
                ownership_tier = 'Chalky'

            return {
                'Lineup': lineup_num,
                'Strategy': strategy,
                'Captain': captain_pick,
                'Captain_Own%': ownership.get(captain_pick, 5),
                'FLEX': flex_picks,
                'Projected': round(total_proj, 2),
                'Salary': int(total_salary),
                'Salary_Remaining': int(
                    OptimizerConfig.SALARY_CAP - total_salary
                ),
                'Total_Ownership': round(total_ownership, 1),
                'Ownership_Tier': ownership_tier,
                'AI_Strategy': strategy,
                'AI_Enforced': True,
                'Confidence': synthesis.get('confidence', 0.5),
                'Leverage_Score': round(leverage_score, 2)
            }

        return None

    def _verify_dk_requirements(self, lineup: Dict, teams: Dict) -> bool:
        """
        Verify lineup meets DraftKings Showdown requirements

        DFS Value: CRITICAL - Ensures valid submissions
        """
        captain = lineup.get('Captain')
        flex_players = lineup.get('FLEX', [])

        if not captain or len(flex_players) != 5:
            return False

        all_players = [captain] + flex_players

        # Check team representation using O(1) lookups
        team_counts = defaultdict(int)
        for player in all_players:
            team = teams.get(player)
            if team:
                team_counts[team] += 1

        # Must have players from both teams
        unique_teams = set(teams.values())
        if len(team_counts) < len(unique_teams):
            return False

        # Max 5 from one team
        for count in team_counts.values():
            if count > 5:
                return False

        return True

    def _is_too_similar(self, new_lineup: Dict) -> bool:
        """
        Check if lineup is too similar to existing lineups
        OPTIMIZED: Use frozensets for fast set operations (Recommendation #5)

        DFS Value: Ensures lineup diversity for tournament equity
        """
        if not self._lineup_signatures:
            # First lineup, store and accept
            new_players = frozenset([new_lineup['Captain']] + new_lineup['FLEX'])
            self._lineup_signatures.append(new_players)
            return False

        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            self.field_size,
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )

        threshold = field_config.get('similarity_threshold', 0.67)

        # Create frozenset for O(1) set operations
        new_players = frozenset([new_lineup['Captain']] + new_lineup['FLEX'])

        # Only check last N lineups
        check_count = min(20, len(self._lineup_signatures))
        recent_signatures = self._lineup_signatures[-check_count:]

        for existing_sig in recent_signatures:
            # Fast set operations with frozensets
            intersection = len(new_players & existing_sig)
            union = len(new_players | existing_sig)

            if union > 0 and (intersection / union) > threshold:
                return True

        # Store signature for future checks
        self._lineup_signatures.append(new_players)

        # Memory management - keep last 100
        if len(self._lineup_signatures) > 100:
            self._lineup_signatures = self._lineup_signatures[-50:]

        return False

    def _apply_ai_adjustments(self, points: Dict, synthesis: Dict) -> Dict:
        """Apply AI-recommended adjustments to projections"""
        adjusted = points.copy()

        # Apply player rankings as multipliers
        rankings = synthesis.get('player_rankings', {})

        for player, score in rankings.items():
            if player in adjusted:
                # Normalize score to multiplier (0.7 to 1.3 range)
                if score > 0:
                    multiplier = 1.0 + min(score * 0.15, 0.3)
                else:
                    multiplier = max(0.7, 1.0 + score * 0.3)

                adjusted[player] *= multiplier

        return adjusted

    def _determine_consensus_level(self, synthesis: Dict) -> str:
        """Determine the level of AI consensus"""
        captain_strategy = synthesis.get('captain_strategy', {})

        if not captain_strategy:
            return 'low'

        consensus_count = len([
            c for c, level in captain_strategy.items()
            if level == 'consensus'
        ])
        total = len(captain_strategy)

        if total == 0:
            return 'low'

        consensus_ratio = consensus_count / total

        if consensus_ratio > 0.3:
            return 'high'
        elif consensus_ratio > 0.1:
            return 'mixed'
        else:
            return 'low'

# ============================================================================
# END OF PART 6
# ============================================================================

# ============================================================================
# PART 7: UI & HELPER FUNCTIONS - FINAL PART
# ============================================================================
# This final part contains all helper functions for validation, display,
# export, and the complete Streamlit UI application.

# ============================================================================
# HELPER FUNCTIONS - VALIDATION AND PROCESSING
# ============================================================================

def validate_and_process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced validation and processing of uploaded DataFrame
    IMPROVED: Better error messages with suggestions (Recommendation #12)

    Returns:
        (processed_df, validation_dict)
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'fixes_applied': [],
        'stats': {}
    }

    try:
        # Column mapping for different CSV formats
        column_mappings = {
            'name': 'Player',
            'first_name': 'First_Name',
            'last_name': 'Last_Name',
            'position': 'Position',
            'team': 'Team',
            'salary': 'Salary',
            'ppg_projection': 'Projected_Points',
            'ownership_projection': 'Ownership',
            'proj': 'Projected_Points',
            'own': 'Ownership',
            'sal': 'Salary',
            'pos': 'Position'
        }

        # Rename columns
        df = df.rename(columns={
            k.lower(): v for k, v in column_mappings.items()
        })

        # Create Player column if needed
        if 'Player' not in df.columns:
            if 'First_Name' in df.columns and 'Last_Name' in df.columns:
                df['Player'] = (
                    df['First_Name'].fillna('') + ' ' +
                    df['Last_Name'].fillna('')
                )
                df['Player'] = df['Player'].str.strip()
                validation['fixes_applied'].append(
                    "Created Player names from first/last name"
                )
            else:
                validation['errors'].append("Cannot determine player names")
                validation['warnings'].append(
                    "Suggestion: Ensure CSV has 'Player', 'Name', or 'First_Name'/'Last_Name' columns"
                )
                validation['is_valid'] = False
                return df, validation

        # Ensure numeric columns
        numeric_columns = ['Salary', 'Projected_Points', 'Ownership']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    na_count = df[col].isna().sum()
                    validation['warnings'].append(
                        f"{na_count} {col} values couldn't be converted to numbers"
                    )

                    # Fill with sensible defaults
                    if col == 'Salary':
                        df[col] = df[col].fillna(OptimizerConfig.MIN_SALARY)
                        validation['fixes_applied'].append(
                            f"Filled {na_count} missing salaries with minimum (${OptimizerConfig.MIN_SALARY:,})"
                        )
                    elif col == 'Projected_Points':
                        median = df[col].median()
                        df[col] = df[col].fillna(
                            median if not pd.isna(median) else 10
                        )
                        validation['fixes_applied'].append(
                            f"Filled {na_count} missing projections with median"
                        )
                    elif col == 'Ownership':
                        df[col] = df[col].fillna(10)
                        validation['fixes_applied'].append(
                            f"Filled {na_count} missing ownership with 10%"
                        )

        # Add ownership if missing
        if 'Ownership' not in df.columns:
            df['Ownership'] = df.apply(
                lambda row: OptimizerConfig.get_default_ownership(
                    row.get('Position', 'FLEX'),
                    row.get('Salary', 5000)
                ), axis=1
            )
            validation['fixes_applied'].append(
                "Added ownership projections based on position/salary"
            )

        # Validate ownership values
        if 'Ownership' in df.columns:
            invalid_own = df[
                (df['Ownership'] < 0) | (df['Ownership'] > 100)
            ]
            if not invalid_own.empty:
                df.loc[df['Ownership'] < 0, 'Ownership'] = 0
                df.loc[df['Ownership'] > 100, 'Ownership'] = 100
                validation['warnings'].append(
                    f"Corrected {len(invalid_own)} invalid ownership values (must be 0-100%)"
                )
                validation['fixes_applied'].append(
                    "Clamped ownership values to 0-100% range"
                )

        # Remove duplicates
        if df.duplicated(subset=['Player']).any():
            dup_count = df.duplicated(subset=['Player']).sum()
            df = df.drop_duplicates(subset=['Player'], keep='first')
            validation['warnings'].append(
                f"Removed {dup_count} duplicate players (kept first occurrence)"
            )

        # Validate minimum requirements
        validation['stats']['total_players'] = len(df)
        if len(df) < 6:
            validation['errors'].append(
                f"Only {len(df)} players found (minimum 6 required for Showdown)"
            )
            validation['warnings'].append(
                "Suggestion: Ensure CSV contains at least 6 players"
            )
            validation['is_valid'] = False
        elif len(df) < 12:
            validation['warnings'].append(
                f"Only {len(df)} players (12+ recommended for lineup diversity)"
            )

        # Validate team count
        teams = df['Team'].unique()
        validation['stats']['teams'] = len(teams)
        if len(teams) != 2:
            validation['warnings'].append(
                f"Expected 2 teams for Showdown, found {len(teams)}"
            )
            if len(teams) > 2:
                validation['warnings'].append(
                    f"Teams found: {', '.join(teams)}"
                )

        # Position distribution
        positions = df['Position'].value_counts()
        validation['stats']['positions'] = positions.to_dict()

        # Check for QBs
        if 'QB' not in positions or positions.get('QB', 0) == 0:
            validation['warnings'].append(
                "No QB in player pool - unusual for Showdown format"
            )

        # Validate salary feasibility
        min_lineup_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
        max_lineup_salary = df.nlargest(6, 'Salary')['Salary'].sum()

        validation['stats']['min_possible_salary'] = min_lineup_salary
        validation['stats']['max_possible_salary'] = max_lineup_salary

        if min_lineup_salary > OptimizerConfig.SALARY_CAP:
            validation['errors'].append(
                f"Minimum possible salary (${min_lineup_salary:,}) exceeds cap (${OptimizerConfig.SALARY_CAP:,})"
            )
            validation['warnings'].append(
                "Suggestion: Check that salary values are correct (should be in dollars, not thousands)"
            )
            validation['is_valid'] = False

        if max_lineup_salary < 35000:
            validation['warnings'].append(
                f"Maximum possible salary only ${max_lineup_salary:,} - limited lineup diversity"
            )

    except Exception as e:
        validation['errors'].append(f"Processing error: {str(e)}")
        validation['warnings'].append(
            "Suggestion: Verify CSV file format and column names"
        )
        validation['is_valid'] = False

    return df, validation

def init_session_state():
    """Initialize Streamlit session state"""
    import streamlit as st

    if 'ai_recommendations' not in st.session_state:
        st.session_state['ai_recommendations'] = {}
    if 'ai_synthesis' not in st.session_state:
        st.session_state['ai_synthesis'] = None
    if 'ai_strategy' not in st.session_state:
        st.session_state['ai_strategy'] = None
    if 'optimizer' not in st.session_state:
        st.session_state['optimizer'] = None
    if 'api_manager' not in st.session_state:
        st.session_state['api_manager'] = None
    if 'lineups_df' not in st.session_state:
        st.session_state['lineups_df'] = pd.DataFrame()
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()
    if 'game_info' not in st.session_state:
        st.session_state['game_info'] = {}
    if 'optimization_count' not in st.session_state:
        st.session_state['optimization_count'] = 0

def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing"""
    sample_data = {
        'Player': [
            'Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Dawson Knox',
            'James Cook', 'Devin Singletary', 'Isaiah McKenzie',
            'Tua Tagovailoa', 'Tyreek Hill', 'Jaylen Waddle',
            'Mike Gesicki', 'Raheem Mostert', 'Jeff Wilson Jr',
            'Cedrick Wilson', 'Durham Smythe', 'Buffalo DST'
        ],
        'Position': [
            'QB', 'WR', 'WR', 'TE', 'RB', 'RB', 'WR',
            'QB', 'WR', 'WR', 'TE', 'RB', 'RB', 'WR', 'TE', 'DST'
        ],
        'Team': [
            'BUF', 'BUF', 'BUF', 'BUF', 'BUF', 'BUF', 'BUF',
            'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'BUF'
        ],
        'Salary': [
            11200, 10800, 8600, 6800, 7200, 5800, 4800,
            10600, 11000, 9400, 6200, 6600, 5200, 4600, 3800, 4200
        ],
        'Projected_Points': [
            23.5, 19.2, 14.8, 11.2, 12.5, 9.8, 8.2,
            21.8, 20.5, 16.3, 10.5, 11.8, 8.9, 7.8, 6.2, 8.5
        ],
        'Ownership': [
            18.5, 22.3, 15.2, 8.5, 12.1, 6.8, 4.2,
            16.2, 25.8, 18.9, 7.2, 10.5, 5.5, 3.8, 2.1, 5.5
        ]
    }

    return pd.DataFrame(sample_data)

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_ai_recommendations(
    recommendations: Dict[AIStrategistType, AIRecommendation]
):
    """Display AI recommendations with clear formatting"""
    import streamlit as st

    st.markdown("### AI Strategic Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_confidence = np.mean([
            rec.confidence for rec in recommendations.values()
        ])
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")

    with col2:
        total_captains = len(set(
            captain for rec in recommendations.values()
            for captain in rec.captain_targets
        ))
        st.metric("Unique Captains", total_captains)

    with col3:
        total_stacks = sum(len(rec.stacks) for rec in recommendations.values())
        st.metric("Total Stacks", total_stacks)

    with col4:
        total_rules = sum(
            len(rec.enforcement_rules) for rec in recommendations.values()
        )
        st.metric("Enforcement Rules", total_rules)

    # Detailed recommendations in tabs
    tab1, tab2, tab3 = st.tabs([
        "Game Theory",
        "Correlation",
        "Contrarian"
    ])

    with tab1:
        rec = recommendations.get(AIStrategistType.GAME_THEORY)
        if rec:
            display_single_ai_recommendation(rec, "Game Theory")

    with tab2:
        rec = recommendations.get(AIStrategistType.CORRELATION)
        if rec:
            display_single_ai_recommendation(rec, "Correlation")

    with tab3:
        rec = recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE)
        if rec:
            display_single_ai_recommendation(rec, "Contrarian")

def display_single_ai_recommendation(rec: AIRecommendation, name: str):
    """Display single AI recommendation"""
    import streamlit as st

    confidence_label = (
        "High" if rec.confidence > 0.7 else
        "Medium" if rec.confidence > 0.5 else
        "Low"
    )

    st.markdown(f"#### {name} Strategy ({confidence_label} Confidence)")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Confidence", f"{rec.confidence:.0%}")

        if rec.narrative:
            with st.expander("Narrative", expanded=False):
                st.write(rec.narrative[:300])

        if rec.captain_targets:
            st.markdown("**Captain Targets:**")
            for i, captain in enumerate(rec.captain_targets[:5], 1):
                st.write(f"{i}. {captain}")

    with col2:
        if rec.must_play:
            st.markdown("**Must Play:**")
            for player in rec.must_play[:4]:
                st.write(f"• {player}")

        if rec.never_play:
            st.markdown("**Fade:**")
            for player in rec.never_play[:3]:
                st.write(f"• {player}")

        if rec.stacks:
            st.markdown("**Stacks:**")
            for stack in rec.stacks[:2]:
                if isinstance(stack, dict):
                    if 'players' in stack and len(stack['players']) > 2:
                        st.write(f"• {len(stack['players'])}-player stack")
                    elif 'player1' in stack and 'player2' in stack:
                        st.write(f"• {stack['player1'][:15]}...")

def display_ai_synthesis(synthesis: Dict):
    """Display AI synthesis"""
    import streamlit as st

    st.markdown("### AI Synthesis & Consensus")

    consensus_score = synthesis.get('confidence', 0) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Consensus Level")
        if consensus_score > 70:
            st.success(f"HIGH ({consensus_score:.0f}%)")
        elif consensus_score > 50:
            st.warning(f"MODERATE ({consensus_score:.0f}%)")
        else:
            st.info(f"LOW ({consensus_score:.0f}%)")

    with col2:
        captain_strategy = synthesis.get('captain_strategy', {})
        consensus_captains = len([
            c for c, l in captain_strategy.items() if l == 'consensus'
        ])
        majority_captains = len([
            c for c, l in captain_strategy.items() if l == 'majority'
        ])
        st.markdown("#### Captain Agreement")
        st.write(f"Consensus: {consensus_captains}")
        st.write(f"Majority: {majority_captains}")

    with col3:
        st.markdown("#### Rules")
        st.write(f"Total: {len(synthesis.get('enforcement_rules', []))}")
        st.write(f"Stacks: {len(synthesis.get('stacking_rules', []))}")

def display_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame,
                           synthesis: Dict, field_size: str):
    """Display comprehensive lineup analysis"""
    import streamlit as st

    if lineups_df.empty:
        st.warning("No lineups to analyze")
        return

    st.markdown("### Lineup Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Lineups", len(lineups_df))
        unique_captains = lineups_df['Captain'].nunique()
        st.metric("Unique Captains", unique_captains)

    with col2:
        if 'Projected' in lineups_df.columns:
            st.metric("Avg Projection", f"{lineups_df['Projected'].mean():.1f}")
            st.metric("Max Projection", f"{lineups_df['Projected'].max():.1f}")

    with col3:
        if 'Total_Ownership' in lineups_df.columns:
            st.metric(
                "Avg Ownership",
                f"{lineups_df['Total_Ownership'].mean():.1f}%"
            )

    with col4:
        if 'Leverage_Score' in lineups_df.columns:
            st.metric(
                "Avg Leverage",
                f"{lineups_df['Leverage_Score'].mean():.1f}"
            )

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Strategy Distribution
    if 'AI_Strategy' in lineups_df.columns:
        strategy_counts = lineups_df['AI_Strategy'].value_counts()
        axes[0, 0].pie(
            strategy_counts.values,
            labels=strategy_counts.index,
            autopct='%1.0f%%',
            startangle=90
        )
        axes[0, 0].set_title('Strategy Distribution')

    # 2. Captain Usage
    if 'Captain' in lineups_df.columns:
        captain_usage = lineups_df['Captain'].value_counts().head(10)
        axes[0, 1].barh(range(len(captain_usage)), captain_usage.values)
        axes[0, 1].set_yticks(range(len(captain_usage)))
        axes[0, 1].set_yticklabels(captain_usage.index)
        axes[0, 1].set_xlabel('Times Used')
        axes[0, 1].set_title('Top 10 Captain Usage')
        axes[0, 1].invert_yaxis()

    # 3. Ownership Distribution
    if 'Total_Ownership' in lineups_df.columns:
        axes[1, 0].hist(
            lineups_df['Total_Ownership'],
            bins=20,
            alpha=0.7,
            edgecolor='black'
        )

        # Add target range
        target_range = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(
            field_size, (60, 90)
        )
        axes[1, 0].axvspan(
            target_range[0], target_range[1],
            alpha=0.2, color='green'
        )

        axes[1, 0].set_xlabel('Total Ownership %')
        axes[1, 0].set_ylabel('Number of Lineups')
        axes[1, 0].set_title('Ownership Distribution')

    # 4. Projection Distribution
    if 'Projected' in lineups_df.columns:
        axes[1, 1].hist(
            lineups_df['Projected'],
            bins=15,
            alpha=0.7,
            edgecolor='black'
        )
        axes[1, 1].set_xlabel('Projected Points')
        axes[1, 1].set_ylabel('Number of Lineups')
        axes[1, 1].set_title('Projection Distribution')

    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_lineups_draftkings(lineups_df: pd.DataFrame) -> str:
    """Export lineups in DraftKings Showdown format"""
    try:
        dk_lineups = []

        for idx, row in lineups_df.iterrows():
            flex_players = (
                row['FLEX'] if isinstance(row['FLEX'], list) else []
            )

            # Ensure exactly 5 FLEX
            while len(flex_players) < 5:
                flex_players.append('')

            dk_lineups.append({
                'CPT': row.get('Captain', ''),
                'FLEX 1': flex_players[0],
                'FLEX 2': flex_players[1],
                'FLEX 3': flex_players[2],
                'FLEX 4': flex_players[3],
                'FLEX 5': flex_players[4]
            })

        dk_df = pd.DataFrame(dk_lineups)
        return dk_df.to_csv(index=False)

    except Exception as e:
        get_logger().log(f"Export error: {e}", "ERROR")
        return ""

def export_detailed_lineups(lineups_df: pd.DataFrame) -> str:
    """Export detailed lineup information"""
    try:
        detailed = []

        for idx, row in lineups_df.iterrows():
            lineup_detail = {
                'Lineup': row.get('Lineup', idx + 1),
                'Strategy': row.get('Strategy', ''),
                'Captain': row.get('Captain', ''),
                'Captain_Own%': row.get('Captain_Own%', 0),
                'Projected': row.get('Projected', 0),
                'Salary': row.get('Salary', 0),
                'Salary_Remaining': row.get('Salary_Remaining', 0),
                'Total_Ownership': row.get('Total_Ownership', 0),
                'Ownership_Tier': row.get('Ownership_Tier', ''),
                'Leverage_Score': row.get('Leverage_Score', 0),
                'AI_Enforced': row.get('AI_Enforced', False),
                'Confidence': row.get('Confidence', 0)
            }

            # Add FLEX players
            flex_players = row.get('FLEX', [])
            for i, player in enumerate(flex_players):
                lineup_detail[f'FLEX_{i+1}'] = player

            # Ensure all 5 FLEX columns
            for i in range(len(flex_players), 5):
                lineup_detail[f'FLEX_{i+1}'] = ''

            detailed.append(lineup_detail)

        detailed_df = pd.DataFrame(detailed)
        return detailed_df.to_csv(index=False)

    except Exception as e:
        get_logger().log(f"Detailed export error: {e}", "ERROR")
        return ""

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main application"""
    import streamlit as st

    # Page configuration
    st.set_page_config(
        page_title="NFL GPP AI-Chef Optimizer",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("NFL GPP Tournament Optimizer - AI-as-Chef Edition")
    st.markdown("*Version 6.5 - Optimized Performance & Three-Tier Relaxation*")

    # Initialize session state
    init_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("AI-Chef Configuration")

        # Quick Settings
        with st.expander("Quick Settings", expanded=True):
            quick_mode = st.radio(
                "Optimization Mode",
                ["Balanced", "Aggressive", "Conservative"]
            )

            if quick_mode == "Aggressive":
                enforcement_level = 'Mandatory'
                field_preset = 'large_field_aggressive'
            elif quick_mode == "Conservative":
                enforcement_level = 'Moderate'
                field_preset = 'small_field'
            else:
                enforcement_level = 'Strong'
                field_preset = 'large_field'

        st.markdown("---")

        # Contest Type
        st.markdown("### Contest Type")
        contest_type = st.selectbox(
            "Select GPP Type",
            list(OptimizerConfig.FIELD_SIZES.keys()),
            index=list(OptimizerConfig.FIELD_SIZES.values()).index(field_preset)
        )
        field_size = OptimizerConfig.FIELD_SIZES[contest_type]

        # API Configuration
        st.markdown("---")
        st.markdown("### AI Connection")

        use_api = st.checkbox("Use Claude API (Optional)", value=False)

        api_manager = None
        if use_api:
            api_key = st.text_input(
                "Claude API Key",
                type="password",
                placeholder="sk-ant-api03-..."
            )

            if api_key and st.button("Connect to Claude"):
                try:
                    with st.spinner("Connecting..."):
                        api_manager = ClaudeAPIManager(api_key)
                        if api_manager.validate_connection():
                            st.success("Connected to Claude AI")
                            st.session_state['api_manager'] = api_manager
                        else:
                            st.error("Connection validation failed")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

            if 'api_manager' in st.session_state:
                api_manager = st.session_state['api_manager']

    # Main Content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Upload",
        "AI Analysis",
        "Generate Lineups",
        "Results",
        "Export"
    ])

    with tab1:
        st.markdown("## Data & Game Configuration")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload DraftKings Showdown CSV",
                type=['csv']
            )

            if st.checkbox("Use sample data for testing"):
                sample_data = create_sample_data()
                st.session_state['df'] = sample_data
                st.success("Sample data loaded")

        with col2:
            st.markdown("### File Requirements")
            st.write("Required columns:")
            st.write("• Player names")
            st.write("• Position")
            st.write("• Team")
            st.write("• Salary")
            st.write("• Projected Points")

        if uploaded_file is not None or (
            'df' in st.session_state and not st.session_state['df'].empty
        ):
            try:
                if uploaded_file is not None:
                    raw_df = pd.read_csv(uploaded_file)
                    df, validation = validate_and_process_dataframe(raw_df)
                else:
                    df = st.session_state['df']
                    validation = {
                        'is_valid': True,
                        'errors': [],
                        'warnings': []
                    }

                # Display validation
                if validation['errors']:
                    st.error("Validation Errors:")
                    for error in validation['errors']:
                        st.write(f"  • {error}")

                if validation['warnings']:
                    st.warning("Warnings:")
                    for warning in validation['warnings']:
                        st.write(f"  • {warning}")

                if validation.get('fixes_applied'):
                    with st.expander("Automatic Fixes Applied", expanded=False):
                        for fix in validation['fixes_applied']:
                            st.write(f"• {fix}")

                if not validation['is_valid']:
                    st.error("Cannot proceed due to validation errors")
                    st.stop()

                st.session_state['df'] = df
                st.success(f"Loaded {len(df)} players successfully")

                # Game configuration
                st.markdown("### Game Setup")
                col1, col2, col3, col4 = st.columns(4)

                teams = df['Team'].unique()
                with col1:
                    team_display = (
                        f"{teams[0]} vs {teams[1]}"
                        if len(teams) >= 2 else "Teams"
                    )
                    game_teams = st.text_input("Teams", team_display)

                with col2:
                    total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5)

                with col3:
                    spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)

                with col4:
                    weather = st.selectbox(
                        "Weather",
                        ["Clear", "Wind", "Rain", "Snow"]
                    )

                game_info = {
                    'teams': game_teams,
                    'total': total,
                    'spread': spread,
                    'weather': weather
                }

                st.session_state['game_info'] = game_info

                # Display summary
                with st.expander("View Player Pool", expanded=False):
                    st.dataframe(
                        df[['Player', 'Position', 'Team', 'Salary',
                           'Projected_Points', 'Ownership']],
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                get_logger().log_exception(e, "data_upload")

    with tab2:
        st.markdown("## AI Strategic Analysis")

        if 'df' not in st.session_state or st.session_state['df'].empty:
            st.warning("Please upload data first")
            st.stop()

        df = st.session_state['df']
        game_info = st.session_state.get('game_info', {})

        if st.button("Generate AI Strategies", type="primary"):
            try:
                optimizer = AIChefGPPOptimizer(
                    df, game_info, field_size, api_manager
                )

                with st.spinner("Consulting AI System..."):
                    ai_recommendations = optimizer.get_triple_ai_strategies(
                        use_api=use_api
                    )

                if ai_recommendations:
                    st.session_state['ai_recommendations'] = ai_recommendations
                    st.session_state['optimizer'] = optimizer

                    display_ai_recommendations(ai_recommendations)

                    with st.spinner("Synthesizing strategies..."):
                        ai_strategy = optimizer.synthesize_ai_strategies(
                            ai_recommendations
                        )

                    st.session_state['ai_synthesis'] = ai_strategy['synthesis']
                    st.session_state['ai_strategy'] = ai_strategy

                    display_ai_synthesis(ai_strategy['synthesis'])

                    st.success("AI Analysis Complete")

            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")
                get_logger().log_exception(e, "ai_analysis")

    with tab3:
        st.markdown("## Generate AI-Driven Lineups")

        if 'ai_strategy' not in st.session_state:
            st.warning("Please generate AI strategies first")
            st.stop()

        num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=20,
            step=5
        )

        if st.button("Generate Lineups", type="primary"):
            try:
                optimizer = st.session_state.get('optimizer')
                ai_strategy = st.session_state.get('ai_strategy')

                if not optimizer or not ai_strategy:
                    st.error("Missing optimizer or strategy")
                    st.stop()

                with st.spinner(f"Building {num_lineups} lineups..."):
                    lineups_df = optimizer.generate_ai_driven_lineups(
                        num_lineups, ai_strategy
                    )

                if not lineups_df.empty:
                    st.session_state['lineups_df'] = lineups_df

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lineups Generated", len(lineups_df))
                    with col2:
                        st.metric(
                            "Unique Captains",
                            lineups_df['Captain'].nunique()
                        )
                    with col3:
                        if 'Total_Ownership' in lineups_df.columns:
                            st.metric(
                                "Avg Ownership",
                                f"{lineups_df['Total_Ownership'].mean():.1f}%"
                            )

                    st.success(f"Generated {len(lineups_df)} lineups")
                else:
                    st.error("No lineups generated")

            except Exception as e:
                st.error(f"Generation error: {str(e)}")
                get_logger().log_exception(e, "lineup_generation")

    with tab4:
        st.markdown("## Results & Analysis")

        if 'lineups_df' not in st.session_state or (
            st.session_state['lineups_df'].empty
        ):
            st.warning("No lineups to analyze")
            st.stop()

        lineups_df = st.session_state['lineups_df']
        df = st.session_state.get('df', pd.DataFrame())
        synthesis = st.session_state.get('ai_synthesis', {})

        display_lineup_analysis(lineups_df, df, synthesis, field_size)

    with tab5:
        st.markdown("## Export Options")

        if 'lineups_df' not in st.session_state or (
            st.session_state['lineups_df'].empty
        ):
            st.warning("No lineups to export")
            st.stop()

        lineups_df = st.session_state['lineups_df']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### DraftKings Format")
            dk_csv = export_lineups_draftkings(lineups_df)
            if dk_csv:
                st.download_button(
                    label="Download DK CSV",
                    data=dk_csv,
                    file_name=f"dk_lineups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

        with col2:
            st.markdown("### Detailed Format")
            detailed_csv = export_detailed_lineups(lineups_df)
            if detailed_csv:
                st.download_button(
                    label="Download Detailed CSV",
                    data=detailed_csv,
                    file_name=f"detailed_lineups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import streamlit as st
        st.error(f"Critical error: {str(e)}")
        get_logger().log_exception(e, "main_entry", critical=True)

# ============================================================================
# END OF PART 7 - SCRIPT COMPLETE
# ============================================================================
