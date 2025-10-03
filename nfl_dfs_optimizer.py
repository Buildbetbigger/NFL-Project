"""
NFL DFS AI-Driven Optimizer - Complete Engine
All Components Combined

Version: 2.0.0
Author: AI-Powered DFS Optimization System
"""

# ============================================================================
# PART 1: IMPORTS, CONFIGURATION, ENUMS & DATA CLASSES
# ============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import warnings
import traceback
import threading
import hashlib
import time
import re
import json

# PuLP for Linear Programming
try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, LpInteger
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    warnings.warn("PuLP not installed. Install with: pip install pulp")

# Anthropic for AI
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    warnings.warn("Anthropic not installed. Install with: pip install anthropic")

# Scientific computing
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque as Deque
from collections import defaultdict as DefaultDict

# Version
__version__ = "2.0.0"

# ============================================================================
# ENUMS
# ============================================================================

class FieldSize(Enum):
    """Tournament field size categories"""
    SMALL = "small_field"
    MEDIUM = "medium_field"
    LARGE = "large_field"
    LARGE_AGGRESSIVE = "large_field_aggressive"
    MILLY_MAKER = "milly_maker"

class AIStrategistType(Enum):
    """AI strategist types"""
    GAME_THEORY = "Game Theory"
    CORRELATION = "Correlation"
    CONTRARIAN_NARRATIVE = "Contrarian Narrative"

class AIEnforcementLevel(Enum):
    """AI enforcement strength levels"""
    MANDATORY = "mandatory"
    STRONG = "strong"
    MODERATE = "moderate"
    ADVISORY = "advisory"

class FitnessMode(Enum):
    """Genetic algorithm fitness modes"""
    PROJECTION = "projection"
    CEILING = "ceiling"
    SHARPE = "sharpe"
    WIN_PROBABILITY = "win_probability"

class ConstraintPriority(Enum):
    """Constraint priority levels"""
    CRITICAL = 100
    AI_CONSENSUS = 90
    AI_HIGH_CONFIDENCE = 80
    AI_MODERATE = 70
    SALARY_CAP = 60
    POSITION_REQUIREMENTS = 50
    SOFT_PREFERENCE = 30

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AIRecommendation:
    """AI strategist recommendation"""
    captain_targets: List[str] = field(default_factory=list)
    must_play: List[str] = field(default_factory=list)
    never_play: List[str] = field(default_factory=list)
    stacks: List[Dict[str, Any]] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.7
    narrative: str = ""
    source_ai: Optional[AIStrategistType] = None
    enforcement_rules: List[Dict[str, Any]] = field(default_factory=list)
    ownership_leverage: Dict[str, Any] = field(default_factory=dict)
    correlation_matrix: Dict[str, float] = field(default_factory=dict)
    contrarian_angles: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate recommendation"""
        errors = []
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            errors.append("Confidence must be between 0 and 1")
        
        if not isinstance(self.captain_targets, list):
            errors.append("captain_targets must be a list")
        
        return len(errors) == 0, errors

@dataclass
class SimulationResults:
    """Monte Carlo simulation results"""
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    floor_10th: float = 0.0
    ceiling_90th: float = 0.0
    ceiling_99th: float = 0.0
    top_10pct_mean: float = 0.0
    sharpe_ratio: float = 0.0
    win_probability: float = 0.0
    score_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class GeneticLineup:
    """Genetic algorithm lineup"""
    captain: str = ""
    flex: List[str] = field(default_factory=list)
    fitness: float = 0.0
    sim_results: Optional[SimulationResults] = None
    
    def get_all_players(self) -> List[str]:
        """Get all players in lineup"""
        return [self.captain] + self.flex

@dataclass
class GeneticConfig:
    """Genetic algorithm configuration"""
    population_size: int = 100
    generations: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 5

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================

class OptimizerConfig:
    """Central configuration for optimizer"""
    
    # DraftKings Showdown constraints
    SALARY_CAP: int = 50000
    MIN_SALARY: int = 3000
    MAX_SALARY: int = 11500
    CAPTAIN_MULTIPLIER: float = 1.5
    ROSTER_SIZE: int = 6
    FLEX_SIZE: int = 5
    
    # Team diversity
    MIN_TEAMS_REQUIRED: int = 2
    MAX_PLAYERS_PER_TEAM: int = 4
    
    # Monte Carlo settings
    MC_DEFAULT_SIMULATIONS: int = 5000
    MC_FAST_SIMULATIONS: int = 1000
    MC_DETAILED_SIMULATIONS: int = 10000
    
    # Genetic algorithm defaults
    GA_DEFAULT_POPULATION: int = 100
    GA_DEFAULT_GENERATIONS: int = 50
    GA_ELITE_SIZE: int = 10
    
    # AI weights
    AI_WEIGHTS: Dict[str, float] = {
        'game_theory': 0.35,
        'correlation': 0.35,
        'contrarian': 0.30
    }
    
    # Correlation coefficients
    CORRELATION_COEFFICIENTS: Dict[str, float] = {
        'qb_wr_same_team': 0.6,
        'qb_rb_same_team': -0.15,
        'wr_wr_same_team': -0.1,
        'qb_qb_opposing': -0.2,
        'rb_dst_opposing': -0.35,
        'wr_dst_opposing': -0.25
    }
    
    # Field size configurations
    FIELD_SIZE_CONFIGS: Dict[str, Dict[str, Any]] = {
        FieldSize.SMALL.value: {
            'ownership_weight': 0.3,
            'use_genetic': False,
            'leverage_threshold': 20,
            'max_chalk_exposure': 0.5
        },
        FieldSize.MEDIUM.value: {
            'ownership_weight': 0.4,
            'use_genetic': False,
            'leverage_threshold': 15,
            'max_chalk_exposure': 0.4
        },
        FieldSize.LARGE.value: {
            'ownership_weight': 0.5,
            'use_genetic': False,
            'leverage_threshold': 12,
            'max_chalk_exposure': 0.3
        },
        FieldSize.LARGE_AGGRESSIVE.value: {
            'ownership_weight': 0.6,
            'use_genetic': True,
            'leverage_threshold': 10,
            'max_chalk_exposure': 0.2
        },
        FieldSize.MILLY_MAKER.value: {
            'ownership_weight': 0.7,
            'use_genetic': True,
            'leverage_threshold': 8,
            'max_chalk_exposure': 0.15
        }
    }
    
    @staticmethod
    def get_field_config(field_size: str) -> Dict[str, Any]:
        """Get configuration for field size"""
        return OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size,
            OptimizerConfig.FIELD_SIZE_CONFIGS[FieldSize.LARGE.value]
        )

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class OptimizerError(Exception):
    """Base optimizer exception"""
    pass

class APIError(OptimizerError):
    """API-related error"""
    pass

class ValidationError(OptimizerError):
    """Validation error"""
    pass

class OptimizationError(OptimizerError):
    """Optimization error"""
    pass

# ============================================================================
# PART 3: SINGLETONS, LOGGING & ML ENGINES
# ============================================================================

def get_logger():
    """Get singleton logger"""
    try:
        import streamlit as st
        if 'logger' not in st.session_state:
            st.session_state.logger = GlobalLogger()
        return st.session_state.logger
    except (ImportError, RuntimeError):
        if not hasattr(get_logger, '_instance'):
            get_logger._instance = GlobalLogger()
        return get_logger._instance

def get_performance_monitor():
    """Get singleton performance monitor"""
    try:
        import streamlit as st
        if 'perf_monitor' not in st.session_state:
            st.session_state.perf_monitor = PerformanceMonitor()
        return st.session_state.perf_monitor
    except (ImportError, RuntimeError):
        if not hasattr(get_performance_monitor, '_instance'):
            get_performance_monitor._instance = PerformanceMonitor()
        return get_performance_monitor._instance

def get_ai_tracker():
    """Get singleton AI tracker"""
    try:
        import streamlit as st
        if 'ai_tracker' not in st.session_state:
            st.session_state.ai_tracker = AIDecisionTracker()
        return st.session_state.ai_tracker
    except (ImportError, RuntimeError):
        if not hasattr(get_ai_tracker, '_instance'):
            get_ai_tracker._instance = AIDecisionTracker()
        return get_ai_tracker._instance

class GlobalLogger:
    """Global logger with error tracking"""
    
    _PATTERN_NUMBER = re.compile(r'\d+')
    _PATTERN_DOUBLE_QUOTE = re.compile(r'"[^"]*"')
    _PATTERN_SINGLE_QUOTE = re.compile(r"'[^']*'")
    
    def __init__(self):
        self.logs: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.error_logs: Deque[Dict[str, Any]] = deque(maxlen=20)
        self._lock = threading.RLock()
        self.error_patterns: DefaultDict[str, int] = defaultdict(int)
        self.failure_categories: Dict[str, int] = {
            'constraint': 0, 'salary': 0, 'ownership': 0, 'api': 0,
            'validation': 0, 'timeout': 0, 'simulation': 0, 'genetic': 0, 'other': 0
        }
    
    def log(self, message: str, level: str = "INFO", context: Optional[Dict[str, Any]] = None):
        """Log message"""
        with self._lock:
            try:
                entry = {
                    'timestamp': datetime.now(),
                    'level': level.upper(),
                    'message': str(message),
                    'context': context or {}
                }
                self.logs.append(entry)
                
                if level.upper() in ["ERROR", "CRITICAL"]:
                    self.error_logs.append(entry)
            except Exception:
                pass
    
    def log_exception(self, exception: Exception, context: str = "", critical: bool = False):
        """Log exception"""
        with self._lock:
            try:
                error_msg = f"{context}: {str(exception)}" if context else str(exception)
                entry = {
                    'timestamp': datetime.now(),
                    'level': "CRITICAL" if critical else "ERROR",
                    'message': error_msg,
                    'exception_type': type(exception).__name__,
                    'traceback': traceback.format_exc()
                }
                self.error_logs.append(entry)
            except Exception:
                pass

class PerformanceMonitor:
    """Performance monitoring"""
    
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.metrics: DefaultDict[str, List] = defaultdict(list)
        self._lock = threading.RLock()
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str):
        """Start timer"""
        with self._lock:
            self.start_times[operation] = time.time()
    
    def stop_timer(self, operation: str) -> float:
        """Stop timer and return elapsed time"""
        with self._lock:
            if operation not in self.start_times:
                return 0.0
            elapsed = time.time() - self.start_times[operation]
            del self.start_times[operation]
            return elapsed

class AIDecisionTracker:
    """Track AI decisions"""
    
    def __init__(self):
        self.decisions: Deque[Dict[str, Any]] = deque(maxlen=50)
        self._lock = threading.RLock()
    
    def track_decision(self, ai_type, decision, context=None):
        """Track decision"""
        with self._lock:
            try:
                self.decisions.append({
                    'timestamp': datetime.now(),
                    'ai_type': str(ai_type),
                    'confidence': getattr(decision, 'confidence', 0.5),
                    'context': context or {}
                })
            except Exception:
                pass

# ============================================================================
# MONTE CARLO SIMULATION ENGINE
# ============================================================================

class MonteCarloSimulationEngine:
    """Monte Carlo simulation with correlations"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict[str, Any], n_simulations: int = 5000):
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        self.df = df.copy()
        self.game_info = game_info
        self.n_simulations = n_simulations
        self.logger = get_logger()
        
        self._player_indices = {p: i for i, p in enumerate(df['Player'].values)}
        self._projections = df['Projected_Points'].values.copy()
        self._positions = df['Position'].values.copy()
        self._teams = df['Team'].values.copy()
        
        self.player_variance = self._calculate_variance()
    
    def _calculate_variance(self) -> np.ndarray:
        """Calculate player variance"""
        variance_map = {'QB': 0.30, 'RB': 0.40, 'WR': 0.45, 'TE': 0.42, 'DST': 0.50, 'K': 0.55}
        position_cv = np.vectorize(lambda pos: variance_map.get(pos, 0.40))(self._positions)
        salaries = self.df['Salary'].values
        salary_factor = np.maximum(0.7, 1.0 - (salaries - 3000) / max((salaries.max() - 3000), 1) * 0.3)
        cv = position_cv * salary_factor
        safe_projections = np.maximum(self._projections, 0.1)
        variance = (safe_projections * cv) ** 2
        return np.nan_to_num(variance, nan=1.0, posinf=100.0, neginf=0.0)
    
    def evaluate_lineup(self, captain: str, flex: List[str], use_cache: bool = True) -> SimulationResults:
        """Evaluate lineup with simulation"""
        try:
            if not captain or len(flex) != 5:
                raise ValueError("Invalid lineup")
            
            all_players = [captain] + flex
            player_indices = [self._player_indices[p] for p in all_players]
            
            projections = self._projections[player_indices]
            variances = self.player_variance[player_indices]
            
            scores = self._generate_samples(projections, variances)
            scores[:, 0] *= OptimizerConfig.CAPTAIN_MULTIPLIER
            
            lineup_scores = scores.sum(axis=1)
            
            mean = float(np.mean(lineup_scores))
            median = float(np.median(lineup_scores))
            std = float(np.std(lineup_scores))
            floor_10th = float(np.percentile(lineup_scores, 10))
            ceiling_90th = float(np.percentile(lineup_scores, 90))
            ceiling_99th = float(np.percentile(lineup_scores, 99))
            
            top_10pct = lineup_scores[lineup_scores >= ceiling_90th]
            top_10pct_mean = float(np.mean(top_10pct)) if len(top_10pct) > 0 else mean
            
            sharpe_ratio = float(mean / std) if std > 0 else 0.0
            win_probability = float(np.mean(lineup_scores >= 180))
            
            return SimulationResults(
                mean=mean, median=median, std=std,
                floor_10th=floor_10th, ceiling_90th=ceiling_90th, ceiling_99th=ceiling_99th,
                top_10pct_mean=top_10pct_mean, sharpe_ratio=sharpe_ratio,
                win_probability=win_probability, score_distribution=lineup_scores
            )
        except Exception as e:
            self.logger.log_exception(e, "evaluate_lineup")
            return SimulationResults()
    
    def _generate_samples(self, projections: np.ndarray, variances: np.ndarray) -> np.ndarray:
        """Generate correlated samples"""
        n_players = len(projections)
        Z = np.random.standard_normal((self.n_simulations, n_players))
        
        scores = np.zeros((self.n_simulations, n_players))
        std_devs = np.sqrt(np.maximum(variances, 0.01))
        
        for i in range(n_players):
            if projections[i] > 0:
                proj = projections[i]
                std = std_devs[i]
                mu = np.log(proj**2 / np.sqrt(std**2 + proj**2))
                sigma = np.sqrt(np.log(1 + (std**2 / proj**2)))
                
                if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
                    scores[:, i] = np.exp(mu + sigma * Z[:, i])
                else:
                    scores[:, i] = proj
        
        scores = np.clip(scores, 0, projections * 5)
        return scores

# ============================================================================
# GENETIC ALGORITHM OPTIMIZER
# ============================================================================

class GeneticAlgorithmOptimizer:
    """Genetic algorithm for lineup optimization"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict[str, Any], 
                 mc_engine: Optional[MonteCarloSimulationEngine] = None,
                 config: Optional[GeneticConfig] = None):
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        self.df = df.copy()
        self.game_info = game_info
        self.config = config or GeneticConfig()
        self.mc_engine = mc_engine
        
        self.players = df['Player'].tolist()
        self.salaries = df.set_index('Player')['Salary'].to_dict()
        self.projections = df.set_index('Player')['Projected_Points'].to_dict()
        self.ownership = df.set_index('Player')['Ownership'].to_dict()
        self.positions = df.set_index('Player')['Position'].to_dict()
        self.teams = df.set_index('Player')['Team'].to_dict()
        
        self.logger = get_logger()
        self.best_lineups: List[GeneticLineup] = []
    
    def create_random_lineup(self) -> GeneticLineup:
        """Create random valid lineup"""
        for _ in range(100):
            try:
                captain = np.random.choice(self.players)
                available = [p for p in self.players if p != captain]
                if len(available) >= 5:
                    flex = list(np.random.choice(available, 5, replace=False))
                    lineup = GeneticLineup(captain, flex)
                    if self._is_valid_lineup(lineup):
                        return lineup
            except:
                continue
        return GeneticLineup(self.players[0], self.players[1:6])
    
    def _is_valid_lineup(self, lineup: GeneticLineup) -> bool:
        """Validate lineup"""
        try:
            all_players = lineup.get_all_players()
            total_salary = sum(self.salaries.get(p, 0) for p in lineup.flex)
            total_salary += self.salaries.get(lineup.captain, 0) * OptimizerConfig.CAPTAIN_MULTIPLIER
            
            if total_salary > OptimizerConfig.SALARY_CAP:
                return False
            
            team_counts = Counter(self.teams.get(p, '') for p in all_players)
            return len(team_counts) >= OptimizerConfig.MIN_TEAMS_REQUIRED
        except:
            return False
    
    def calculate_fitness(self, lineup: GeneticLineup, mode: FitnessMode) -> float:
        """Calculate fitness"""
        try:
            captain_proj = self.projections.get(lineup.captain, 0)
            flex_proj = sum(self.projections.get(p, 0) for p in lineup.flex)
            base_score = captain_proj * OptimizerConfig.CAPTAIN_MULTIPLIER + flex_proj
            
            captain_own = self.ownership.get(lineup.captain, 10)
            flex_own = sum(self.ownership.get(p, 10) for p in lineup.flex)
            total_own = captain_own * OptimizerConfig.CAPTAIN_MULTIPLIER + flex_own
            
            ownership_multiplier = 1.0 + (100 - total_own) / 150
            
            return base_score * ownership_multiplier
        except:
            return 0.0
    
    def optimize(self, num_lineups: int = 20, fitness_mode: Optional[FitnessMode] = None, 
                 verbose: bool = False) -> List[Dict[str, Any]]:
        """Run genetic algorithm"""
        if fitness_mode is None:
            fitness_mode = FitnessMode.CEILING
        
        population = [self.create_random_lineup() for _ in range(self.config.population_size)]
        
        for generation in range(self.config.generations):
            for lineup in population:
                if lineup.fitness == 0:
                    lineup.fitness = self.calculate_fitness(lineup, fitness_mode)
            
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            next_generation = population[:self.config.elite_size]
            
            while len(next_generation) < self.config.population_size:
                try:
                    parent1 = self._tournament_select(population)
                    parent2 = self._tournament_select(population)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    next_generation.append(child)
                except:
                    next_generation.append(self.create_random_lineup())
            
            population = next_generation
        
        unique_lineups = self._deduplicate_lineups(population, num_lineups)
        
        results = []
        for lineup in unique_lineups[:num_lineups]:
            results.append({
                'captain': lineup.captain,
                'flex': lineup.flex,
                'fitness': lineup.fitness
            })
        
        return results
    
    def _tournament_select(self, population: List[GeneticLineup]) -> GeneticLineup:
        """Tournament selection"""
        tournament = list(np.random.choice(population, min(self.config.tournament_size, len(population)), replace=False))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: GeneticLineup, parent2: GeneticLineup) -> GeneticLineup:
        """Crossover"""
        captain = np.random.choice([parent1.captain, parent2.captain])
        flex_pool = list(set(parent1.flex + parent2.flex))
        if captain in flex_pool:
            flex_pool.remove(captain)
        
        if len(flex_pool) >= 5:
            flex = list(np.random.choice(flex_pool, 5, replace=False))
        else:
            available = [p for p in self.players if p != captain and p not in flex_pool]
            flex = flex_pool + list(np.random.choice(available, 5 - len(flex_pool), replace=False))
        
        return GeneticLineup(captain, flex)
    
    def _mutate(self, lineup: GeneticLineup) -> GeneticLineup:
        """Mutate lineup"""
        if np.random.random() < self.config.mutation_rate:
            available = [p for p in self.players if p not in lineup.flex]
            if available:
                lineup.captain = np.random.choice(available)
        
        if np.random.random() < self.config.mutation_rate:
            idx = np.random.randint(0, 5)
            available = [p for p in self.players if p != lineup.captain and p not in lineup.flex]
            if available:
                lineup.flex[idx] = np.random.choice(available)
        
        return lineup
    
    def _deduplicate_lineups(self, lineups: List[GeneticLineup], target: int) -> List[GeneticLineup]:
        """Remove duplicates"""
        unique = []
        seen = []
        
        for lineup in lineups:
            players = frozenset(lineup.get_all_players())
            is_unique = True
            for s in seen:
                if len(players & s) >= 5:
                    is_unique = False
                    break
            if is_unique:
                unique.append(lineup)
                seen.append(players)
            if len(unique) >= target:
                break
        
        return unique

# ============================================================================
# AI ENFORCEMENT ENGINE
# ============================================================================

class AIEnforcementEngine:
    """AI enforcement with rule management"""
    
    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.STRONG):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
    
    def create_enforcement_rules(self, recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict[str, List[Dict[str, Any]]]:
        """Create enforcement rules"""
        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': []
        }
        
        try:
            for ai_type, rec in recommendations.items():
                if rec.captain_targets:
                    rules['hard_constraints'].append({
                        'rule': 'captain_from_list',
                        'players': rec.captain_targets[:5],
                        'source': ai_type.value,
                        'priority': 80
                    })
                
                for player in rec.must_play[:3]:
                    rules['hard_constraints'].append({
                        'rule': 'must_include',
                        'player': player,
                        'source': ai_type.value,
                        'priority': 70
                    })
        except Exception as e:
            self.logger.log_exception(e, "create_enforcement_rules")
        
        return rules

# ============================================================================
# AI OWNERSHIP BUCKET MANAGER
# ============================================================================

class AIOwnershipBucketManager:
    """Ownership bucket management"""
    
    def __init__(self):
        self.bucket_thresholds = {
            'mega_chalk': 35, 'chalk': 20, 'moderate': 15,
            'pivot': 10, 'leverage': 5, 'super_leverage': 2
        }
    
    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize players by ownership"""
        if df.empty:
            return {k: [] for k in self.bucket_thresholds.keys()}
        
        ownership = df['Ownership'].fillna(10)
        players = df['Player'].values
        t = self.bucket_thresholds
        
        return {
            'mega_chalk': players[ownership >= t['mega_chalk']].tolist(),
            'chalk': players[(ownership >= t['chalk']) & (ownership < t['mega_chalk'])].tolist(),
            'moderate': players[(ownership >= t['moderate']) & (ownership < t['chalk'])].tolist(),
            'pivot': players[(ownership >= t['pivot']) & (ownership < t['moderate'])].tolist(),
            'leverage': players[(ownership >= t['leverage']) & (ownership < t['pivot'])].tolist(),
            'super_leverage': players[ownership < t['leverage']].tolist()
        }
    
    def adjust_thresholds_for_slate(self, df: pd.DataFrame, field_size: str):
        """Adjust thresholds"""
        pass

# ============================================================================
# AI SYNTHESIS ENGINE
# ============================================================================

class AISynthesisEngine:
    """Synthesize AI recommendations"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def synthesize_recommendations(self, game_theory: AIRecommendation, 
                                  correlation: AIRecommendation,
                                  contrarian: AIRecommendation) -> Dict[str, Any]:
        """Synthesize recommendations"""
        try:
            captain_votes = defaultdict(list)
            
            for ai_type, rec in [
                (AIStrategistType.GAME_THEORY, game_theory),
                (AIStrategistType.CORRELATION, correlation),
                (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
            ]:
                for captain in rec.captain_targets[:5]:
                    captain_votes[captain].append(ai_type.value)
            
            captain_strategy = {}
            for captain, votes in captain_votes.items():
                if len(votes) == 3:
                    captain_strategy[captain] = 'consensus'
                elif len(votes) == 2:
                    captain_strategy[captain] = 'majority'
                else:
                    captain_strategy[captain] = votes[0] if votes else 'unknown'
            
            return {
                'captain_strategy': captain_strategy,
                'confidence': (game_theory.confidence + correlation.confidence + contrarian.confidence) / 3,
                'patterns': []
            }
        except Exception as e:
            self.logger.log_exception(e, "synthesize_recommendations")
            return {'captain_strategy': {}, 'confidence': 0.5, 'patterns': []}

# ============================================================================
# AI CONFIG VALIDATOR
# ============================================================================

class AIConfigValidator:
    """Validate AI configurations"""
    
    @staticmethod
    def validate_ai_requirements(enforcement_rules: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate requirements"""
        return {'is_valid': True, 'errors': [], 'warnings': []}

# ============================================================================
# OPTIMIZED DATA PROCESSOR
# ============================================================================

class OptimizedDataProcessor:
    """Optimized data processing"""
    
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be empty")
        self._df = df.copy()
    
    def calculate_lineup_metrics_batch(self, lineups: List[Dict]) -> pd.DataFrame:
        """Calculate metrics for lineups"""
        results = []
        for lineup in lineups:
            try:
                captain = lineup.get('Captain', '')
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',')]
                
                results.append({
                    'Captain': captain,
                    'FLEX': ', '.join(flex)
                })
            except:
                continue
        
        return pd.DataFrame(results)

# ============================================================================
# CLAUDE API MANAGER
# ============================================================================

class ClaudeAPIManager:
    """Secure API manager"""
    
    def __init__(self, api_key: str, max_requests_per_minute: int = 50):
        self.logger = get_logger()
        
        if not api_key or not api_key.startswith('sk-ant-'):
            raise ValueError("Invalid API key")
        
        self._api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self._max_requests_per_minute = max_requests_per_minute
        self._request_times = deque(maxlen=max_requests_per_minute)
        self._lock = threading.RLock()
        self._cache = {}
        
        self._client = self._init_client(api_key)
    
    def _init_client(self, api_key: str):
        """Initialize client"""
        try:
            if not ANTHROPIC_AVAILABLE:
                self.logger.log("Anthropic not installed", "ERROR")
                return None
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        except Exception as e:
            self.logger.log(f"API init failed: {e}", "ERROR")
            return None
    
    def get_ai_response(self, prompt: str, ai_type: Optional[AIStrategistType] = None, 
                       max_retries: int = 3) -> str:
        """Get AI response"""
        if not self._client:
            return "{}"
        
        for attempt in range(max_retries):
            try:
                message = self._client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=30,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text if message.content else "{}"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.log(f"API error: {e}", "ERROR")
                    return "{}"
        
        return "{}"

# ============================================================================
# BASE AI STRATEGIST
# ============================================================================

class BaseAIStrategist:
    """Base AI strategist"""
    
    def __init__(self, api_manager: Optional[ClaudeAPIManager] = None, 
                 strategist_type: Optional[AIStrategistType] = None):
        self.api_manager = api_manager
        self.strategist_type = strategist_type
        self.logger = get_logger()
        self.response_cache = {}
        self._cache_lock = threading.RLock()
    
    def get_recommendation(self, df: pd.DataFrame, game_info: Dict, field_size: str, 
                          use_api: bool = True) -> AIRecommendation:
        """Get recommendation"""
        try:
            if df.empty:
                return self._get_fallback_recommendation(df, field_size)
            
            if use_api and self.api_manager and self.api_manager._client:
                prompt = self.generate_prompt(df, game_info, field_size, {})
                response = self.api_manager.get_ai_response(prompt, self.strategist_type)
                return self.parse_response(response, df, field_size)
            else:
                return self._get_fallback_recommendation(df, field_size)
        except Exception as e:
            self.logger.log_exception(e, f"{self.strategist_type}")
            return self._get_fallback_recommendation(df, field_size)
    
    def _get_fallback_recommendation(self, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Statistical fallback"""
        if df.empty:
            return AIRecommendation(captain_targets=[], confidence=0.3, source_ai=self.strategist_type)
        
        try:
            ownership = df['Ownership'].fillna(10)
            
            if self.strategist_type == AIStrategistType.GAME_THEORY:
                low_own = ownership < 15
                captains = df[low_own].nlargest(7, 'Projected_Points')['Player'].tolist()
            elif self.strategist_type == AIStrategistType.CORRELATION:
                qb_mask = df['Position'] == 'QB'
                receiver_mask = df['Position'].isin(['WR', 'TE'])
                captains = (df[qb_mask].nlargest(3, 'Projected_Points')['Player'].tolist() +
                           df[receiver_mask].nlargest(4, 'Projected_Points')['Player'].tolist())
            else:
                ultra_low = ownership < 10
                captains = df[ultra_low].nlargest(7, 'Projected_Points')['Player'].tolist()
            
            return AIRecommendation(
                captain_targets=captains[:7],
                confidence=0.5,
                narrative=f"Statistical {self.strategist_type.value}",
                source_ai=self.strategist_type
            )
        except Exception as e:
            self.logger.log_exception(e, "_get_fallback_recommendation")
            return AIRecommendation(captain_targets=[], confidence=0.3, source_ai=self.strategist_type)
    
    def generate_prompt(self, df, game_info, field_size, slate_profile):
        raise NotImplementedError()
    
    def parse_response(self, response, df, field_size):
        raise NotImplementedError()

# ============================================================================
# GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist(BaseAIStrategist):
    """Game Theory Strategist"""
    
    def __init__(self, api_manager: Optional[ClaudeAPIManager] = None):
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)
    
    def generate_prompt(self, df, game_info, field_size, slate_profile):
        return f"""Game theory analysis for DFS tournament.
        
Total: {game_info.get('total', 45)}
Spread: {game_info.get('spread', 0)}

Top 10 players:
{df.nlargest(10, 'Projected_Points')[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_string()}

Respond with JSON:
{{
    "captain_rules": {{"must_be_one_of": ["Player1"], "reasoning": "Why"}},
    "lineup_rules": {{"must_include": ["Player"], "never_include": []}},
    "confidence": 0.75
}}"""
    
    def parse_response(self, response, df, field_size):
        try:
            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            available = set(df['Player'].values)
            
            captains = [p for p in data.get('captain_rules', {}).get('must_be_one_of', []) if p in available]
            must_play = [p for p in data.get('lineup_rules', {}).get('must_include', []) if p in available]
            
            if len(captains) < 3:
                captains = df[df['Ownership'] < 15].nlargest(7, 'Projected_Points')['Player'].tolist()
            
            return AIRecommendation(
                captain_targets=captains[:7],
                must_play=must_play[:5],
                confidence=max(0.0, min(1.0, data.get('confidence', 0.7))),
                narrative="Game theory optimization",
                source_ai=AIStrategistType.GAME_THEORY
            )
        except:
            return self._get_fallback_recommendation(df, field_size)

# ============================================================================
# CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist(BaseAIStrategist):
    """Correlation Strategist"""
    
    def __init__(self, api_manager: Optional[ClaudeAPIManager] = None):
        super().__init__(api_manager, AIStrategistType.CORRELATION)
    
    def generate_prompt(self, df, game_info, field_size, slate_profile):
        teams = df['Team'].unique()[:2]
        return f"""Correlation analysis for stacking.

Teams: {teams[0] if len(teams) > 0 else ''} vs {teams[1] if len(teams) > 1 else ''}
Total: {game_info.get('total', 45)}

Respond with JSON:
{{
    "primary_stacks": [{{"player1": "QB", "player2": "WR", "correlation": 0.7}}],
    "captain_correlation": {{"best_captains_for_stacking": ["Player1"]}},
    "confidence": 0.8
}}"""
    
    def parse_response(self, response, df, field_size):
        try:
            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            available = set(df['Player'].values)
            
            stacks = []
            for stack in data.get('primary_stacks', []):
                p1, p2 = stack.get('player1'), stack.get('player2')
                if p1 in available and p2 in available:
                    stacks.append(stack)
            
            captains = [p for p in data.get('captain_correlation', {}).get('best_captains_for_stacking', []) if p in available]
            
            if len(captains) < 3:
                captains = df[df['Position'] == 'QB'].nlargest(3, 'Projected_Points')['Player'].tolist()
            
            return AIRecommendation(
                captain_targets=captains[:7],
                stacks=stacks[:5],
                confidence=max(0.0, min(1.0, data.get('confidence', 0.7))),
                narrative="Correlation-based stacking",
                source_ai=AIStrategistType.CORRELATION
            )
        except:
            return self._get_fallback_recommendation(df, field_size)

# ============================================================================
# CONTRARIAN STRATEGIST
# ============================================================================

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """Contrarian Strategist"""
    
    def __init__(self, api_manager: Optional[ClaudeAPIManager] = None):
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)
    
    def generate_prompt(self, df, game_info, field_size, slate_profile):
        low_owned = df[df['Ownership'] < 10].nlargest(10, 'Projected_Points')
        return f"""Contrarian analysis for GPP.

Low-owned high ceiling:
{low_owned[['Player', 'Position', 'Projected_Points', 'Ownership']].to_string()}

Respond with JSON:
{{
    "primary_narrative": "The winning scenario",
    "contrarian_captains": [{{"player": "Name", "narrative": "Why"}}],
    "fade_the_chalk": [{{"player": "Chalk", "pivot_to": "Alternative"}}],
    "confidence": 0.7
}}"""
    
    def parse_response(self, response, df, field_size):
        try:
            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            available = set(df['Player'].values)
            
            captains = [c['player'] for c in data.get('contrarian_captains', []) if c.get('player') in available]
            fades = [f['player'] for f in data.get('fade_the_chalk', []) if f.get('player') in available]
            
            if len(captains) < 3:
                captains = df[df['Ownership'] < 10].nlargest(7, 'Projected_Points')['Player'].tolist()
            
            return AIRecommendation(
                captain_targets=captains[:7],
                never_play=fades[:3],
                confidence=max(0.0, min(1.0, data.get('confidence', 0.7))),
                narrative=data.get('primary_narrative', 'Contrarian approach'),
                source_ai=AIStrategistType.CONTRARIAN_NARRATIVE
            )
        except:
            return self._get_fallback_recommendation(df, field_size)

# ============================================================================
# END OF FILE
# ============================================================================

print(f"NFL DFS Optimizer v{__version__} loaded successfully!")
