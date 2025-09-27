# NFL GPP DUAL-AI OPTIMIZER - PART 1: CONFIGURATION & MONITORING
# Version 6.1 - COMPLETE WITH ALL METHODS AND SAFETY MEASURES

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
import json
from datetime import datetime
import streamlit as st

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class StrategyType(Enum):
    """GPP Strategy Types - AI-Driven and Legacy"""
    AI_CONSENSUS = 'ai_consensus'
    AI_MAJORITY = 'ai_majority'
    AI_CONTRARIAN = 'ai_contrarian'
    AI_CORRELATION = 'ai_correlation'
    AI_GAME_THEORY = 'ai_game_theory'
    AI_MIXED = 'ai_mixed'
    LEVERAGE = 'leverage'
    CONTRARIAN = 'contrarian'
    GAME_STACK = 'game_stack'
    STARS_SCRUBS = 'stars_scrubs'
    CORRELATION = 'correlation'
    SUPER_CONTRARIAN = 'super_contrarian'

class AIEnforcementLevel(Enum):
    """How strictly to enforce AI decisions"""
    MANDATORY = 'mandatory'
    STRONG = 'strong'
    MODERATE = 'moderate'
    SUGGESTION = 'suggestion'

class AIStrategistType(Enum):
    """Three AI Strategist Types"""
    GAME_THEORY = 'game_theory'
    CORRELATION = 'correlation'
    CONTRARIAN_NARRATIVE = 'contrarian_narrative'

# ============================================================================
# CONFIGURATION
# ============================================================================

class OptimizerConfig:
    """Core configuration for GPP optimizer - AI-as-Chef version"""
    
    # Salary and roster constraints
    SALARY_CAP = 50000
    MIN_SALARY = 3000
    MAX_PLAYERS_PER_TEAM = 5
    ROSTER_SIZE = 6
    
    # Default ownership
    DEFAULT_OWNERSHIP = 5.0
    
    # AI Configuration
    AI_ENFORCEMENT_MODE = AIEnforcementLevel.MANDATORY
    REQUIRE_AI_FOR_GENERATION = False  # Set to False for easier testing
    MIN_AI_CONFIDENCE = 0.3
    
    # Triple AI Weights
    AI_WEIGHTS = {
        AIStrategistType.GAME_THEORY: 0.35,
        AIStrategistType.CORRELATION: 0.35,
        AIStrategistType.CONTRARIAN_NARRATIVE: 0.30
    }
    
    # AI Consensus Requirements
    AI_CONSENSUS_THRESHOLDS = {
        'captain': 2,
        'must_play': 3,
        'never_play': 2,
    }
    
    # GPP Ownership targets
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (80, 120),
        'medium_field': (70, 100),
        'large_field': (60, 90),
        'milly_maker': (50, 80)
    }
    
    # Field size configurations
    FIELD_SIZE_CONFIGS = {
        'small_field': {
            'min_unique_captains': 3,
            'max_chalk_players': 4,
            'min_leverage_players': 0,
            'max_ownership_per_player': 40,
            'min_total_ownership': 80,
            'max_total_ownership': 120,
            'ai_enforcement': AIEnforcementLevel.MODERATE,
            'ai_strategy_distribution': {
                'ai_consensus': 0.4,
                'ai_majority': 0.3,
                'ai_mixed': 0.3
            }
        },
        'medium_field': {
            'min_unique_captains': 6,
            'max_chalk_players': 2,
            'min_leverage_players': 1,
            'max_ownership_per_player': 20,
            'min_total_ownership': 70,
            'max_total_ownership': 100,
            'ai_enforcement': AIEnforcementLevel.STRONG,
            'ai_strategy_distribution': {
                'ai_consensus': 0.2,
                'ai_majority': 0.3,
                'ai_contrarian': 0.2,
                'ai_correlation': 0.2,
                'ai_game_theory': 0.1
            }
        },
        'large_field': {
            'min_unique_captains': 10,
            'max_chalk_players': 1,
            'min_leverage_players': 2,
            'max_ownership_per_player': 15,
            'min_total_ownership': 60,
            'max_total_ownership': 90,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_consensus': 0.1,
                'ai_majority': 0.2,
                'ai_contrarian': 0.3,
                'ai_correlation': 0.2,
                'ai_game_theory': 0.2
            }
        },
        'milly_maker': {
            'min_unique_captains': 20,
            'max_chalk_players': 0,
            'min_leverage_players': 3,
            'max_ownership_per_player': 10,
            'min_total_ownership': 50,
            'max_total_ownership': 80,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_contrarian': 0.4,
                'ai_game_theory': 0.3,
                'ai_correlation': 0.2,
                'ai_mixed': 0.1
            }
        }
    }
    
    # AI Strategy Requirements
    AI_STRATEGY_REQUIREMENTS = {
        'ai_consensus': {
            'must_use_consensus_captain': True,
            'must_include_agreed_players': True,
            'must_avoid_fade_players': True
        },
        'ai_majority': {
            'must_use_majority_captain': True,
            'should_include_agreed_players': True
        },
        'ai_contrarian': {
            'must_use_contrarian_captain': True,
            'must_include_narrative_plays': True,
            'must_fade_chalk_narrative': True
        },
        'ai_correlation': {
            'must_use_correlation_captain': True,
            'must_include_primary_stack': True,
            'should_include_secondary_correlation': True
        },
        'ai_game_theory': {
            'must_use_leverage_captain': True,
            'must_meet_ownership_targets': True,
            'must_avoid_ownership_traps': True
        }
    }
    
    # Ownership buckets
    OWNERSHIP_BUCKETS = {
        'mega_chalk': 35,
        'chalk': 20,
        'pivot': 10,
        'leverage': 5,
        'super_leverage': 0
    }
    
    # Contest types
    FIELD_SIZES = {
        'Single Entry': 'small_field',
        '3-Max Entry': 'small_field',
        '20-Max Entry': 'medium_field',
        '150-Max Entry': 'large_field',
        'Milly Maker': 'milly_maker'
    }
    
    # Correlation thresholds
    CORRELATION_THRESHOLDS = {
        'strong_positive': 0.6,
        'moderate_positive': 0.3,
        'weak': 0.0,
        'moderate_negative': -0.3,
        'strong_negative': -0.6
    }

# ============================================================================
# AI DECISION TRACKER
# ============================================================================

class AIDecisionTracker:
    """Tracks AI decisions and enforcement throughout optimization"""
    
    def __init__(self):
        self.ai_decisions = []
        self.enforcement_stats = {
            'total_rules': 0,
            'enforced_rules': 0,
            'violated_rules': 0,
            'consensus_decisions': 0,
            'majority_decisions': 0,
            'single_ai_decisions': 0
        }
        self.ai_performance = {
            AIStrategistType.GAME_THEORY: {'suggestions': 0, 'used': 0},
            AIStrategistType.CORRELATION: {'suggestions': 0, 'used': 0},
            AIStrategistType.CONTRARIAN_NARRATIVE: {'suggestions': 0, 'used': 0}
        }
    
    def track_decision(self, decision_type: str, ai_source: str, 
                      enforced: bool, details: Dict):
        """Track an AI decision and whether it was enforced"""
        self.ai_decisions.append({
            'timestamp': datetime.now(),
            'type': decision_type,
            'source': ai_source,
            'enforced': enforced,
            'details': details
        })
        
        self.enforcement_stats['total_rules'] += 1
        if enforced:
            self.enforcement_stats['enforced_rules'] += 1
        else:
            self.enforcement_stats['violated_rules'] += 1
    
    def track_consensus(self, consensus_type: str, ais_agreeing: List[str]):
        """Track consensus between AIs"""
        if len(ais_agreeing) == 3:
            self.enforcement_stats['consensus_decisions'] += 1
        elif len(ais_agreeing) == 2:
            self.enforcement_stats['majority_decisions'] += 1
        else:
            self.enforcement_stats['single_ai_decisions'] += 1
    
    def get_enforcement_rate(self) -> float:
        """Get the rate of AI decision enforcement"""
        if self.enforcement_stats['total_rules'] == 0:
            return 0.0
        return self.enforcement_stats['enforced_rules'] / self.enforcement_stats['total_rules']
    
    def get_summary(self) -> Dict:
        """Get summary of AI decision tracking"""
        return {
            'enforcement_rate': self.get_enforcement_rate(),
            'stats': self.enforcement_stats,
            'ai_performance': self.ai_performance,
            'recent_decisions': self.ai_decisions[-10:] if self.ai_decisions else []
        }

# ============================================================================
# GLOBAL LOGGER - COMPLETE WITH ALL METHODS
# ============================================================================

class GlobalLogger:
    """Enhanced logger with complete method set and safety measures"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """Initialize logger state"""
        self.entries = []
        self.ai_tracker = AIDecisionTracker()
        self.verbose = False
        self.log_to_file = False
    
    def log(self, message: str, level: str = "INFO"):
        """Basic logging method"""
        try:
            entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': str(message)
            }
            self.entries.append(entry)
            
            if self.verbose or level in ["ERROR", "WARNING"]:
                timestamp = entry['timestamp'].strftime('%H:%M:%S')
                print(f"[{timestamp}] {message}")
            
            if self.log_to_file:
                self._write_to_file(entry)
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_ai_decision(self, decision_type: str, ai_source: str, 
                       enforced: bool, details: Dict):
        """Log an AI decision"""
        try:
            self.ai_tracker.track_decision(decision_type, ai_source, enforced, details)
            message = f"AI Decision [{ai_source}]: {decision_type} - {'ENFORCED' if enforced else 'VIOLATED'}"
            self.log(message, "AI_DECISION")
        except Exception as e:
            self.log(f"Error logging AI decision: {e}", "ERROR")
    
    def log_ai_consensus(self, consensus_type: str, ais_agreeing: List[str], 
                        decision: str):
        """Log AI consensus"""
        try:
            self.ai_tracker.track_consensus(consensus_type, ais_agreeing)
            message = f"AI Consensus ({len(ais_agreeing)}/3): {decision}"
            self.log(message, "AI_CONSENSUS")
        except Exception as e:
            self.log(f"Error logging AI consensus: {e}", "ERROR")
    
    def log_optimization_start(self, num_lineups: int, field_size: str, 
                              settings: Dict):
        """Log optimization start"""
        try:
            self.log(f"Starting optimization: {num_lineups} lineups for {field_size}", "INFO")
            self.log(f"Settings: {settings}", "DEBUG")
        except Exception as e:
            self.log(f"Error in log_optimization_start: {e}", "ERROR")
    
    def log_optimization_end(self, lineups_generated: int, total_time: float):
        """Log optimization completion - CRITICAL METHOD"""
        try:
            self.log(f"Optimization complete: {lineups_generated} lineups in {total_time:.2f}s", "INFO")
            if lineups_generated > 0:
                avg_time = total_time / lineups_generated
                self.log(f"Average time per lineup: {avg_time:.3f}s", "DEBUG")
        except Exception as e:
            print(f"Error in log_optimization_end: {e}")
    
    def log_lineup_generation(self, strategy: str, lineup_num: int, 
                             status: str, ai_rules_enforced: int = 0):
        """Log lineup generation"""
        try:
            message = f"Lineup {lineup_num} ({strategy}): {status}"
            if ai_rules_enforced > 0:
                message += f" - {ai_rules_enforced} AI rules enforced"
            self.log(message, "DEBUG" if status == "SUCCESS" else "WARNING")
        except Exception as e:
            self.log(f"Error in log_lineup_generation: {e}", "ERROR")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log an exception"""
        try:
            message = f"Exception in {context}: {str(exception)}"
            self.log(message, "ERROR")
        except Exception as e:
            print(f"Error logging exception: {e}")
    
    def get_ai_summary(self) -> Dict:
        """Get AI tracking summary"""
        try:
            return self.ai_tracker.get_summary()
        except:
            return {
                'enforcement_rate': 0.0,
                'stats': {
                    'total_rules': 0,
                    'enforced_rules': 0,
                    'violated_rules': 0,
                    'consensus_decisions': 0
                },
                'ai_performance': {},
                'recent_decisions': []
            }
    
    def display_ai_enforcement(self):
        """Display AI enforcement statistics in Streamlit"""
        try:
            summary = self.get_ai_summary()
            
            st.markdown("### 🤖 AI Decision Enforcement")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rate = summary['enforcement_rate'] * 100
                st.metric("Enforcement Rate", f"{rate:.1f}%")
            
            with col2:
                stats = summary['stats']
                st.metric("Rules Enforced", 
                         f"{stats['enforced_rules']}/{stats['total_rules']}")
            
            with col3:
                st.metric("Consensus Decisions", stats.get('consensus_decisions', 0))
            
            # Show AI performance
            if summary.get('ai_performance'):
                st.markdown("#### AI Performance")
                for ai_type, perf in summary['ai_performance'].items():
                    if perf.get('suggestions', 0) > 0:
                        usage_rate = (perf.get('used', 0) / perf['suggestions']) * 100
                        st.write(f"**{ai_type.value}**: {usage_rate:.1f}% usage rate")
        except Exception as e:
            st.error(f"Error displaying AI enforcement: {e}")
    
    def display_log_summary(self):
        """Display log summary in Streamlit"""
        try:
            st.markdown("### 📋 Log Summary")
            
            # Count by level
            level_counts = {}
            for entry in self.entries:
                level = entry.get('level', 'UNKNOWN')
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Display counts
            if level_counts:
                cols = st.columns(len(level_counts))
                for i, (level, count) in enumerate(level_counts.items()):
                    cols[i].metric(level, count)
            
            # Show recent logs
            if self.entries:
                st.markdown("#### Recent Entries")
                for entry in self.entries[-10:]:
                    timestamp = entry['timestamp'].strftime('%H:%M:%S')
                    level_color = {
                        'ERROR': '🔴',
                        'WARNING': '🟡',
                        'INFO': '🟢',
                        'DEBUG': '🔵',
                        'AI_DECISION': '🤖',
                        'AI_CONSENSUS': '🤝'
                    }.get(entry.get('level', 'UNKNOWN'), '⚪')
                    st.text(f"{level_color} [{timestamp}] {entry.get('level', 'UNKNOWN')}: {entry.get('message', '')}")
        except Exception as e:
            st.error(f"Error displaying log summary: {e}")
    
    def export_logs(self) -> str:
        """Export logs as string"""
        try:
            output = "=== OPTIMIZATION LOG ===\n\n"
            for entry in self.entries:
                timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                output += f"[{timestamp}] {entry.get('level', 'UNKNOWN')}: {entry.get('message', '')}\n"
            
            output += "\n=== AI DECISION SUMMARY ===\n"
            summary = self.get_ai_summary()
            output += f"Enforcement Rate: {summary['enforcement_rate']*100:.1f}%\n"
            output += f"Total AI Rules: {summary['stats']['total_rules']}\n"
            output += f"Enforced: {summary['stats']['enforced_rules']}\n"
            output += f"Violated: {summary['stats']['violated_rules']}\n"
            
            return output
        except Exception as e:
            return f"Error exporting logs: {e}"
    
    def _write_to_file(self, entry: Dict):
        """Write log entry to file"""
        try:
            with open('optimizer_log.txt', 'a') as f:
                timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {entry.get('level', 'UNKNOWN')}: {entry.get('message', '')}\n")
        except:
            pass  # Fail silently

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Monitor performance with AI-specific metrics"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """Initialize performance metrics"""
        self.timers = {}
        self.counters = {}
        self.ai_metrics = {
            'ai_api_calls': 0,
            'ai_response_time': [],
            'ai_cache_hits': 0,
            'ai_enforcement_time': []
        }
    
    def start_timer(self, name: str):
        """Start a timer"""
        try:
            if name not in self.timers:
                self.timers[name] = {}
            self.timers[name]['start'] = datetime.now()
        except Exception as e:
            print(f"Error starting timer {name}: {e}")
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time"""
        try:
            if name in self.timers and 'start' in self.timers[name]:
                elapsed = (datetime.now() - self.timers[name]['start']).total_seconds()
                
                if 'total' not in self.timers[name]:
                    self.timers[name]['total'] = 0
                    self.timers[name]['count'] = 0
                    self.timers[name]['average'] = 0
                
                self.timers[name]['total'] += elapsed
                self.timers[name]['count'] += 1
                self.timers[name]['average'] = self.timers[name]['total'] / self.timers[name]['count']
                
                # Track AI-specific timings
                if 'ai' in name.lower():
                    self.ai_metrics['ai_response_time'].append(elapsed)
                
                return elapsed
            return 0.0
        except Exception as e:
            print(f"Error stopping timer {name}: {e}")
            return 0.0
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        try:
            if name not in self.counters:
                self.counters[name] = 0
            self.counters[name] += value
            
            # Track AI-specific counters
            if 'ai' in name.lower():
                if 'cache' in name.lower():
                    self.ai_metrics['ai_cache_hits'] += value
                elif 'api' in name.lower():
                    self.ai_metrics['ai_api_calls'] += value
        except Exception as e:
            print(f"Error incrementing counter {name}: {e}")
    
    def get_metrics(self) -> Dict:
        """Get all metrics"""
        try:
            avg_ai_response = (
                np.mean(self.ai_metrics['ai_response_time']) 
                if self.ai_metrics['ai_response_time'] else 0
            )
            
            cache_hit_rate = 0
            total_calls = self.ai_metrics['ai_api_calls'] + self.ai_metrics['ai_cache_hits']
            if total_calls > 0:
                cache_hit_rate = self.ai_metrics['ai_cache_hits'] / total_calls
            
            return {
                'timers': self.timers,
                'counters': self.counters,
                'ai_metrics': {
                    'api_calls': self.ai_metrics['ai_api_calls'],
                    'cache_hits': self.ai_metrics['ai_cache_hits'],
                    'avg_response_time': avg_ai_response,
                    'cache_hit_rate': cache_hit_rate
                }
            }
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return {'timers': {}, 'counters': {}, 'ai_metrics': {}}
    
    def display_metrics(self):
        """Display performance metrics in Streamlit"""
        try:
            metrics = self.get_metrics()
            
            st.markdown("### ⚡ Performance Metrics")
            
            # AI Metrics
            if metrics['ai_metrics']['api_calls'] > 0:
                st.markdown("#### AI Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("API Calls", metrics['ai_metrics']['api_calls'])
                with col2:
                    st.metric("Avg Response", f"{metrics['ai_metrics']['avg_response_time']:.2f}s")
                with col3:
                    st.metric("Cache Hit Rate", f"{metrics['ai_metrics']['cache_hit_rate']*100:.1f}%")
            
            # General Metrics
            if metrics['timers']:
                st.markdown("#### Optimization Performance")
                for name, stats in metrics['timers'].items():
                    if 'average' in stats:
                        st.write(f"**{name}**: {stats['average']:.3f}s avg ({stats['count']} ops)")
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")

# ============================================================================
# SINGLETON GETTERS WITH SAFETY
# ============================================================================

def get_logger() -> GlobalLogger:
    """Get the global logger instance with safety check"""
    try:
        return GlobalLogger()
    except Exception as e:
        print(f"Error creating logger: {e}")
        # Return a minimal logger if creation fails
        class MinimalLogger:
            def __getattr__(self, name):
                def method(*args, **kwargs):
                    print(f"Logger.{name} called with args: {args[:2] if args else 'none'}")
                return method
        return MinimalLogger()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance with safety check"""
    try:
        return PerformanceMonitor()
    except Exception as e:
        print(f"Error creating performance monitor: {e}")
        # Return a minimal monitor if creation fails
        class MinimalMonitor:
            def __getattr__(self, name):
                def method(*args, **kwargs):
                    if name == 'stop_timer':
                        return 0.0
                    return None
                return method
        return MinimalMonitor()

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AIRecommendation:
    """Enhanced AI recommendation with enforcement rules"""
    captain_targets: List[str]
    must_play: List[str]
    never_play: List[str]
    stacks: List[Dict]
    key_insights: List[str]
    confidence: float
    enforcement_rules: List[Dict]
    narrative: str
    source_ai: AIStrategistType
    
    # Optional fields
    contrarian_angles: Optional[List[str]] = None
    correlation_matrix: Optional[Dict] = None
    ownership_leverage: Optional[Dict] = None
    boosts: Optional[List[str]] = None
    fades: Optional[List[str]] = None
    
    def get_hard_constraints(self) -> List[str]:
        """Get constraints that MUST be enforced"""
        return [r['constraint'] for r in self.enforcement_rules if r.get('type') == 'hard']
    
    def get_soft_constraints(self) -> List[Tuple[str, float]]:
        """Get constraints with weights"""
        return [(r['constraint'], r.get('weight', 1.0)) 
                for r in self.enforcement_rules if r.get('type') == 'soft']

# ============================================================================
# STUB CLASSES FOR OTHER PARTS
# ============================================================================

class GPPCaptainPivotGenerator:
    """Stub for captain pivot generator"""
    def __init__(self):
        self.logger = get_logger()

class GPPCorrelationEngine:
    """Stub for correlation engine"""
    def __init__(self):
        self.logger = get_logger()

class GPPTournamentSimulator:
    """Stub for tournament simulator"""
    def __init__(self):
        self.logger = get_logger()

class OwnershipBucketManager:
    """Stub for ownership bucket manager"""
    def __init__(self):
        self.logger = get_logger()
    
    @staticmethod
    def get_bucket(ownership: float) -> str:
        """Get ownership bucket for a given ownership percentage"""
        if ownership >= 35:
            return 'mega_chalk'
        elif ownership >= 20:
            return 'chalk'
        elif ownership >= 10:
            return 'pivot'
        elif ownership >= 5:
            return 'leverage'
        else:
            return 'super_leverage'

class ConfigValidator:
    """Stub for config validator"""
    @staticmethod
    def validate_field_config(field_size: str, num_lineups: int) -> Dict:
        """Validate and return field configuration"""
        return OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size, 
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )
    
    @staticmethod
    def validate_player_pool(df: pd.DataFrame, field_size: str) -> Dict:
        """Validate player pool"""
        return {'is_valid': True, 'errors': [], 'warnings': []}
    
    @staticmethod
    def get_strategy_distribution(field_size: str, num_lineups: int) -> Dict:
        """Get strategy distribution for lineups"""
        # Simple fallback distribution
        return {StrategyType.LEVERAGE: num_lineups}

# NFL GPP DUAL-AI OPTIMIZER - PART 2: CORE COMPONENTS (AI-AS-CHEF VERSION)
# Version 6.0 - AI-Driven Core Components with Enforcement

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import streamlit as st
import json

# ============================================================================
# AI ENFORCEMENT ENGINE
# ============================================================================

class AIEnforcementEngine:
    """Core engine for enforcing AI decisions throughout optimization"""
    
    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.MANDATORY):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
        self.enforcement_history = []
    
    def create_enforcement_rules(self, ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Convert AI recommendations into enforceable optimization rules"""
        
        rules = {
            'hard_constraints': [],  # Must be satisfied
            'soft_constraints': [],  # Should be satisfied with weights
            'objective_modifiers': {},  # Modifications to objective function
            'variable_locks': {},  # Specific variable assignments
        }
        
        # Find consensus among AIs
        consensus = self._find_consensus(ai_recommendations)
        
        # Create rules based on consensus and enforcement level
        if self.enforcement_level == AIEnforcementLevel.MANDATORY:
            # All consensus decisions become hard constraints
            rules['hard_constraints'].extend(self._create_hard_constraints(consensus['must_play'], 'include'))
            rules['hard_constraints'].extend(self._create_hard_constraints(consensus['never_play'], 'exclude'))
            rules['variable_locks']['captain'] = consensus['consensus_captains']
            
        elif self.enforcement_level == AIEnforcementLevel.STRONG:
            # Consensus becomes hard, majority becomes soft
            rules['hard_constraints'].extend(self._create_hard_constraints(consensus['must_play'], 'include'))
            rules['soft_constraints'].extend(self._create_soft_constraints(consensus['should_play'], 'include', 0.8))
            
        # Add AI-specific modifiers
        for ai_type, rec in ai_recommendations.items():
            if rec.confidence > OptimizerConfig.MIN_AI_CONFIDENCE:
                rules['objective_modifiers'].update(self._create_objective_modifiers(rec))
        
        self.logger.log_ai_decision(
            "enforcement_rules_created",
            "AIEnforcementEngine",
            True,
            {'num_hard': len(rules['hard_constraints']), 
             'num_soft': len(rules['soft_constraints'])}
        )
        
        return rules
    
    def _find_consensus(self, ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Find consensus among AI recommendations"""
        
        consensus = {
            'consensus_captains': [],  # All 3 AIs agree
            'majority_captains': [],   # 2 of 3 agree
            'must_play': [],           # All agree these must play
            'should_play': [],         # 2 of 3 agree these should play
            'never_play': [],          # 2+ agree these should be avoided
        }
        
        # Aggregate captain recommendations
        captain_votes = {}
        for ai_type, rec in ai_recommendations.items():
            for captain in rec.captain_targets:
                if captain not in captain_votes:
                    captain_votes[captain] = []
                captain_votes[captain].append(ai_type)
        
        # Classify by agreement level
        for captain, voters in captain_votes.items():
            if len(voters) == 3:
                consensus['consensus_captains'].append(captain)
                self.logger.log_ai_consensus("captain", voters, f"Consensus captain: {captain}")
            elif len(voters) == 2:
                consensus['majority_captains'].append(captain)
                self.logger.log_ai_consensus("captain", voters, f"Majority captain: {captain}")
        
        # Aggregate must_play and never_play
        must_play_votes = {}
        never_play_votes = {}
        
        for ai_type, rec in ai_recommendations.items():
            for player in rec.must_play:
                if player not in must_play_votes:
                    must_play_votes[player] = []
                must_play_votes[player].append(ai_type)
            
            for player in rec.never_play:
                if player not in never_play_votes:
                    never_play_votes[player] = []
                never_play_votes[player].append(ai_type)
        
        # Classify must_play by agreement
        for player, voters in must_play_votes.items():
            if len(voters) == 3:
                consensus['must_play'].append(player)
            elif len(voters) == 2:
                consensus['should_play'].append(player)
        
        # Classify never_play (2+ agreement for fades)
        for player, voters in never_play_votes.items():
            if len(voters) >= 2:
                consensus['never_play'].append(player)
        
        return consensus
    
    def _create_hard_constraints(self, players: List[str], constraint_type: str) -> List[Dict]:
        """Create hard constraints for optimization"""
        constraints = []
        
        for player in players:
            if constraint_type == 'include':
                constraints.append({
                    'type': 'hard',
                    'player': player,
                    'rule': 'must_include',
                    'description': f"AI consensus: must include {player}"
                })
            elif constraint_type == 'exclude':
                constraints.append({
                    'type': 'hard',
                    'player': player,
                    'rule': 'must_exclude',
                    'description': f"AI consensus: must exclude {player}"
                })
        
        return constraints
    
    def _create_soft_constraints(self, players: List[str], constraint_type: str, weight: float) -> List[Dict]:
        """Create weighted soft constraints"""
        constraints = []
        
        for player in players:
            constraints.append({
                'type': 'soft',
                'player': player,
                'rule': f"should_{constraint_type}",
                'weight': weight,
                'description': f"AI majority: should {constraint_type} {player}"
            })
        
        return constraints
    
    def _create_objective_modifiers(self, recommendation: AIRecommendation) -> Dict:
        """Create objective function modifiers based on AI recommendation"""
        modifiers = {}
        
        # Boost recommended players
        for player in recommendation.captain_targets:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 + recommendation.confidence * 0.3)
        
        for player in recommendation.must_play:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 + recommendation.confidence * 0.2)
        
        # Penalize fade targets
        for player in recommendation.never_play:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 - recommendation.confidence * 0.3)
        
        return modifiers
    
    def validate_lineup_against_ai(self, lineup: Dict, ai_rules: Dict) -> Tuple[bool, List[str]]:
        """Validate that a lineup satisfies AI requirements"""
        violations = []
        
        lineup_players = [lineup.get('Captain')] + lineup.get('FLEX', [])
        
        # Check hard constraints
        for constraint in ai_rules.get('hard_constraints', []):
            player = constraint['player']
            rule = constraint['rule']
            
            if rule == 'must_include' and player not in lineup_players:
                violations.append(f"Missing required player: {player}")
            elif rule == 'must_exclude' and player in lineup_players:
                violations.append(f"Included banned player: {player}")
        
        # Check captain requirements
        required_captains = ai_rules.get('variable_locks', {}).get('captain', [])
        if required_captains and lineup.get('Captain') not in required_captains:
            violations.append(f"Captain {lineup.get('Captain')} not in AI requirements: {required_captains}")
        
        is_valid = len(violations) == 0
        
        # Track enforcement
        self.enforcement_history.append({
            'lineup': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': violations
        })
        
        return is_valid, violations

# ============================================================================
# AI-DRIVEN OWNERSHIP BUCKET MANAGER
# ============================================================================

class AIOwnershipBucketManager:
    """Enhanced bucket manager that respects AI decisions"""
    
    def __init__(self, ai_enforcement_engine: Optional[AIEnforcementEngine] = None):
        self.thresholds = OptimizerConfig.OWNERSHIP_BUCKETS
        self.ai_engine = ai_enforcement_engine
        self.logger = get_logger()
    
    def categorize_players_with_ai(self, df: pd.DataFrame, 
                                   ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Categorize players considering AI input"""
        
        # Standard categorization first
        buckets = self.categorize_players(df)
        
        if self.ai_engine and ai_recommendations:
            # Apply AI overrides
            consensus = self.ai_engine._find_consensus(ai_recommendations)
            
            # Move consensus captains to their own category
            buckets['ai_consensus_captains'] = consensus['consensus_captains']
            
            # Remove them from other buckets
            for captain in consensus['consensus_captains']:
                for bucket_name, players in buckets.items():
                    if bucket_name != 'ai_consensus_captains' and captain in players:
                        players.remove(captain)
            
            # Create AI leverage category
            buckets['ai_leverage'] = []
            for ai_type, rec in ai_recommendations.items():
                if ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
                    # Contrarian AI's unique picks are leverage
                    buckets['ai_leverage'].extend([
                        p for p in rec.captain_targets 
                        if p not in consensus['consensus_captains']
                    ])
            
            self.logger.log(f"AI-enhanced buckets: {len(buckets['ai_consensus_captains'])} consensus captains", "DEBUG")
        
        return buckets
    
    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Standard categorization by ownership"""
        buckets = {
            'mega_chalk': [],
            'chalk': [],
            'pivot': [],
            'leverage': [],
            'super_leverage': []
        }
        
        for _, row in df.iterrows():
            player = row['Player']
            ownership = row['Ownership']
            
            if ownership >= self.thresholds['mega_chalk']:
                buckets['mega_chalk'].append(player)
            elif ownership >= self.thresholds['chalk']:
                buckets['chalk'].append(player)
            elif ownership >= self.thresholds['pivot']:
                buckets['pivot'].append(player)
            elif ownership >= self.thresholds['leverage']:
                buckets['leverage'].append(player)
            else:
                buckets['super_leverage'].append(player)
        
        return buckets
    
    def calculate_ai_leverage_score(self, lineup_players: List[str], df: pd.DataFrame,
                                   ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> float:
        """Calculate leverage score with AI weighting"""
        
        base_score = self.calculate_gpp_leverage(lineup_players, df)
        
        if not ai_recommendations:
            return base_score
        
        # Bonus for using AI contrarian picks
        ai_bonus = 0
        for player in lineup_players:
            for ai_type, rec in ai_recommendations.items():
                if ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
                    if player in rec.captain_targets:
                        ai_bonus += 3
                    elif player in rec.must_play:
                        ai_bonus += 2
                elif player in rec.captain_targets:
                    ai_bonus += 1
        
        return base_score + ai_bonus
    
    def calculate_gpp_leverage(self, lineup_players: List[str], df: pd.DataFrame) -> float:
        """Calculate standard GPP leverage score"""
        score = 0
        
        for player in lineup_players:
            if player in df['Player'].values:
                ownership = df[df['Player'] == player]['Ownership'].values[0]
                
                if ownership < 5:
                    score += 3
                elif ownership < 10:
                    score += 2
                elif ownership < 15:
                    score += 1
                elif ownership > 30:
                    score -= 2
                elif ownership > 20:
                    score -= 1
        
        return score
    
    def get_gpp_summary(self, lineup_players: List[str], df: pd.DataFrame, 
                       field_size: str, ai_enforced: bool = False) -> str:
        """Get GPP summary with AI enforcement indicator"""
        
        ownership_counts = {'<5%': 0, '5-10%': 0, '10-20%': 0, '20-30%': 0, '30%+': 0}
        
        for player in lineup_players:
            if player in df['Player'].values:
                ownership = df[df['Player'] == player]['Ownership'].values[0]
                if ownership < 5:
                    ownership_counts['<5%'] += 1
                elif ownership < 10:
                    ownership_counts['5-10%'] += 1
                elif ownership < 20:
                    ownership_counts['10-20%'] += 1
                elif ownership < 30:
                    ownership_counts['20-30%'] += 1
                else:
                    ownership_counts['30%+'] += 1
        
        summary = ' | '.join([f"{k}:{v}" for k, v in ownership_counts.items() if v > 0])
        
        if ai_enforced:
            summary = "🤖 AI-ENFORCED | " + summary
        
        return summary

# ============================================================================
# AI-DRIVEN CONFIG VALIDATOR
# ============================================================================

class AIConfigValidator:
    """Enhanced validator that ensures AI requirements are feasible"""
    
    @staticmethod
    def validate_ai_requirements(ai_rules: Dict, player_pool: pd.DataFrame) -> Dict:
        """Validate that AI requirements can be satisfied"""
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'adjustments': []
        }
        
        available_players = set(player_pool['Player'].tolist())
        
        # Check hard constraints feasibility
        must_include = []
        must_exclude = []
        
        for constraint in ai_rules.get('hard_constraints', []):
            player = constraint['player']
            rule = constraint['rule']
            
            if player not in available_players:
                validation['errors'].append(f"AI required player '{player}' not in pool")
                validation['is_valid'] = False
                continue
            
            if rule == 'must_include':
                must_include.append(player)
            elif rule == 'must_exclude':
                must_exclude.append(player)
        
        # Check if we have enough players after exclusions
        available_after_exclusions = len(available_players) - len(must_exclude)
        required_spots = 6  # 1 CPT + 5 FLEX
        
        if len(must_include) > required_spots:
            validation['errors'].append(f"AI requires {len(must_include)} players but lineup has {required_spots} spots")
            validation['is_valid'] = False
        
        if available_after_exclusions < required_spots:
            validation['errors'].append(f"Not enough players after AI exclusions")
            validation['is_valid'] = False
        
        # Check captain requirements
        required_captains = ai_rules.get('variable_locks', {}).get('captain', [])
        if required_captains:
            valid_captains = [c for c in required_captains if c in available_players]
            if not valid_captains:
                validation['errors'].append("No valid AI captains in player pool")
                validation['is_valid'] = False
            elif len(valid_captains) < len(required_captains):
                validation['warnings'].append(f"Only {len(valid_captains)} of {len(required_captains)} AI captains available")
                ai_rules['variable_locks']['captain'] = valid_captains
                validation['adjustments'].append("Reduced captain pool to available players")
        
        # Check salary feasibility with requirements
        if must_include:
            must_include_df = player_pool[player_pool['Player'].isin(must_include)]
            min_required_salary = must_include_df['Salary'].sum()
            
            if min_required_salary > OptimizerConfig.SALARY_CAP:
                validation['errors'].append(f"AI required players cost ${min_required_salary} (exceeds cap)")
                validation['is_valid'] = False
        
        return validation
    
    @staticmethod
    def validate_field_config_with_ai(field_size: str, num_lineups: int, 
                                      ai_recommendations: Dict) -> Dict:
        """Validate configuration considering AI strategy distribution"""
        
        base_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        
        # Get AI strategy distribution
        ai_distribution = base_config.get('ai_strategy_distribution', {})
        
        # Adjust based on AI consensus
        if ai_recommendations:
            consensus_count = len([1 for rec in ai_recommendations.values() if rec.confidence > 0.8])
            
            if consensus_count == 3:
                # Strong consensus - increase consensus strategy allocation
                ai_distribution['ai_consensus'] = min(0.5, ai_distribution.get('ai_consensus', 0.2) * 2)
            
        return {
            **base_config,
            'ai_strategy_distribution': ai_distribution,
            'ai_enforcement': base_config.get('ai_enforcement', AIEnforcementLevel.MANDATORY)
        }
    
    @staticmethod
    def get_ai_strategy_distribution(field_size: str, num_lineups: int, 
                                     ai_consensus_level: str = 'mixed') -> Dict[StrategyType, int]:
        """Get lineup distribution across AI strategies"""
        
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        distribution = config.get('ai_strategy_distribution', {})
        
        # Convert percentages to lineup counts
        strategy_counts = {}
        remaining = num_lineups
        
        # Prioritize by consensus level
        if ai_consensus_level == 'high':
            # More consensus lineups
            priority = ['ai_consensus', 'ai_majority', 'ai_mixed', 'ai_contrarian', 'ai_correlation', 'ai_game_theory']
        elif ai_consensus_level == 'low':
            # More diverse lineups
            priority = ['ai_contrarian', 'ai_game_theory', 'ai_correlation', 'ai_mixed', 'ai_majority', 'ai_consensus']
        else:
            # Balanced
            priority = list(distribution.keys())
        
        for strategy in priority:
            if strategy in distribution and remaining > 0:
                count = int(distribution[strategy] * num_lineups)
                count = min(count, remaining)
                if count > 0:
                    # Convert string to StrategyType enum
                    try:
                        strategy_enum = StrategyType[strategy.upper()]
                        strategy_counts[strategy_enum] = count
                        remaining -= count
                    except KeyError:
                        # New AI strategies not in enum yet, use string
                        strategy_counts[strategy] = count
                        remaining -= count
        
        # Distribute remaining lineups
        if remaining > 0:
            # Add to first strategy
            first_strategy = list(strategy_counts.keys())[0] if strategy_counts else StrategyType.AI_MIXED
            if first_strategy in strategy_counts:
                strategy_counts[first_strategy] += remaining
            else:
                strategy_counts[first_strategy] = remaining
        
        return strategy_counts
    
    @staticmethod
    def validate_player_pool_for_ai(df: pd.DataFrame, field_size: str, 
                                    ai_requirements: Dict) -> Dict:
        """Validate player pool can satisfy AI requirements"""
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'ai_feasibility': {}
        }
        
        # Standard validation
        base_validation = ConfigValidator.validate_player_pool(df, field_size)
        validation['errors'].extend(base_validation.get('errors', []))
        validation['warnings'].extend(base_validation.get('warnings', []))
        
        # Check AI-specific requirements
        if ai_requirements:
            ai_validation = AIConfigValidator.validate_ai_requirements(ai_requirements, df)
            validation['errors'].extend(ai_validation.get('errors', []))
            validation['warnings'].extend(ai_validation.get('warnings', []))
            validation['ai_feasibility'] = ai_validation
            
            if not ai_validation['is_valid']:
                validation['is_valid'] = False
        
        return validation

# ============================================================================
# STANDARD CONFIG VALIDATOR (Legacy compatibility)
# ============================================================================

class ConfigValidator:
    """Standard configuration validator for backward compatibility"""
    
    @staticmethod
    def validate_field_config(field_size: str, num_lineups: int) -> Dict:
        """Validate and return field configuration"""
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size)
        
        if not config:
            st.warning(f"Unknown field size '{field_size}', using large_field defaults")
            config = OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        
        # Adjust unique captains if needed
        if num_lineups < config['min_unique_captains']:
            config['min_unique_captains'] = max(1, num_lineups // 2)
        
        return config
    
    @staticmethod
    def validate_player_pool(df: pd.DataFrame, field_size: str) -> Dict:
        """Validate player pool for optimization"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check minimum players
        if len(df) < 12:
            validation['errors'].append(f"Only {len(df)} players available (minimum 12 recommended)")
            if len(df) < 6:
                validation['is_valid'] = False
        
        # Check position distribution
        positions = df['Position'].value_counts()
        if 'QB' not in positions or positions.get('QB', 0) == 0:
            validation['warnings'].append("No QB in player pool")
        
        # Check team distribution
        teams = df['Team'].unique()
        if len(teams) != 2:
            validation['errors'].append(f"Expected 2 teams, found {len(teams)}")
            validation['is_valid'] = False
        
        # Check salary distribution
        avg_salary = df['Salary'].mean()
        if avg_salary < 5000:
            validation['warnings'].append(f"Low average salary (${avg_salary:.0f})")
        
        return validation
    
    @staticmethod
    def get_strategy_distribution(field_size: str, num_lineups: int) -> Dict:
        """Get standard strategy distribution (for fallback)"""
        
        # Use AI strategy distribution if available
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        if 'ai_strategy_distribution' in config:
            return AIConfigValidator.get_ai_strategy_distribution(field_size, num_lineups)
        
        # Fallback to legacy distribution
        if field_size == 'milly_maker':
            return {
                StrategyType.CONTRARIAN: num_lineups // 2,
                StrategyType.LEVERAGE: num_lineups - (num_lineups // 2)
            }
        else:
            return {
                StrategyType.LEVERAGE: num_lineups
            }

# ============================================================================
# AI SYNTHESIS ENGINE
# ============================================================================

class AISynthesisEngine:
    """Synthesizes recommendations from multiple AI strategists"""
    
    def __init__(self):
        self.logger = get_logger()
        self.synthesis_history = []
    
    def synthesize_recommendations(self, 
                                  game_theory_rec: AIRecommendation,
                                  correlation_rec: AIRecommendation,
                                  contrarian_rec: AIRecommendation) -> Dict:
        """Combine three AI perspectives into unified strategy"""
        
        synthesis = {
            'captain_strategy': {},
            'player_rankings': {},
            'stacking_rules': [],
            'avoidance_rules': [],
            'narrative': "",
            'confidence': 0.0,
            'enforcement_rules': []
        }
        
        # Combine captain recommendations with weighted voting
        all_captains = set()
        captain_votes = {}
        
        for rec, weight in [(game_theory_rec, 0.35), (correlation_rec, 0.35), (contrarian_rec, 0.30)]:
            for captain in rec.captain_targets:
                all_captains.add(captain)
                if captain not in captain_votes:
                    captain_votes[captain] = {'score': 0, 'voters': []}
                captain_votes[captain]['score'] += weight * rec.confidence
                captain_votes[captain]['voters'].append(rec.source_ai)
        
        # Classify captains by consensus level
        for captain, data in captain_votes.items():
            if len(data['voters']) == 3:
                synthesis['captain_strategy'][captain] = 'consensus'
            elif len(data['voters']) == 2:
                synthesis['captain_strategy'][captain] = 'majority'
            else:
                synthesis['captain_strategy'][captain] = data['voters'][0].value
        
        # Rank all players
        all_players = set()
        player_scores = {}
        
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            weight = OptimizerConfig.AI_WEIGHTS[rec.source_ai]
            
            # Score captains
            for captain in rec.captain_targets:
                all_players.add(captain)
                player_scores[captain] = player_scores.get(captain, 0) + weight * 1.5
            
            # Score must plays
            for player in rec.must_play:
                all_players.add(player)
                player_scores[player] = player_scores.get(player, 0) + weight * 1.0
            
            # Penalize never plays
            for player in rec.never_play:
                all_players.add(player)
                player_scores[player] = player_scores.get(player, 0) - weight * 0.5
        
        # Create rankings
        synthesis['player_rankings'] = dict(sorted(player_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Combine stacking rules
        stack_map = {}
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            for stack in rec.stacks:
                stack_key = f"{stack.get('player1')}_{stack.get('player2')}"
                if stack_key not in stack_map:
                    stack_map[stack_key] = {'stack': stack, 'support': []}
                stack_map[stack_key]['support'].append(rec.source_ai)
        
        # Prioritize stacks by support
        for stack_key, data in stack_map.items():
            if len(data['support']) >= 2:
                synthesis['stacking_rules'].append({
                    **data['stack'],
                    'strength': 'strong',
                    'support': data['support']
                })
            else:
                synthesis['stacking_rules'].append({
                    **data['stack'],
                    'strength': 'moderate',
                    'support': data['support']
                })
        
        # Combine avoidance rules
        avoid_players = set()
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            avoid_players.update(rec.never_play)
        synthesis['avoidance_rules'] = list(avoid_players)
        
        # Create narrative
        narratives = []
        if game_theory_rec.narrative:
            narratives.append(f"Game Theory: {game_theory_rec.narrative}")
        if correlation_rec.narrative:
            narratives.append(f"Correlation: {correlation_rec.narrative}")
        if contrarian_rec.narrative:
            narratives.append(f"Contrarian: {contrarian_rec.narrative}")
        synthesis['narrative'] = " | ".join(narratives)
        
        # Calculate combined confidence
        synthesis['confidence'] = (
            game_theory_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.GAME_THEORY] +
            correlation_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.CORRELATION] +
            contrarian_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.CONTRARIAN_NARRATIVE]
        )
        
        # Create enforcement rules
        synthesis['enforcement_rules'] = self._create_enforcement_rules(synthesis)
        
        # Log synthesis
        self.logger.log(f"AI Synthesis complete: {len(synthesis['captain_strategy'])} captains, "
                       f"{len(synthesis['stacking_rules'])} stacks, "
                       f"confidence: {synthesis['confidence']:.2f}", "INFO")
        
        self.synthesis_history.append(synthesis)
        
        return synthesis
    
    def _create_enforcement_rules(self, synthesis: Dict) -> List[Dict]:
        """Convert synthesis into enforcement rules"""
        rules = []
        
        # Captain rules
        consensus_captains = [c for c, level in synthesis['captain_strategy'].items() if level == 'consensus']
        if consensus_captains:
            rules.append({
                'type': 'hard',
                'rule': 'captain_from_list',
                'players': consensus_captains,
                'description': 'Must use consensus captain'
            })
        
        # Stacking rules
        strong_stacks = [s for s in synthesis['stacking_rules'] if s['strength'] == 'strong']
        for stack in strong_stacks[:2]:  # Limit to top 2
            rules.append({
                'type': 'soft',
                'rule': 'include_stack',
                'players': [stack['player1'], stack['player2']],
                'weight': 0.8,
                'description': f"Include stack: {stack['player1']}-{stack['player2']}"
            })
        
        # Avoidance rules
        for player in synthesis['avoidance_rules']:
            rules.append({
                'type': 'soft',
                'rule': 'avoid_player',
                'player': player,
                'weight': 0.7,
                'description': f"Avoid: {player}"
            })
        
        return rules
    
    def get_synthesis_summary(self) -> Dict:
        """Get summary of synthesis history"""
        if not self.synthesis_history:
            return {}
        
        latest = self.synthesis_history[-1]
        return {
            'total_syntheses': len(self.synthesis_history),
            'latest_confidence': latest['confidence'],
            'consensus_captains': len([c for c, l in latest['captain_strategy'].items() if l == 'consensus']),
            'total_captains': len(latest['captain_strategy']),
            'strong_stacks': len([s for s in latest['stacking_rules'] if s['strength'] == 'strong'])
        }

# NFL GPP DUAL-AI OPTIMIZER - PART 3: AI STRATEGISTS (AI-AS-CHEF VERSION)
# Version 6.0 - Triple AI System with Contrarian Narrative Strategist

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import streamlit as st

# ============================================================================
# BASE AI STRATEGIST CLASS
# ============================================================================

class BaseAIStrategist:
    """Base class for all AI strategists with enforcement capabilities"""
    
    def __init__(self, api_manager=None, strategist_type: AIStrategistType = None):
        self.api_manager = api_manager
        self.strategist_type = strategist_type
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
    
    def get_recommendation(self, df: pd.DataFrame, game_info: Dict, 
                          field_size: str, use_api: bool = True) -> AIRecommendation:
        """Get AI recommendation with enforcement rules"""
        
        # Generate prompt
        prompt = self.generate_prompt(df, game_info, field_size)
        
        # Get response
        if use_api and self.api_manager and self.api_manager.client:
            response = self._get_api_response(prompt)
        else:
            response = '{}'  # Empty response triggers enforcement-based fallback
        
        # Parse response into enforcement rules
        recommendation = self.parse_response(response, df, field_size)
        
        # Add enforcement rules
        recommendation.enforcement_rules = self.create_enforcement_rules(recommendation, df, field_size)
        
        return recommendation
    
    def _get_api_response(self, prompt: str) -> str:
        """Get API response with performance tracking"""
        self.perf_monitor.start_timer(f"ai_{self.strategist_type.value}_api")
        
        try:
            response = self.api_manager.get_ai_response(prompt)
            self.perf_monitor.increment_counter("ai_api_calls")
            return response
        except Exception as e:
            self.logger.log(f"API error for {self.strategist_type.value}: {e}", "ERROR")
            return '{}'
        finally:
            self.perf_monitor.stop_timer(f"ai_{self.strategist_type.value}_api")
    
    def create_enforcement_rules(self, recommendation: AIRecommendation, 
                                df: pd.DataFrame, field_size: str) -> List[Dict]:
        """Create specific enforcement rules from recommendation"""
        rules = []
        
        # Captain enforcement
        if recommendation.captain_targets:
            if recommendation.confidence > 0.8:
                # High confidence = hard constraint
                rules.append({
                    'type': 'hard',
                    'constraint': f'captain_in_{recommendation.captain_targets}',
                    'players': recommendation.captain_targets,
                    'description': f'{self.strategist_type.value}: Must use one of these captains'
                })
            else:
                # Lower confidence = soft constraint
                rules.append({
                    'type': 'soft',
                    'constraint': f'prefer_captain_{recommendation.captain_targets}',
                    'players': recommendation.captain_targets,
                    'weight': recommendation.confidence,
                    'description': f'{self.strategist_type.value}: Prefer these captains'
                })
        
        # Must play enforcement
        for player in recommendation.must_play[:3]:  # Limit to top 3
            rules.append({
                'type': 'hard' if recommendation.confidence > 0.7 else 'soft',
                'constraint': f'must_include_{player}',
                'player': player,
                'weight': recommendation.confidence if recommendation.confidence <= 0.7 else 1.0,
                'description': f'{self.strategist_type.value}: Include {player}'
            })
        
        # Never play enforcement
        for player in recommendation.never_play[:3]:  # Limit to top 3
            rules.append({
                'type': 'hard' if recommendation.confidence > 0.8 else 'soft',
                'constraint': f'must_exclude_{player}',
                'player': player,
                'weight': recommendation.confidence if recommendation.confidence <= 0.8 else 1.0,
                'description': f'{self.strategist_type.value}: Exclude {player}'
            })
        
        return rules
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str) -> str:
        """Generate prompt - to be overridden by subclasses"""
        raise NotImplementedError
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse response - to be overridden by subclasses"""
        raise NotImplementedError

# ============================================================================
# GPP GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist(BaseAIStrategist):
    """AI Strategist 1: Game Theory and Ownership Leverage"""
    
    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate game theory focused prompt"""
        
        self.logger.log(f"Generating Game Theory prompt for {field_size}", "DEBUG")
        
        # Prepare data summaries
        bucket_manager = AIOwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)
        
        # Get ownership distribution
        ownership_summary = df.groupby(pd.cut(df['Ownership'], bins=[0, 5, 10, 20, 30, 100]))['Player'].count()
        
        # Field-specific strategy
        field_strategies = {
            'small_field': "Focus on slight differentiation while maintaining optimal plays",
            'medium_field': "Balance chalk with 2-3 strong leverage plays",
            'large_field': "Aggressive leverage with <15% owned captains",
            'milly_maker': "Maximum contrarian approach with <10% captains only"
        }
        
        prompt = f"""
        You are an expert DFS game theory strategist. Create an ENFORCEABLE lineup strategy for {field_size} GPP tournaments.
        
        GAME CONTEXT:
        {game_info}
        
        PLAYER POOL ANALYSIS:
        Total players: {len(df)}
        Ownership distribution: {ownership_summary.to_dict()}
        
        HIGH LEVERAGE PLAYS (<10% ownership):
        {df[df['Ownership'] < 10].nlargest(10, 'Projected_Points')[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_string()}
        
        FIELD STRATEGY:
        {field_strategies.get(field_size, 'Standard GPP')}
        
        PROVIDE SPECIFIC, ENFORCEABLE RULES IN JSON:
        {{
            "captain_rules": {{
                "must_be_one_of": ["player1", "player2", "player3"],
                "ownership_ceiling": 15,
                "reasoning": "Why these specific captains win"
            }},
            "lineup_rules": {{
                "must_include": ["player_x"],
                "never_include": ["player_y"],
                "ownership_sum_range": [60, 90]
            }},
            "correlation_rules": {{
                "required_stacks": [{{"player1": "name1", "player2": "name2"}}],
                "banned_combinations": []
            }},
            "confidence": 0.85,
            "key_insight": "The ONE thing that makes this lineup win tournaments"
        }}
        
        Be SPECIFIC with player names. Your rules will be ENFORCED as constraints.
        """
        
        return prompt
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse response into enforceable recommendation"""
        
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Extract captain rules
        captain_rules = data.get('captain_rules', {})
        captain_targets = captain_rules.get('must_be_one_of', [])
        
        # Validate captains exist in pool
        valid_captains = [c for c in captain_targets if c in df['Player'].values]
        
        # If no valid captains from AI, use game theory selection
        if not valid_captains:
            ownership_ceiling = captain_rules.get('ownership_ceiling', 15)
            eligible = df[df['Ownership'] <= ownership_ceiling].nlargest(5, 'Projected_Points')
            valid_captains = eligible['Player'].tolist()
        
        # Extract lineup rules
        lineup_rules = data.get('lineup_rules', {})
        must_include = [p for p in lineup_rules.get('must_include', []) if p in df['Player'].values]
        never_include = [p for p in lineup_rules.get('never_include', []) if p in df['Player'].values]
        
        # Extract stacks
        correlation_rules = data.get('correlation_rules', {})
        stacks = correlation_rules.get('required_stacks', [])
        
        # Build enforcement rules
        enforcement_rules = []
        
        # Captain constraint
        if valid_captains:
            enforcement_rules.append({
                'type': 'hard',
                'constraint': 'captain_selection',
                'players': valid_captains[:5],  # Top 5
                'description': 'Game theory optimal captains'
            })
        
        # Ownership sum constraint
        ownership_range = lineup_rules.get('ownership_sum_range', [60, 90])
        enforcement_rules.append({
            'type': 'hard',
            'constraint': 'ownership_sum',
            'min': ownership_range[0],
            'max': ownership_range[1],
            'description': f'Total ownership must be {ownership_range[0]}-{ownership_range[1]}%'
        })
        
        return AIRecommendation(
            captain_targets=valid_captains[:5],
            must_play=must_include,
            never_play=never_include,
            stacks=stacks,
            key_insights=[data.get('key_insight', 'Leverage game theory for GPP edge')],
            confidence=data.get('confidence', 0.7),
            enforcement_rules=enforcement_rules,
            narrative=data.get('key_insight', ''),
            source_ai=AIStrategistType.GAME_THEORY,
            ownership_leverage={'ownership_range': ownership_range}
        )

# ============================================================================
# GPP CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist(BaseAIStrategist):
    """AI Strategist 2: Correlation and Stacking Patterns"""
    
    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CORRELATION)
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate correlation focused prompt"""
        
        self.logger.log(f"Generating Correlation prompt for {field_size}", "DEBUG")
        
        # Team analysis
        teams = df['Team'].unique()[:2]
        team1_df = df[df['Team'] == teams[0]] if len(teams) > 0 else pd.DataFrame()
        team2_df = df[df['Team'] == teams[1]] if len(teams) > 1 else pd.DataFrame()
        
        # Game environment
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        
        prompt = f"""
        You are an expert DFS correlation strategist. Create SPECIFIC stacking rules for {field_size} GPP.
        
        GAME ENVIRONMENT:
        Total: {total} | Spread: {spread}
        
        TEAM 1 - {teams[0] if len(teams) > 0 else 'Unknown'}:
        {team1_df[['Player', 'Position', 'Projected_Points']].head(8).to_string() if not team1_df.empty else 'No data'}
        
        TEAM 2 - {teams[1] if len(teams) > 1 else 'Unknown'}:
        {team2_df[['Player', 'Position', 'Projected_Points']].head(8).to_string() if not team2_df.empty else 'No data'}
        
        CREATE ENFORCEABLE CORRELATION RULES IN JSON:
        {{
            "primary_stacks": [
                {{"type": "QB_WR", "player1": "exact_name", "player2": "exact_name", "correlation": 0.7}},
                {{"type": "QB_TE", "player1": "exact_name", "player2": "exact_name", "correlation": 0.6}}
            ],
            "game_stacks": [
                {{"players": ["player1", "player2", "player3"], "narrative": "shootout correlation"}}
            ],
            "leverage_stacks": [
                {{"type": "contrarian", "player1": "exact_name", "player2": "exact_name", "ownership_sum": 15}}
            ],
            "negative_correlation": [
                {{"avoid_together": ["player_x", "player_y"], "reason": "competing for touches"}}
            ],
            "captain_correlation": {{
                "best_captains_for_stacking": ["player_name"],
                "bring_back_rules": {{"if_captain": "player_x", "must_have_opponent": true}}
            }},
            "confidence": 0.8,
            "stack_narrative": "WHY this correlation wins GPPs"
        }}
        
        Use EXACT player names from the data. Focus on correlations that create tournament-winning ceilings.
        """
        
        return prompt
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse correlation response into enforceable rules"""
        
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Extract stacks
        all_stacks = []
        
        # Primary stacks (highest priority)
        for stack in data.get('primary_stacks', []):
            if self._validate_stack(stack, df):
                all_stacks.append({
                    **stack,
                    'priority': 'high',
                    'enforced': True
                })
        
        # Game stacks
        for stack in data.get('game_stacks', []):
            players = stack.get('players', [])
            if len(players) >= 2:
                valid_players = [p for p in players if p in df['Player'].values]
                if len(valid_players) >= 2:
                    all_stacks.append({
                        'player1': valid_players[0],
                        'player2': valid_players[1],
                        'type': 'game_stack',
                        'priority': 'medium',
                        'narrative': stack.get('narrative', '')
                    })
        
        # Leverage stacks
        for stack in data.get('leverage_stacks', []):
            if self._validate_stack(stack, df):
                all_stacks.append({
                    **stack,
                    'priority': 'high',
                    'leverage': True
                })
        
        # Captain correlation
        captain_rules = data.get('captain_correlation', {})
        captain_targets = captain_rules.get('best_captains_for_stacking', [])
        valid_captains = [c for c in captain_targets if c in df['Player'].values]
        
        # If no valid captains, select based on correlation potential
        if not valid_captains:
            qbs = df[df['Position'] == 'QB']['Player'].tolist()
            valid_captains = qbs[:2] if qbs else df.nlargest(3, 'Projected_Points')['Player'].tolist()
        
        # Build enforcement rules
        enforcement_rules = []
        
        # Enforce primary stacks
        high_priority_stacks = [s for s in all_stacks if s.get('priority') == 'high']
        for stack in high_priority_stacks[:2]:  # Top 2 stacks
            enforcement_rules.append({
                'type': 'hard',
                'constraint': 'must_stack',
                'players': [stack['player1'], stack['player2']],
                'description': f"Required correlation: {stack.get('type', 'stack')}"
            })
        
        # Enforce negative correlations
        for neg_corr in data.get('negative_correlation', []):
            players = neg_corr.get('avoid_together', [])
            if len(players) == 2 and all(p in df['Player'].values for p in players):
                enforcement_rules.append({
                    'type': 'hard',
                    'constraint': 'avoid_together',
                    'players': players,
                    'description': neg_corr.get('reason', 'Negative correlation')
                })
        
        # Bring-back rules
        bring_back = captain_rules.get('bring_back_rules', {})
        if bring_back.get('must_have_opponent'):
            enforcement_rules.append({
                'type': 'soft',
                'constraint': 'bring_back',
                'weight': 0.7,
                'description': 'Include opponent when stacking'
            })
        
        return AIRecommendation(
            captain_targets=valid_captains[:5],
            must_play=[],  # Filled by stacks
            never_play=[],
            stacks=all_stacks[:10],  # Top 10 stacks
            key_insights=[data.get('stack_narrative', 'Correlation-based lineup construction')],
            confidence=data.get('confidence', 0.75),
            enforcement_rules=enforcement_rules,
            narrative=data.get('stack_narrative', ''),
            source_ai=AIStrategistType.CORRELATION,
            correlation_matrix={s['type']: s for s in all_stacks[:5]}
        )
    
    def _validate_stack(self, stack: Dict, df: pd.DataFrame) -> bool:
        """Validate that stack players exist"""
        player1 = stack.get('player1')
        player2 = stack.get('player2')
        
        if not player1 or not player2:
            return False
        
        return (player1 in df['Player'].values and 
                player2 in df['Player'].values)

# ============================================================================
# GPP CONTRARIAN NARRATIVE STRATEGIST (NEW - 3RD AI)
# ============================================================================

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """AI Strategist 3: Contrarian Narratives and Hidden Angles"""
    
    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate contrarian narrative focused prompt"""
        
        self.logger.log(f"Generating Contrarian Narrative prompt for {field_size}", "DEBUG")
        
        # Find potential narrative plays
        low_owned_high_ceiling = df[df['Ownership'] < 10].nlargest(10, 'Projected_Points')
        
        # Find salary value plays
        df['Value'] = df['Projected_Points'] / (df['Salary'] / 1000)
        value_plays = df[df['Ownership'] < 15].nlargest(10, 'Value')
        
        # Team and game narrative
        teams = df['Team'].unique()[:2]
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        weather = game_info.get('weather', 'Clear')
        
        prompt = f"""
        You are a contrarian DFS strategist who finds the NON-OBVIOUS narratives that win GPP tournaments.
        Your job is to identify the 1% scenarios that create massive scores while everyone else follows chalk.
        
        GAME SETUP:
        {teams[0] if len(teams) > 0 else 'Team1'} vs {teams[1] if len(teams) > 1 else 'Team2'}
        Total: {total} | Spread: {spread} | Weather: {weather}
        
        LOW-OWNED CEILING PLAYS (<10% owned):
        {low_owned_high_ceiling[['Player', 'Position', 'Team', 'Projected_Points', 'Ownership']].to_string()}
        
        HIDDEN VALUE (High proj/$ but low owned):
        {value_plays[['Player', 'Position', 'Salary', 'Value', 'Ownership']].head(7).to_string()}
        
        CREATE CONTRARIAN TOURNAMENT-WINNING NARRATIVES IN JSON:
        {{
            "primary_narrative": "The ONE scenario everyone is missing",
            "contrarian_captains": [
                {{"player": "exact_name", "narrative": "Why this 5% captain wins", "ceiling_scenario": "specific game flow"}}
            ],
            "hidden_correlations": [
                {{"player1": "name1", "player2": "name2", "narrative": "Non-obvious connection"}}
            ],
            "fade_the_field": [
                {{"player": "chalky_player", "fade_reason": "Why the field is wrong", "pivot_to": "alternative_player"}}
            ],
            "boom_scenarios": [
                {{"players": ["player1", "player2"], "scenario": "What needs to happen", "probability": "low but possible"}}
            ],
            "contrarian_game_theory": {{
                "what_field_expects": "Common narrative",
                "why_field_is_wrong": "Your contrarian take",
                "exploit_this": "Specific players/stacks to exploit the field's mistake"
            }},
            "tournament_winner": {{
                "captain": "exact_name",
                "core": ["player1", "player2"],
                "narrative": "The story of how this lineup wins the million"
            }},
            "confidence": 0.7
        }}
        
        Be SPECIFIC. Name exact players. Your narratives become ENFORCED constraints.
        Find the story that makes a 2% owned player the optimal captain.
        """
        
        return prompt
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse contrarian narrative into enforceable rules"""
        
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Extract contrarian captains
        contrarian_captains = []
        captain_narratives = {}
        
        for captain_data in data.get('contrarian_captains', []):
            player = captain_data.get('player')
            if player and player in df['Player'].values:
                contrarian_captains.append(player)
                captain_narratives[player] = captain_data.get('narrative', '')
        
        # If no valid contrarian captains, find them algorithmically
        if not contrarian_captains:
            # Find low-owned players with high ceiling
            low_owned = df[df['Ownership'] < 10]
            if not low_owned.empty:
                # Sort by projected points and take top ones
                candidates = low_owned.nlargest(5, 'Projected_Points')
                contrarian_captains = candidates['Player'].tolist()
                
                # Create narratives
                for player in contrarian_captains:
                    row = df[df['Player'] == player].iloc[0]
                    captain_narratives[player] = f"Hidden ceiling at {row['Ownership']:.1f}% ownership"
        
        # Extract tournament winner lineup
        tournament_winner = data.get('tournament_winner', {})
        tw_captain = tournament_winner.get('captain')
        tw_core = tournament_winner.get('core', [])
        
        # Validate and use tournament winner
        must_play = []
        if tw_captain and tw_captain in df['Player'].values:
            if tw_captain not in contrarian_captains:
                contrarian_captains.insert(0, tw_captain)  # Priority captain
        
        for player in tw_core:
            if player in df['Player'].values:
                must_play.append(player)
        
        # Extract fades and pivots
        fades = []
        pivots = {}
        
        for fade_data in data.get('fade_the_field', []):
            fade_player = fade_data.get('player')
            pivot_player = fade_data.get('pivot_to')
            
            if fade_player and fade_player in df['Player'].values:
                fades.append(fade_player)
                if pivot_player and pivot_player in df['Player'].values:
                    pivots[fade_player] = pivot_player
                    if pivot_player not in must_play:
                        must_play.append(pivot_player)
        
        # Extract hidden correlations
        hidden_stacks = []
        for corr in data.get('hidden_correlations', []):
            if self._validate_correlation(corr, df):
                hidden_stacks.append({
                    'player1': corr['player1'],
                    'player2': corr['player2'],
                    'type': 'hidden',
                    'narrative': corr.get('narrative', 'Non-obvious correlation')
                })
        
        # Build enforcement rules
        enforcement_rules = []
        
        # Enforce contrarian captain
        if contrarian_captains:
            enforcement_rules.append({
                'type': 'hard',
                'constraint': 'contrarian_captain',
                'players': contrarian_captains[:3],  # Top 3
                'description': 'Must use contrarian captain for tournament upside'
            })
        
        # Enforce tournament winner core
        if must_play:
            for player in must_play[:2]:  # Top 2 core plays
                enforcement_rules.append({
                    'type': 'hard',
                    'constraint': f'tournament_core_{player}',
                    'player': player,
                    'description': f'Tournament winner core: {player}'
                })
        
        # Enforce fades
        for fade in fades[:2]:  # Top 2 fades
            enforcement_rules.append({
                'type': 'hard',
                'constraint': f'fade_{fade}',
                'player': fade,
                'exclude': True,
                'description': data.get('fade_the_field', [{}])[0].get('fade_reason', f'Fade {fade}')
            })
        
        # Enforce hidden correlations
        for stack in hidden_stacks[:1]:  # Top hidden stack
            enforcement_rules.append({
                'type': 'soft',
                'constraint': 'hidden_correlation',
                'players': [stack['player1'], stack['player2']],
                'weight': 0.8,
                'description': stack['narrative']
            })
        
        # Extract insights
        insights = []
        insights.append(data.get('primary_narrative', 'Contrarian approach'))
        
        game_theory = data.get('contrarian_game_theory', {})
        if game_theory.get('exploit_this'):
            insights.append(f"Exploit: {game_theory['exploit_this']}")
        
        return AIRecommendation(
            captain_targets=contrarian_captains[:5],
            must_play=must_play[:5],
            never_play=fades[:5],
            stacks=hidden_stacks,
            key_insights=insights,
            confidence=data.get('confidence', 0.7),
            enforcement_rules=enforcement_rules,
            narrative=data.get('primary_narrative', 'Contrarian narrative'),
            source_ai=AIStrategistType.CONTRARIAN_NARRATIVE,
            contrarian_angles=list(captain_narratives.values())
        )
    
    def _validate_correlation(self, correlation: Dict, df: pd.DataFrame) -> bool:
        """Validate correlation players exist"""
        player1 = correlation.get('player1')
        player2 = correlation.get('player2')
        
        return (player1 and player2 and 
                player1 in df['Player'].values and 
                player2 in df['Player'].values)

# ============================================================================
# CLAUDE API MANAGER (Updated for Triple AI)
# ============================================================================

class ClaudeAPIManager:
    """Manages Claude API interactions for all three AIs"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.cache = {}
        self.stats = {
            'requests': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_size': 0,
            'by_ai': {
                AIStrategistType.GAME_THEORY: {'requests': 0, 'errors': 0},
                AIStrategistType.CORRELATION: {'requests': 0, 'errors': 0},
                AIStrategistType.CONTRARIAN_NARRATIVE: {'requests': 0, 'errors': 0}
            }
        }
        self.logger = get_logger()
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Claude client"""
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            self.logger.log("Claude API client initialized", "INFO")
        except Exception as e:
            self.logger.log(f"Failed to initialize Claude API: {e}", "ERROR")
            self.client = None
    
    def get_ai_response(self, prompt: str, ai_type: Optional[AIStrategistType] = None) -> str:
        """Get response from Claude API with caching"""
        
        # Check cache
        prompt_hash = hash(prompt[:100])  # Use first 100 chars for hash
        if prompt_hash in self.cache:
            self.stats['cache_hits'] += 1
            self.logger.log("Cache hit for AI response", "DEBUG")
            return self.cache[prompt_hash]
        
        # Make API call
        self.stats['requests'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['requests'] += 1
        
        try:
            if not self.client:
                raise Exception("API client not initialized")
            
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1500,
                temperature=0.7,
                system="You are an expert DFS optimizer creating specific, enforceable lineup rules.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = message.content[0].text if message.content else "{}"
            
            # Cache response
            self.cache[prompt_hash] = response
            self.stats['cache_size'] = len(self.cache)
            
            self.logger.log(f"AI response received ({len(response)} chars)", "DEBUG")
            return response
            
        except Exception as e:
            self.stats['errors'] += 1
            if ai_type:
                self.stats['by_ai'][ai_type]['errors'] += 1
            
            self.logger.log(f"API error: {e}", "ERROR")
            return "{}"
    
    def get_stats(self) -> Dict:
        """Get API usage statistics"""
        return self.stats
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        self.stats['cache_size'] = 0
        self.logger.log("API cache cleared", "INFO")

# ============================================================================
# AI RESPONSE FALLBACK SYSTEM
# ============================================================================

class AIFallbackSystem:
    """Provides enforcement-ready fallbacks when AI is unavailable"""
    
    @staticmethod
    def get_game_theory_fallback(df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate game theory rules without AI"""
        
        # Find leverage captains
        low_owned = df[df['Ownership'] < 15]
        captains = low_owned.nlargest(5, 'Projected_Points')['Player'].tolist()
        
        if not captains:
            captains = df.nlargest(5, 'Projected_Points')['Player'].tolist()
        
        enforcement_rules = [
            {
                'type': 'hard',
                'constraint': 'leverage_captain',
                'players': captains[:3],
                'description': 'Statistical leverage captains'
            }
        ]
        
        return AIRecommendation(
            captain_targets=captains,
            must_play=[],
            never_play=df[df['Ownership'] > 35]['Player'].tolist()[:3],
            stacks=[],
            key_insights=['Using statistical game theory'],
            confidence=0.5,
            enforcement_rules=enforcement_rules,
            narrative='Fallback game theory strategy',
            source_ai=AIStrategistType.GAME_THEORY
        )
    
    @staticmethod
    def get_correlation_fallback(df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate correlation rules without AI"""
        
        stacks = []
        teams = df['Team'].unique()
        
        for team in teams[:2]:
            team_df = df[df['Team'] == team]
            qbs = team_df[team_df['Position'] == 'QB']['Player'].tolist()
            pass_catchers = team_df[team_df['Position'].isin(['WR', 'TE'])].nlargest(3, 'Projected_Points')['Player'].tolist()
            
            for qb in qbs:
                for pc in pass_catchers:
                    stacks.append({
                        'player1': qb,
                        'player2': pc,
                        'type': 'QB_stack'
                    })
        
        enforcement_rules = []
        if stacks:
            enforcement_rules.append({
                'type': 'soft',
                'constraint': 'correlation_stack',
                'players': [stacks[0]['player1'], stacks[0]['player2']],
                'weight': 0.7,
                'description': 'Primary correlation stack'
            })
        
        return AIRecommendation(
            captain_targets=df[df['Position'] == 'QB']['Player'].tolist()[:2],
            must_play=[],
            never_play=[],
            stacks=stacks[:5],
            key_insights=['Using statistical correlations'],
            confidence=0.5,
            enforcement_rules=enforcement_rules,
            narrative='Fallback correlation strategy',
            source_ai=AIStrategistType.CORRELATION
        )
    
    @staticmethod
    def get_contrarian_fallback(df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate contrarian rules without AI"""
        
        # Find contrarian plays
        df['Contrarian_Score'] = (df['Projected_Points'] / df['Projected_Points'].max()) / (df['Ownership'] / 100 + 0.1)
        contrarian = df.nlargest(5, 'Contrarian_Score')
        
        captains = contrarian['Player'].tolist()
        
        enforcement_rules = [
            {
                'type': 'hard',
                'constraint': 'contrarian_captain',
                'players': captains[:3],
                'description': 'Statistical contrarian captains'
            }
        ]
        
        return AIRecommendation(
            captain_targets=captains,
            must_play=df[df['Ownership'] < 5]['Player'].tolist()[:2],
            never_play=df[df['Ownership'] > 30]['Player'].tolist()[:3],
            stacks=[],
            key_insights=['Using statistical contrarian analysis'],
            confidence=0.5,
            enforcement_rules=enforcement_rules,
            narrative='Fallback contrarian strategy',
            source_ai=AIStrategistType.CONTRARIAN_NARRATIVE
        )

# NFL GPP DUAL-AI OPTIMIZER - PART 4: MAIN OPTIMIZER & LINEUP GENERATION (COMPLETE CORRECTED v6.2)
# With Safe Logger Implementation and Full Error Handling

import pandas as pd
import numpy as np
import pulp
from typing import Dict, List, Tuple, Optional, Set, Any
import streamlit as st
import json
from datetime import datetime

# ============================================================================
# AI-DRIVEN GPP OPTIMIZER WITH SAFE LOGGER
# ============================================================================

class AIChefGPPOptimizer:
    """Main optimizer where AI is the chef and optimization is just execution"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field', 
                 api_manager=None):
        self.df = df
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager
        
        # Initialize the three AI strategists
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.contrarian_ai = GPPContrarianNarrativeStrategist(api_manager)
        
        # Use safe logger instead of get_logger()
        self.logger = self._create_safe_logger()
        
        # AI enforcement and synthesis
        self.enforcement_engine = AIEnforcementEngine(
            OptimizerConfig.FIELD_SIZE_CONFIGS[field_size].get(
                'ai_enforcement', AIEnforcementLevel.MANDATORY
            )
        )
        self.synthesis_engine = AISynthesisEngine()
        
        # Supporting components
        self.bucket_manager = AIOwnershipBucketManager(self.enforcement_engine)
        self.pivot_generator = GPPCaptainPivotGenerator()
        
        # Use safe performance monitor
        self.perf_monitor = self._create_safe_performance_monitor()
        
        # Tracking
        self.ai_decisions_log = []
        self.optimization_log = []
    
    def _create_safe_logger(self):
        """Create a logger that definitely has all required methods"""
        
        class SafeLogger:
            def __init__(self):
                self.entries = []
                self.verbose = False
                self.ai_tracker = self._create_ai_tracker()
            
            def _create_ai_tracker(self):
                """Create a basic AI tracker"""
                class BasicTracker:
                    def __init__(self):
                        self.stats = {
                            'total_rules': 0,
                            'enforced_rules': 0,
                            'violated_rules': 0,
                            'consensus_decisions': 0,
                            'majority_decisions': 0,
                            'single_ai_decisions': 0
                        }
                        self.ai_performance = {}
                    
                    def track_decision(self, decision_type, ai_source, enforced, details):
                        self.stats['total_rules'] += 1
                        if enforced:
                            self.stats['enforced_rules'] += 1
                        else:
                            self.stats['violated_rules'] += 1
                    
                    def track_consensus(self, consensus_type, ais_agreeing):
                        if len(ais_agreeing) == 3:
                            self.stats['consensus_decisions'] += 1
                        elif len(ais_agreeing) == 2:
                            self.stats['majority_decisions'] += 1
                        else:
                            self.stats['single_ai_decisions'] += 1
                    
                    def get_enforcement_rate(self):
                        if self.stats['total_rules'] == 0:
                            return 0.0
                        return self.stats['enforced_rules'] / self.stats['total_rules']
                    
                    def get_summary(self):
                        return {
                            'enforcement_rate': self.get_enforcement_rate(),
                            'stats': self.stats,
                            'ai_performance': self.ai_performance,
                            'recent_decisions': []
                        }
                
                return BasicTracker()
            
            def log(self, message, level="INFO"):
                """Basic logging"""
                try:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] {message}")
                    self.entries.append({
                        'timestamp': datetime.now(),
                        'level': level,
                        'message': str(message)
                    })
                except Exception as e:
                    print(f"Logging error: {e}")
            
            def log_optimization_start(self, num_lineups, field_size, settings):
                """Log optimization start"""
                self.log(f"Starting optimization: {num_lineups} lineups for {field_size}", "INFO")
                if self.verbose:
                    self.log(f"Settings: {settings}", "DEBUG")
            
            def log_optimization_end(self, lineups_generated, total_time):
                """Log optimization end - THE CRITICAL METHOD"""
                self.log(f"Optimization complete: {lineups_generated} lineups in {total_time:.2f}s", "INFO")
                if lineups_generated > 0:
                    avg_time = total_time / lineups_generated
                    self.log(f"Average time per lineup: {avg_time:.3f}s", "DEBUG")
            
            def log_lineup_generation(self, strategy, lineup_num, status, ai_rules_enforced=0):
                """Log lineup generation"""
                message = f"Lineup {lineup_num} ({strategy}): {status}"
                if ai_rules_enforced > 0:
                    message += f" - {ai_rules_enforced} AI rules enforced"
                self.log(message, "DEBUG" if status == "SUCCESS" else "WARNING")
            
            def log_exception(self, exception, context=""):
                """Log exception"""
                self.log(f"Exception in {context}: {str(exception)}", "ERROR")
            
            def log_ai_decision(self, decision_type, ai_source, enforced, details):
                """Log AI decision"""
                try:
                    self.ai_tracker.track_decision(decision_type, ai_source, enforced, details)
                    message = f"AI Decision [{ai_source}]: {decision_type} - {'ENFORCED' if enforced else 'VIOLATED'}"
                    self.log(message, "AI_DECISION")
                except Exception as e:
                    self.log(f"Error logging AI decision: {e}", "ERROR")
            
            def log_ai_consensus(self, consensus_type, ais_agreeing, decision):
                """Log AI consensus"""
                try:
                    self.ai_tracker.track_consensus(consensus_type, ais_agreeing)
                    message = f"AI Consensus ({len(ais_agreeing)}/3): {decision}"
                    self.log(message, "AI_CONSENSUS")
                except Exception as e:
                    self.log(f"Error logging AI consensus: {e}", "ERROR")
            
            def get_ai_summary(self):
                """Get AI summary"""
                try:
                    return self.ai_tracker.get_summary()
                except:
                    return {
                        'enforcement_rate': 0.0,
                        'stats': {
                            'total_rules': 0,
                            'enforced_rules': 0,
                            'violated_rules': 0,
                            'consensus_decisions': 0,
                            'majority_decisions': 0,
                            'single_ai_decisions': 0
                        },
                        'ai_performance': {},
                        'recent_decisions': []
                    }
            
            def display_ai_enforcement(self):
                """Display AI enforcement in Streamlit"""
                try:
                    summary = self.get_ai_summary()
                    st.markdown("### 🤖 AI Decision Enforcement")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        rate = summary['enforcement_rate'] * 100
                        st.metric("Enforcement Rate", f"{rate:.1f}%")
                    
                    with col2:
                        stats = summary['stats']
                        st.metric("Rules Enforced", 
                                 f"{stats['enforced_rules']}/{stats['total_rules']}")
                    
                    with col3:
                        st.metric("Consensus Decisions", stats.get('consensus_decisions', 0))
                except Exception as e:
                    st.error(f"Error displaying AI enforcement: {e}")
            
            def display_log_summary(self):
                """Display log summary"""
                try:
                    if self.entries:
                        st.markdown("### 📋 Recent Logs")
                        for entry in self.entries[-5:]:
                            st.text(f"[{entry['timestamp'].strftime('%H:%M:%S')}] {entry['message']}")
                except:
                    pass
            
            def export_logs(self):
                """Export logs"""
                try:
                    return "\n".join([
                        f"{e['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}: {e['message']}" 
                        for e in self.entries
                    ])
                except:
                    return "No logs available"
        
        return SafeLogger()
    
    def _create_safe_performance_monitor(self):
        """Create a safe performance monitor"""
        
        class SafePerformanceMonitor:
            def __init__(self):
                self.timers = {}
                self.counters = {}
            
            def start_timer(self, name):
                """Start a timer"""
                try:
                    self.timers[name] = datetime.now()
                except:
                    pass
            
            def stop_timer(self, name):
                """Stop a timer and return elapsed time"""
                try:
                    if name in self.timers:
                        elapsed = (datetime.now() - self.timers[name]).total_seconds()
                        return elapsed
                except:
                    pass
                return 0.0
            
            def increment_counter(self, name, value=1):
                """Increment a counter"""
                try:
                    if name not in self.counters:
                        self.counters[name] = 0
                    self.counters[name] += value
                except:
                    pass
            
            def get_metrics(self):
                """Get metrics"""
                return {'timers': {}, 'counters': self.counters}
            
            def display_metrics(self):
                """Display metrics"""
                try:
                    if self.counters:
                        st.markdown("### Performance Metrics")
                        for name, count in self.counters.items():
                            st.metric(name, count)
                except:
                    pass
        
        return SafePerformanceMonitor()
    
    def get_triple_ai_strategies(self, use_api: bool = True) -> Dict:
        """Get strategies from all three AIs"""
        
        self.logger.log("Getting strategies from three AI strategists", "INFO")
        recommendations = {}
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode
            with st.spinner("🎯 Game Theory AI analyzing..."):
                self.perf_monitor.start_timer("ai_game_theory")
                try:
                    recommendations[AIStrategistType.GAME_THEORY] = self.game_theory_ai.get_recommendation(
                        self.df, self.game_info, self.field_size, use_api=True
                    )
                except Exception as e:
                    self.logger.log(f"Game Theory AI error: {e}", "ERROR")
                    recommendations[AIStrategistType.GAME_THEORY] = self._get_fallback_recommendation(AIStrategistType.GAME_THEORY)
                self.perf_monitor.stop_timer("ai_game_theory")
            
            with st.spinner("🔗 Correlation AI analyzing..."):
                self.perf_monitor.start_timer("ai_correlation")
                try:
                    recommendations[AIStrategistType.CORRELATION] = self.correlation_ai.get_recommendation(
                        self.df, self.game_info, self.field_size, use_api=True
                    )
                except Exception as e:
                    self.logger.log(f"Correlation AI error: {e}", "ERROR")
                    recommendations[AIStrategistType.CORRELATION] = self._get_fallback_recommendation(AIStrategistType.CORRELATION)
                self.perf_monitor.stop_timer("ai_correlation")
            
            with st.spinner("🎭 Contrarian AI analyzing..."):
                self.perf_monitor.start_timer("ai_contrarian")
                try:
                    recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self.contrarian_ai.get_recommendation(
                        self.df, self.game_info, self.field_size, use_api=True
                    )
                except Exception as e:
                    self.logger.log(f"Contrarian AI error: {e}", "ERROR")
                    recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self._get_fallback_recommendation(AIStrategistType.CONTRARIAN_NARRATIVE)
                self.perf_monitor.stop_timer("ai_contrarian")
        
        else:
            # Manual mode
            st.subheader("📝 Triple AI Strategy Input")
            
            responses = {}
            tab1, tab2, tab3 = st.tabs(["🎯 Game Theory", "🔗 Correlation", "🎭 Contrarian"])
            
            with tab1:
                gt_response = self._get_manual_ai_input("Game Theory", self.game_theory_ai)
                responses[AIStrategistType.GAME_THEORY] = gt_response
            
            with tab2:
                corr_response = self._get_manual_ai_input("Correlation", self.correlation_ai)
                responses[AIStrategistType.CORRELATION] = corr_response
            
            with tab3:
                contra_response = self._get_manual_ai_input("Contrarian", self.contrarian_ai)
                responses[AIStrategistType.CONTRARIAN_NARRATIVE] = contra_response
            
            # Parse responses
            for ai_type, response in responses.items():
                try:
                    if ai_type == AIStrategistType.GAME_THEORY:
                        recommendations[ai_type] = self.game_theory_ai.parse_response(response, self.df, self.field_size)
                    elif ai_type == AIStrategistType.CORRELATION:
                        recommendations[ai_type] = self.correlation_ai.parse_response(response, self.df, self.field_size)
                    else:
                        recommendations[ai_type] = self.contrarian_ai.parse_response(response, self.df, self.field_size)
                except Exception as e:
                    self.logger.log(f"Error parsing {ai_type.value}: {e}", "ERROR")
                    recommendations[ai_type] = self._get_fallback_recommendation(ai_type)
        
        # Validate we have recommendations
        if not recommendations:
            self.logger.log("No AI recommendations available, using fallback", "WARNING")
            for ai_type in [AIStrategistType.GAME_THEORY, AIStrategistType.CORRELATION, AIStrategistType.CONTRARIAN_NARRATIVE]:
                recommendations[ai_type] = self._get_fallback_recommendation(ai_type)
        
        return recommendations
    
    def _get_fallback_recommendation(self, ai_type: AIStrategistType) -> AIRecommendation:
        """Get a fallback recommendation when AI fails"""
        
        # Get top players by projection
        top_players = self.df.nlargest(10, 'Projected_Points')['Player'].tolist()
        
        return AIRecommendation(
            captain_targets=top_players[:5],
            must_play=[],
            never_play=[],
            stacks=[],
            key_insights=["Fallback strategy due to AI unavailable"],
            confidence=0.3,
            enforcement_rules=[],
            narrative="Using statistical fallback",
            source_ai=ai_type,
            boosts=[],
            fades=[]
        )
    
    def _get_manual_ai_input(self, ai_name: str, strategist) -> str:
        """Get manual AI input with validation"""
        with st.expander(f"View {ai_name} Prompt"):
            prompt = strategist.generate_prompt(self.df, self.game_info, self.field_size)
            st.text_area(f"Copy this:", value=prompt, height=250, key=f"{ai_name}_prompt_display")
        
        response = st.text_area(f"Paste {ai_name} Response (JSON):", 
                               height=200, key=f"{ai_name}_response", value='{}')
        
        # Validate JSON
        if response and response != '{}':
            try:
                json.loads(response)
                st.success(f"✅ Valid {ai_name} JSON")
            except:
                st.error(f"❌ Invalid {ai_name} JSON")
        
        return response
    
    def synthesize_ai_strategies(self, recommendations: Dict) -> Dict:
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
            enforcement_rules = self.enforcement_engine.create_enforcement_rules(recommendations)
            
            # Validate rules
            validation = AIConfigValidator.validate_ai_requirements(enforcement_rules, self.df)
            
            if not validation['is_valid']:
                st.warning("⚠️ Some AI requirements cannot be satisfied:")
                for error in validation.get('errors', []):
                    st.write(f"  - {error}")
            
            return {
                'synthesis': synthesis,
                'enforcement_rules': enforcement_rules,
                'validation': validation
            }
        
        except Exception as e:
            self.logger.log(f"Error in synthesis: {e}", "ERROR")
            # Return minimal valid synthesis
            return {
                'synthesis': {
                    'captain_strategy': {},
                    'player_rankings': {},
                    'stacking_rules': [],
                    'enforcement_rules': [],
                    'confidence': 0.3,
                    'narrative': "Synthesis failed, using basic strategy"
                },
                'enforcement_rules': {'hard_constraints': [], 'soft_constraints': []},
                'validation': {'is_valid': True}
            }
    
    def generate_ai_driven_lineups(self, num_lineups: int, ai_strategy: Dict) -> pd.DataFrame:
        """Generate lineups following AI directives"""
        
        self.perf_monitor.start_timer("total_optimization")
        self.logger.log_optimization_start(num_lineups, self.field_size, {
            'mode': 'AI-as-Chef',
            'enforcement': self.enforcement_engine.enforcement_level.value
        })
        
        # Extract components
        synthesis = ai_strategy.get('synthesis', {})
        enforcement_rules = ai_strategy.get('enforcement_rules', {})
        
        # Show AI consensus
        self._display_ai_consensus(synthesis)
        
        # Get player data
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Apply AI modifications
        ai_adjusted_points = self._apply_ai_adjustments(points, synthesis)
        
        # Get strategy distribution
        consensus_level = self._determine_consensus_level(synthesis)
        strategy_distribution = AIConfigValidator.get_ai_strategy_distribution(
            self.field_size, num_lineups, consensus_level
        )
        
        self.logger.log(f"AI Strategy distribution: {strategy_distribution}", "INFO")
        
        all_lineups = []
        used_captains = set()
        
        # Generate lineups by AI strategy
        for strategy, count in strategy_distribution.items():
            strategy_name = strategy if isinstance(strategy, str) else strategy.value if hasattr(strategy, 'value') else str(strategy)
            self.logger.log(f"Generating {count} lineups with strategy: {strategy_name}", "INFO")
            
            for i in range(count):
                lineup_num = len(all_lineups) + 1
                
                try:
                    # Build lineup following AI rules
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
                        # Validate against AI requirements
                        is_valid, violations = self.enforcement_engine.validate_lineup_against_ai(
                            lineup, enforcement_rules
                        )
                        
                        if is_valid or len(all_lineups) < num_lineups / 2:  # Accept some invalid lineups if struggling
                            all_lineups.append(lineup)
                            used_captains.add(lineup['Captain'])
                            
                            self.logger.log_lineup_generation(
                                strategy_name, lineup_num, "SUCCESS", 
                                len(enforcement_rules.get('hard_constraints', []))
                            )
                        else:
                            self.logger.log(f"Lineup {lineup_num} violated: {violations}", "WARNING")
                    else:
                        self.logger.log(f"Failed to generate lineup {lineup_num}", "WARNING")
                
                except Exception as e:
                    self.logger.log_exception(e, f"Lineup {lineup_num} generation")
        
        # Check results
        total_time = self.perf_monitor.stop_timer("total_optimization")
        
        # This is the critical line that was causing the error - now it will work
        self.logger.log_optimization_end(len(all_lineups), total_time)
        
        if len(all_lineups) == 0:
            st.error("❌ No valid lineups generated")
            self._display_optimization_issues()
            return pd.DataFrame()
        
        if len(all_lineups) < num_lineups:
            st.warning(f"Generated {len(all_lineups)}/{num_lineups} lineups")
        else:
            st.success(f"✅ Generated {len(all_lineups)} lineups!")
        
        # Display AI enforcement statistics
        self.logger.display_ai_enforcement()
        
        return pd.DataFrame(all_lineups)
    
    def _build_ai_enforced_lineup(self, lineup_num: int, strategy: str, players: List[str],
                                 salaries: Dict, points: Dict, ownership: Dict,
                                 positions: Dict, teams: Dict, enforcement_rules: Dict,
                                 synthesis: Dict, used_captains: Set[str]) -> Optional[Dict]:
        """Build a single lineup enforcing AI rules"""
        
        try:
            import pulp
            
            model = pulp.LpProblem(f"AI_Lineup_{lineup_num}_{strategy}", pulp.LpMaximize)
            
            # Variables
            flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
            captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
            
            # AI-modified objective
            player_weights = synthesis.get('player_rankings', {})
            
            objective = pulp.lpSum([
                points.get(p, 0) * player_weights.get(p, 1.0) * flex[p] +
                1.5 * points.get(p, 0) * player_weights.get(p, 1.0) * captain[p]
                for p in players
            ])
            
            model += objective
            
            # Basic constraints
            model += pulp.lpSum(captain.values()) == 1
            model += pulp.lpSum(flex.values()) == 5
            
            for p in players:
                model += flex[p] + captain[p] <= 1
            
            # Salary constraint
            model += pulp.lpSum([
                salaries.get(p, 0) * flex[p] + 1.5 * salaries.get(p, 0) * captain[p]
                for p in players
            ]) <= OptimizerConfig.SALARY_CAP
            
            # Team constraint
            for team in set(teams.values()):
                team_players = [p for p in players if teams.get(p) == team]
                if team_players:
                    model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
            
            # Apply AI constraints (simplified to avoid over-constraining)
            captain_requirements = []
            
            # Get captain recommendations from synthesis
            if synthesis.get('captain_strategy'):
                for captain_name, consensus_type in synthesis['captain_strategy'].items():
                    if captain_name in players and captain_name not in used_captains:
                        if consensus_type == 'consensus' or (strategy == 'ai_consensus' and consensus_type == 'majority'):
                            captain_requirements.append(captain_name)
            
            # If we have captain requirements, enforce one
            if captain_requirements:
                model += pulp.lpSum([captain[c] for c in captain_requirements]) == 1
            elif used_captains and len(used_captains) < len(players):
                # Force unique captain
                for prev_captain in used_captains:
                    if prev_captain in players:
                        model += captain[prev_captain] == 0
            
            # Solve
            model.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[model.status] == 'Optimal':
                captain_pick = None
                flex_picks = []
                
                for p in players:
                    if captain[p].value() == 1:
                        captain_pick = p
                    if flex[p].value() == 1:
                        flex_picks.append(p)
                
                if captain_pick and len(flex_picks) == 5:
                    total_salary = sum(salaries.get(p, 0) for p in flex_picks) + 1.5 * salaries.get(captain_pick, 0)
                    total_proj = sum(points.get(p, 0) for p in flex_picks) + 1.5 * points.get(captain_pick, 0)
                    total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                    
                    return {
                        'Lineup': lineup_num,
                        'Strategy': strategy,
                        'Captain': captain_pick,
                        'Captain_Own%': ownership.get(captain_pick, 5),
                        'FLEX': flex_picks,
                        'Projected': round(total_proj, 2),
                        'Salary': int(total_salary),
                        'Salary_Remaining': int(OptimizerConfig.SALARY_CAP - total_salary),
                        'Total_Ownership': round(total_ownership, 1),
                        'AI_Strategy': strategy,
                        'AI_Enforced': True,
                        'Confidence': synthesis.get('confidence', 0.5)
                    }
            
            return None
            
        except Exception as e:
            self.logger.log_exception(e, f"Building lineup {lineup_num}")
            return None
    
    def _apply_ai_adjustments(self, points: Dict, synthesis: Dict) -> Dict:
        """Apply AI-recommended adjustments to projections"""
        adjusted = points.copy()
        
        try:
            # Apply player rankings as multipliers
            rankings = synthesis.get('player_rankings', {})
            
            for player, score in rankings.items():
                if player in adjusted:
                    # Convert score to multiplier
                    multiplier = 1.0 + (score * 0.2)
                    adjusted[player] = adjusted.get(player, 0) * multiplier
        except Exception as e:
            self.logger.log(f"Error applying AI adjustments: {e}", "ERROR")
        
        return adjusted
    
    def _determine_consensus_level(self, synthesis: Dict) -> str:
        """Determine the level of AI consensus"""
        try:
            captain_strategy = synthesis.get('captain_strategy', {})
            
            if not captain_strategy:
                return 'low'
            
            consensus_count = len([c for c, level in captain_strategy.items() if level == 'consensus'])
            majority_count = len([c for c, level in captain_strategy.items() if level == 'majority'])
            
            total = len(captain_strategy)
            
            if total == 0:
                return 'low'
            
            consensus_ratio = consensus_count / total
            majority_ratio = (consensus_count + majority_count) / total
            
            if consensus_ratio > 0.3:
                return 'high'
            elif majority_ratio > 0.5:
                return 'mixed'
            else:
                return 'low'
        except:
            return 'low'
    
    def _display_ai_consensus(self, synthesis: Dict):
        """Display AI consensus analysis"""
        try:
            st.markdown("### 🤖 AI Consensus Analysis")
            
            captain_strategy = synthesis.get('captain_strategy', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                consensus_captains = len([c for c, l in captain_strategy.items() if l == 'consensus'])
                st.metric("Consensus Captains", consensus_captains)
            
            with col2:
                strong_stacks = len([s for s in synthesis.get('stacking_rules', []) if s.get('strength') == 'strong'])
                st.metric("Strong Stacks", strong_stacks)
            
            with col3:
                st.metric("AI Confidence", f"{synthesis.get('confidence', 0):.0%}")
            
            with col4:
                enforcement = len(synthesis.get('enforcement_rules', []))
                st.metric("Enforcement Rules", enforcement)
            
            # Show narrative
            if synthesis.get('narrative'):
                st.info(f"**AI Narrative:** {synthesis['narrative']}")
        except Exception as e:
            self.logger.log(f"Error displaying AI consensus: {e}", "ERROR")
    
    def _display_optimization_issues(self):
        """Display optimization issues for debugging"""
        try:
            if self.optimization_log:
                with st.expander("⚠️ Optimization Issues", expanded=True):
                    for issue in self.optimization_log[-10:]:
                        st.write(f"- {issue}")
        except:
            pass

# ============================================================================
# CAPTAIN PIVOT GENERATOR (AI-Enhanced)
# ============================================================================

class GPPCaptainPivotGenerator:
    """Generate captain pivots with AI guidance"""
    
    def __init__(self):
        try:
            self.logger = get_logger()
        except:
            self.logger = None
    
    def generate_ai_guided_pivots(self, lineup: Dict, df: pd.DataFrame, 
                                 synthesis: Dict, max_pivots: int = 5) -> List[Dict]:
        """Generate pivots following AI recommendations"""
        
        pivots = []
        
        try:
            current_captain = lineup.get('Captain')
            flex_players = lineup.get('FLEX', [])
            
            # Get AI-recommended captains
            captain_strategy = synthesis.get('captain_strategy', {})
            ai_captains = list(captain_strategy.keys())
            
            # Prioritize AI captains not yet used
            for new_captain in ai_captains:
                if new_captain != current_captain and new_captain in df['Player'].values:
                    if new_captain in flex_players:
                        # Swap captain with flex
                        new_flex = flex_players.copy()
                        new_flex.remove(new_captain)
                        new_flex.append(current_captain)
                        
                        pivot = {
                            'Original_Captain': current_captain,
                            'Captain': new_captain,
                            'FLEX': new_flex,
                            'Pivot_Type': f"AI-{captain_strategy.get(new_captain, 'recommended')}",
                            'AI_Recommended': True
                        }
                        
                        pivots.append(pivot)
                        
                        if len(pivots) >= max_pivots:
                            break
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error generating pivots: {e}", "ERROR")
        
        return pivots

# NFL GPP DUAL-AI OPTIMIZER - PART 5: MAIN UI AND HELPER FUNCTIONS (AI-AS-CHEF VERSION)
# Version 6.0 - Triple AI System with AI-Driven UI

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from datetime import datetime
import hashlib
import pickle
import json

# ============================================================================
# PERFORMANCE OPTIMIZATION - AI-FOCUSED CACHING
# ============================================================================

@st.cache_data(ttl=3600)
def cache_ai_synthesis(gt_json: str, corr_json: str, contra_json: str, 
                      field_size: str) -> Dict:
    """Cache AI synthesis results"""
    # Parse JSON recommendations
    gt_rec = json.loads(gt_json)
    corr_rec = json.loads(corr_json)
    contra_rec = json.loads(contra_json)
    
    # Synthesize
    synthesis_engine = AISynthesisEngine()
    synthesis = synthesis_engine.synthesize_recommendations(
        gt_rec, corr_rec, contra_rec
    )
    
    return synthesis

@st.cache_data(ttl=1800)
def cache_ai_enforcement_rules(synthesis_json: str, df_json: str, 
                              field_size: str) -> Dict:
    """Cache AI enforcement rule generation"""
    synthesis = json.loads(synthesis_json)
    df = pd.read_json(df_json)
    
    enforcement_engine = AIEnforcementEngine()
    rules = enforcement_engine.create_enforcement_rules(synthesis)
    
    # Validate rules
    validation = AIConfigValidator.validate_ai_requirements(rules, df)
    
    return {
        'rules': rules,
        'validation': validation
    }

# ============================================================================
# SESSION STATE MANAGEMENT - AI TRACKING
# ============================================================================

def init_ai_session_state():
    """Initialize session state for AI-driven optimization"""
    if 'ai_recommendations' not in st.session_state:
        st.session_state['ai_recommendations'] = {}
    if 'ai_synthesis' not in st.session_state:
        st.session_state['ai_synthesis'] = None
    if 'ai_enforcement_history' not in st.session_state:
        st.session_state['ai_enforcement_history'] = []
    if 'optimization_history' not in st.session_state:
        st.session_state['optimization_history'] = []
    if 'ai_mode' not in st.session_state:
        st.session_state['ai_mode'] = 'enforced'  # 'enforced' or 'advisory'

def save_ai_optimization_session(lineups_df: pd.DataFrame, ai_synthesis: Dict, 
                                settings: Dict):
    """Save AI-driven optimization session"""
    session = {
        'timestamp': datetime.now(),
        'lineups': lineups_df.to_dict('records'),
        'ai_synthesis': ai_synthesis,
        'settings': settings,
        'ai_consensus_level': determine_consensus_level(ai_synthesis),
        'enforcement_stats': get_logger().get_ai_summary()
    }
    
    st.session_state['optimization_history'].append(session)
    
    # Limit history
    if len(st.session_state['optimization_history']) > 10:
        st.session_state['optimization_history'].pop(0)

def determine_consensus_level(synthesis: Dict) -> str:
    """Determine AI consensus level"""
    if not synthesis:
        return 'none'
    
    captain_strategy = synthesis.get('captain_strategy', {})
    consensus = len([c for c, l in captain_strategy.items() if l == 'consensus'])
    total = len(captain_strategy)
    
    if total == 0:
        return 'none'
    
    ratio = consensus / total
    if ratio > 0.5:
        return 'high'
    elif ratio > 0.2:
        return 'medium'
    else:
        return 'low'

# ============================================================================
# AI VALIDATION AND DISPLAY
# ============================================================================

def display_ai_recommendations(recommendations: Dict[AIStrategistType, AIRecommendation]):
    """Display the three AI recommendations"""
    st.markdown("### 🤖 Triple AI Strategic Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Game Theory", "🔗 Correlation", "🎭 Contrarian"])
    
    with tab1:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.GAME_THEORY),
            "Game Theory",
            "🎯"
        )
    
    with tab2:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CORRELATION),
            "Correlation",
            "🔗"
        )
    
    with tab3:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE),
            "Contrarian",
            "🎭"
        )

def display_single_ai_recommendation(rec: AIRecommendation, name: str, emoji: str):
    """Display a single AI's recommendation"""
    if not rec:
        st.warning(f"No {name} recommendation available")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"#### {emoji} {name} Strategy")
        st.metric("Confidence", f"{rec.confidence:.0%}")
        
        if rec.narrative:
            st.info(f"**Narrative:** {rec.narrative}")
        
        if rec.captain_targets:
            st.markdown("**Captain Targets:**")
            for i, captain in enumerate(rec.captain_targets[:5], 1):
                st.write(f"{i}. {captain}")
        
        if rec.must_play:
            st.markdown("**Must Play:**")
            st.write(", ".join(rec.must_play[:5]))
        
        if rec.never_play:
            st.markdown("**Fade:**")
            st.write(", ".join(rec.never_play[:5]))
    
    with col2:
        if rec.stacks:
            st.markdown("**Key Stacks:**")
            for stack in rec.stacks[:3]:
                if isinstance(stack, dict):
                    p1 = stack.get('player1', '')
                    p2 = stack.get('player2', '')
                    st.write(f"• {p1} + {p2}")
        
        if rec.enforcement_rules:
            st.markdown("**Enforcement Rules:**")
            hard_rules = len([r for r in rec.enforcement_rules if r['type'] == 'hard'])
            soft_rules = len([r for r in rec.enforcement_rules if r['type'] == 'soft'])
            st.write(f"Hard: {hard_rules} | Soft: {soft_rules}")
        
        if hasattr(rec, 'contrarian_angles') and rec.contrarian_angles:
            st.markdown("**Contrarian Angles:**")
            for angle in rec.contrarian_angles[:2]:
                st.write(f"• {angle}")

def display_ai_synthesis(synthesis: Dict):
    """Display the synthesized AI strategy"""
    st.markdown("### 🔮 AI Synthesis & Consensus")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        consensus_captains = len([c for c, l in synthesis['captain_strategy'].items() 
                                 if l == 'consensus'])
        st.metric("Consensus Captains", consensus_captains)
    
    with col2:
        majority_captains = len([c for c, l in synthesis['captain_strategy'].items() 
                               if l == 'majority'])
        st.metric("Majority Captains", majority_captains)
    
    with col3:
        st.metric("Overall Confidence", f"{synthesis['confidence']:.0%}")
    
    with col4:
        st.metric("Enforcement Rules", len(synthesis['enforcement_rules']))
    
    # Show captain consensus details
    with st.expander("Captain Consensus Details"):
        for captain, consensus_type in synthesis['captain_strategy'].items():
            if consensus_type == 'consensus':
                st.write(f"✅ **{captain}** - All 3 AIs agree")
            elif consensus_type == 'majority':
                st.write(f"🤝 **{captain}** - 2 of 3 AIs agree")
            else:
                st.write(f"💭 **{captain}** - Single AI ({consensus_type})")
    
    # Show stacking consensus
    if synthesis.get('stacking_rules'):
        with st.expander("Stack Consensus"):
            for stack in synthesis['stacking_rules'][:5]:
                strength = stack.get('strength', 'moderate')
                support = stack.get('support', [])
                st.write(f"• {stack.get('player1')} + {stack.get('player2')} "
                        f"({strength}, {len(support)} AIs)")

def display_ai_enforcement_results(lineups_df: pd.DataFrame):
    """Display AI enforcement results for generated lineups"""
    if lineups_df.empty:
        return
    
    st.markdown("### ✅ AI Enforcement Results")
    
    # Get enforcement summary
    logger = get_logger()
    ai_summary = logger.get_ai_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enforcement_rate = ai_summary['enforcement_rate'] * 100
        st.metric("Enforcement Rate", f"{enforcement_rate:.1f}%")
    
    with col2:
        stats = ai_summary['stats']
        st.metric("Rules Enforced", f"{stats['enforced_rules']}/{stats['total_rules']}")
    
    with col3:
        consensus_used = len(lineups_df[lineups_df['AI_Strategy'] == 'ai_consensus']) if 'AI_Strategy' in lineups_df.columns else 0
        st.metric("Consensus Lineups", consensus_used)
    
    # Show which AI drove each lineup
    if 'AI_Sources' in lineups_df.columns:
        ai_usage = Counter()
        for sources in lineups_df['AI_Sources']:
            if sources:
                for source in sources:
                    ai_usage[source] += 1
        
        if ai_usage:
            st.markdown("#### AI Strategy Usage")
            for ai, count in ai_usage.most_common():
                pct = (count / len(lineups_df)) * 100
                st.write(f"• {ai}: {count} lineups ({pct:.0f}%)")

# ============================================================================
# LINEUP ANALYSIS - AI FOCUSED
# ============================================================================

def display_ai_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame, 
                              synthesis: Dict, field_size: str):
    """Display AI-driven lineup analysis"""
    if lineups_df.empty:
        st.warning("No lineups to analyze")
        return
    
    st.markdown("### 📊 AI-Driven Lineup Analysis")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. AI Strategy Distribution
    ax1 = axes[0, 0]
    if 'AI_Strategy' in lineups_df.columns:
        strategy_counts = lineups_df['AI_Strategy'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        ax1.pie(strategy_counts.values, labels=strategy_counts.index, 
               autopct='%1.0f%%', colors=colors[:len(strategy_counts)])
        ax1.set_title('AI Strategy Distribution')
    
    # 2. Captain Consensus Usage
    ax2 = axes[0, 1]
    if 'Captain' in lineups_df.columns and synthesis:
        captain_strategy = synthesis.get('captain_strategy', {})
        captain_usage = lineups_df['Captain'].value_counts()
        
        consensus_captains = [c for c, l in captain_strategy.items() if l == 'consensus']
        colors = ['green' if c in consensus_captains else 'blue' 
                 for c in captain_usage.index[:10]]
        
        ax2.bar(range(len(captain_usage.head(10))), captain_usage.head(10).values, 
               color=colors)
        ax2.set_xticks(range(len(captain_usage.head(10))))
        ax2.set_xticklabels(captain_usage.head(10).index, rotation=45, ha='right')
        ax2.set_title('Captain Usage (Green = Consensus)')
        ax2.set_ylabel('Times Used')
    
    # 3. AI Confidence Distribution
    ax3 = axes[0, 2]
    if 'Confidence' in lineups_df.columns:
        ax3.hist(lineups_df['Confidence'], bins=20, alpha=0.7, color='purple')
        ax3.axvline(lineups_df['Confidence'].mean(), color='red', 
                   linestyle='--', label=f"Mean: {lineups_df['Confidence'].mean():.2f}")
        ax3.set_xlabel('AI Confidence')
        ax3.set_ylabel('Number of Lineups')
        ax3.set_title('AI Confidence Distribution')
        ax3.legend()
    
    # 4. Enforcement Success Rate
    ax4 = axes[1, 0]
    ai_summary = get_logger().get_ai_summary()
    categories = ['Enforced', 'Violated']
    values = [ai_summary['stats']['enforced_rules'], 
             ai_summary['stats']['violated_rules']]
    colors = ['green', 'red']
    ax4.bar(categories, values, color=colors)
    ax4.set_title('AI Rule Enforcement')
    ax4.set_ylabel('Number of Rules')
    
    # 5. Consensus Level by Lineup
    ax5 = axes[1, 1]
    if 'AI_Strategy' in lineups_df.columns:
        consensus_types = ['ai_consensus', 'ai_majority', 'ai_contrarian']
        consensus_counts = [len(lineups_df[lineups_df['AI_Strategy'] == ct]) 
                           for ct in consensus_types]
        ax5.bar(consensus_types, consensus_counts, 
               color=['gold', 'silver', 'bronze'])
        ax5.set_title('Consensus Level Distribution')
        ax5.set_ylabel('Number of Lineups')
        ax5.set_xticklabels(['Consensus', 'Majority', 'Contrarian'], rotation=45)
    
    # 6. Stack Implementation
    ax6 = axes[1, 2]
    if synthesis and 'stacking_rules' in synthesis:
        implemented_stacks = 0
        total_stacks = len(synthesis['stacking_rules'])
        
        # Check which stacks were implemented (simplified check)
        for lineup_idx, row in lineups_df.iterrows():
            lineup_players = [row['Captain']] + row['FLEX']
            for stack in synthesis['stacking_rules']:
                p1 = stack.get('player1') or stack.get('stack', {}).get('player1')
                p2 = stack.get('player2') or stack.get('stack', {}).get('player2')
                if p1 in lineup_players and p2 in lineup_players:
                    implemented_stacks += 1
                    break
        
        categories = ['AI Recommended', 'Implemented']
        values = [total_stacks, implemented_stacks]
        ax6.bar(categories, values, color=['blue', 'green'])
        ax6.set_title('Stack Implementation')
        ax6.set_ylabel('Number of Stacks')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# MAIN STREAMLIT UI
# ============================================================================

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="NFL GPP AI-Chef Optimizer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🏈 NFL GPP Tournament Optimizer - AI-as-Chef Edition")
    st.markdown("*Version 6.0 - Triple AI System where AI makes all strategic decisions*")
    
    # Initialize session state
    init_ai_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🤖 AI-Chef Configuration")
        
        # AI Mode Selection
        st.markdown("### AI Enforcement Level")
        enforcement_level = st.select_slider(
            "How strictly to enforce AI decisions",
            options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
            value='Mandatory',
            help="Mandatory = AI decisions are hard constraints"
        )
        
        if enforcement_level != 'Mandatory':
            st.warning("⚠️ AI-as-Chef mode works best with Mandatory enforcement")
        
        # Contest Type
        st.markdown("### 🎯 Contest Type")
        contest_type = st.selectbox(
            "Select GPP Type",
            list(OptimizerConfig.FIELD_SIZES.keys()),
            index=2,
            help="Different contests require different AI strategies"
        )
        field_size = OptimizerConfig.FIELD_SIZES[contest_type]
        
        # Display AI strategy for field
        ai_config = OptimizerConfig.FIELD_SIZE_CONFIGS[field_size]
        st.info(f"**{field_size.replace('_', ' ').title()} AI Strategy:**\n"
               f"Enforcement: {ai_config['ai_enforcement'].value}\n"
               f"Min Leverage: {ai_config['min_leverage_players']}")
        
        st.markdown("---")
        
        # API Configuration
        st.markdown("### 🔌 AI Connection")
        api_mode = st.radio(
            "AI Input Mode",
            ["API (Automated)", "Manual (Copy/Paste)"],
            help="API mode for automatic AI analysis"
        )
        
        api_manager = None
        use_api = False
        
        if api_mode == "API (Automated)":
            api_key = st.text_input(
                "Claude API Key",
                type="password",
                placeholder="sk-ant-api03-...",
                help="Get from console.anthropic.com"
            )
            
            if api_key:
                if st.button("🔌 Connect to Claude"):
                    api_manager = ClaudeAPIManager(api_key)
                    if api_manager.client:
                        st.success("✅ Connected to Claude AI")
                        use_api = True
                    else:
                        st.error("❌ Failed to connect")
        
        st.markdown("---")
        
        # Advanced Settings
        with st.expander("⚙️ Advanced AI Settings"):
            st.markdown("### Consensus Requirements")
            
            min_ai_confidence = st.slider(
                "Minimum AI Confidence",
                0.0, 1.0, 0.3, 0.1,
                help="Minimum confidence to use AI recommendations"
            )
            
            require_consensus = st.checkbox(
                "Require AI Consensus for Captains",
                value=False,
                help="Only use captains where 2+ AIs agree"
            )
            
            st.markdown("### Lineup Generation")
            
            force_unique_captains = st.checkbox(
                "Force Unique Captains",
                value=True,
                help="Each lineup gets a different captain"
            )
            
            st.markdown("### AI Weights")
            
            gt_weight = st.slider("Game Theory Weight", 0.0, 1.0, 0.35)
            corr_weight = st.slider("Correlation Weight", 0.0, 1.0, 0.35)
            contra_weight = st.slider("Contrarian Weight", 0.0, 1.0, 0.30)
            
            # Normalize weights
            total_weight = gt_weight + corr_weight + contra_weight
            if total_weight > 0:
                OptimizerConfig.AI_WEIGHTS = {
                    AIStrategistType.GAME_THEORY: gt_weight / total_weight,
                    AIStrategistType.CORRELATION: corr_weight / total_weight,
                    AIStrategistType.CONTRARIAN_NARRATIVE: contra_weight / total_weight
                }
        
        # Debug Panel
        with st.expander("🐛 Debug & Monitoring"):
            if st.button("Show AI Decision Log"):
                logger = get_logger()
                logger.display_ai_enforcement()
            
            if st.button("Show Performance Metrics"):
                perf = get_performance_monitor()
                perf.display_metrics()
            
            if st.button("Export Session"):
                if st.session_state.get('optimization_history'):
                    session_data = json.dumps(
                        st.session_state['optimization_history'][-1],
                        default=str
                    )
                    st.download_button(
                        "Download Session",
                        data=session_data,
                        file_name=f"ai_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Main Content Area
    st.markdown("## 📊 Data & Game Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload DraftKings CSV",
        type="csv",
        help="Export from DraftKings Showdown contest"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Data processing (simplified for brevity)
        required_cols = ['first_name', 'last_name', 'position', 'team', 'salary', 'ppg_projection']
        
        # Create player names
        df['Player'] = (df['first_name'].fillna('') + ' ' + df['last_name'].fillna('')).str.strip()
        
        # Rename columns
        df = df.rename(columns={
            'position': 'Position',
            'team': 'Team',
            'salary': 'Salary',
            'ppg_projection': 'Projected_Points'
        })
        
        # Add ownership (would be from projections normally)
        df['Ownership'] = 5.0  # Default
        
        # Game configuration
        st.markdown("### ⚙️ Game Setup")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            teams = st.text_input("Teams", "SEA vs LAR")
        with col2:
            total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5)
        with col3:
            spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)
        with col4:
            weather = st.selectbox("Weather", ["Clear", "Wind", "Rain", "Snow"])
        
        game_info = {
            'teams': teams,
            'total': total,
            'spread': spread,
            'weather': weather
        }
        
        # Display player pool
        st.markdown("### 📋 Player Pool")
        st.dataframe(
            df[['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']].head(20),
            use_container_width=True
        )
        
        # Optimization section
        st.markdown("---")
        st.markdown("## 🚀 AI-Driven Lineup Generation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_lineups = st.number_input(
                "Number of Lineups",
                5, 150, 20, 5,
                help="AI will determine strategy distribution"
            )
        
        with col2:
            st.metric("Field Size", field_size.replace('_', ' ').title())
            st.metric("AI Mode", enforcement_level)
        
        with col3:
            generate_button = st.button(
                "🤖 Generate AI-Driven Lineups",
                type="primary",
                use_container_width=True
            )
        
        if generate_button:
            # Initialize optimizer
            optimizer = AIChefGPPOptimizer(df, game_info, field_size, api_manager)
            
            # Get AI strategies
            with st.spinner("🤖 Consulting Triple AI System..."):
                ai_recommendations = optimizer.get_triple_ai_strategies(use_api=use_api)
            
            if not ai_recommendations:
                st.error("Failed to get AI recommendations")
                st.stop()
            
            # Display AI recommendations
            display_ai_recommendations(ai_recommendations)
            
            # Synthesize strategies
            with st.spinner("🔮 Synthesizing AI strategies..."):
                ai_strategy = optimizer.synthesize_ai_strategies(ai_recommendations)
            
            # Display synthesis
            display_ai_synthesis(ai_strategy['synthesis'])
            
            # Generate lineups
            with st.spinner(f"⚡ Generating {num_lineups} AI-enforced lineups..."):
                lineups_df = optimizer.generate_ai_driven_lineups(num_lineups, ai_strategy)
            
            if not lineups_df.empty:
                # Store in session
                st.session_state['lineups_df'] = lineups_df
                st.session_state['ai_synthesis'] = ai_strategy['synthesis']
                st.session_state['df'] = df
                
                # Save session
                save_ai_optimization_session(
                    lineups_df,
                    ai_strategy['synthesis'],
                    {
                        'num_lineups': num_lineups,
                        'field_size': field_size,
                        'enforcement': enforcement_level
                    }
                )
                
                # Display enforcement results
                display_ai_enforcement_results(lineups_df)
    
    # Display results if available
    if 'lineups_df' in st.session_state and not st.session_state['lineups_df'].empty:
        lineups_df = st.session_state['lineups_df']
        synthesis = st.session_state.get('ai_synthesis', {})
        df = st.session_state.get('df', pd.DataFrame())
        
        st.markdown("---")
        st.markdown("## 📊 AI Optimization Results")
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Lineups", "📊 Analysis", "🤖 AI Metrics", 
            "📈 Visualizations", "💾 Export"
        ])
        
        with tab1:
            st.markdown("### 🏆 AI-Generated Lineups")
            
            # Display lineups
            for idx, row in lineups_df.head(10).iterrows():
                with st.expander(f"Lineup {row['Lineup']} - {row.get('AI_Strategy', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Roster:**")
                        st.write(f"CPT: {row['Captain']}")
                        for player in row['FLEX']:
                            st.write(f"FLEX: {player}")
                    
                    with col2:
                        st.markdown("**Metrics:**")
                        st.write(f"Projected: {row['Projected']:.1f}")
                        st.write(f"Salary: ${row['Salary']:,}")
                        st.write(f"Ownership: {row.get('Total_Ownership', 0):.1f}%")
                    
                    with col3:
                        st.markdown("**AI Info:**")
                        st.write(f"Strategy: {row.get('AI_Strategy', 'N/A')}")
                        st.write(f"Confidence: {row.get('Confidence', 0):.0%}")
                        if row.get('AI_Sources'):
                            st.write(f"Sources: {', '.join(row['AI_Sources'])}")
        
        with tab2:
            display_ai_lineup_analysis(lineups_df, df, synthesis, field_size)
        
        with tab3:
            st.markdown("### 🤖 AI Performance Metrics")
            logger = get_logger()
            logger.display_ai_enforcement()
            
            # Show synthesis summary
            st.markdown("#### AI Synthesis Summary")
            synthesis_summary = st.session_state['ai_synthesis']
            st.json({
                'confidence': synthesis_summary.get('confidence'),
                'total_captains': len(synthesis_summary.get('captain_strategy', {})),
                'total_stacks': len(synthesis_summary.get('stacking_rules', [])),
                'enforcement_rules': len(synthesis_summary.get('enforcement_rules', []))
            })
        
        with tab4:
            st.markdown("### 📈 AI Strategy Visualizations")
            
            # Create custom visualizations for AI strategies
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Captain consensus visualization
            ax1 = axes[0]
            captain_strategy = synthesis.get('captain_strategy', {})
            consensus_types = Counter(captain_strategy.values())
            ax1.pie(consensus_types.values(), labels=consensus_types.keys(),
                   autopct='%1.0f%%', colors=['gold', 'silver', 'bronze', 'gray'])
            ax1.set_title('Captain Consensus Distribution')
            
            # Stack implementation
            ax2 = axes[1]
            stack_strengths = Counter([s['strength'] for s in synthesis.get('stacking_rules', [])])
            ax2.bar(stack_strengths.keys(), stack_strengths.values())
            ax2.set_title('Stack Strength Distribution')
            ax2.set_xlabel('Strength')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab5:
            st.markdown("### 💾 Export AI-Optimized Lineups")
            
            # DraftKings format
            dk_lineups = []
            for idx, row in lineups_df.iterrows():
                flex_players = row['FLEX'] if isinstance(row['FLEX'], list) else []
                dk_lineups.append({
                    'CPT': row['Captain'],
                    'FLEX 1': flex_players[0] if len(flex_players) > 0 else '',
                    'FLEX 2': flex_players[1] if len(flex_players) > 1 else '',
                    'FLEX 3': flex_players[2] if len(flex_players) > 2 else '',
                    'FLEX 4': flex_players[3] if len(flex_players) > 3 else '',
                    'FLEX 5': flex_players[4] if len(flex_players) > 4 else ''
                })
            
            dk_df = pd.DataFrame(dk_lineups)
            
            csv = dk_df.to_csv(index=False)
            st.download_button(
                label="📥 Download DraftKings CSV",
                data=csv,
                file_name=f"ai_chef_lineups_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # AI Strategy export
            strategy_export = {
                'timestamp': datetime.now().isoformat(),
                'field_size': field_size,
                'synthesis': synthesis,
                'lineups': lineups_df.to_dict('records')
            }
            
            st.download_button(
                label="🤖 Download AI Strategy",
                data=json.dumps(strategy_export, default=str),
                file_name=f"ai_strategy_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

