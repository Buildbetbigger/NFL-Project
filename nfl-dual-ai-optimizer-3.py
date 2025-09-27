# NFL GPP DUAL-AI OPTIMIZER - PART 1: CONFIGURATION & MONITORING
# Version 6.0 - AI-as-Chef Architecture with Triple AI System

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
    """GPP Strategy Types - Now AI-Driven"""
    AI_CONSENSUS = 'ai_consensus'        # All 3 AIs agree
    AI_MAJORITY = 'ai_majority'          # 2 of 3 AIs agree  
    AI_CONTRARIAN = 'ai_contrarian'      # Following contrarian AI specifically
    AI_CORRELATION = 'ai_correlation'    # Following correlation AI specifically
    AI_GAME_THEORY = 'ai_game_theory'    # Following game theory AI specifically
    AI_MIXED = 'ai_mixed'                # Mixing different AI perspectives
    LEVERAGE = 'leverage'                # Legacy - will be AI-driven
    CONTRARIAN = 'contrarian'            # Legacy - will be AI-driven

class AIEnforcementLevel(Enum):
    """How strictly to enforce AI decisions"""
    MANDATORY = 'mandatory'              # AI decisions are hard constraints
    STRONG = 'strong'                    # AI decisions heavily weighted
    MODERATE = 'moderate'                # AI decisions moderately weighted
    SUGGESTION = 'suggestion'            # AI decisions as hints only

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
    ROSTER_SIZE = 6  # 1 CPT + 5 FLEX
    
    # Default ownership when unknown (pre-lock)
    DEFAULT_OWNERSHIP = 5.0
    
    # AI Configuration
    AI_ENFORCEMENT_MODE = AIEnforcementLevel.MANDATORY
    REQUIRE_AI_FOR_GENERATION = True  # If True, won't generate without AI input
    MIN_AI_CONFIDENCE = 0.3  # Minimum confidence to use AI recommendations
    
    # Triple AI Weights (how much each AI contributes)
    AI_WEIGHTS = {
        AIStrategistType.GAME_THEORY: 0.35,
        AIStrategistType.CORRELATION: 0.35,
        AIStrategistType.CONTRARIAN_NARRATIVE: 0.30
    }
    
    # AI Consensus Requirements
    AI_CONSENSUS_THRESHOLDS = {
        'captain': 2,  # How many AIs must agree for consensus captain
        'must_play': 3,  # All 3 must agree for "must play"
        'never_play': 2,  # 2 must agree for "never play"
    }
    
    # Field size configurations - Now with AI strategy distribution
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
    
    # AI Strategy Requirements - What each AI strategy type enforces
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
    
    # Ownership bucket thresholds
    OWNERSHIP_BUCKETS = {
        'mega_chalk': 35,
        'chalk': 20,
        'pivot': 10,
        'leverage': 5,
        'super_leverage': 0
    }
    
    # Contest types to field sizes
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
# MONITORING AND LOGGING - AI-FOCUSED
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
            'recent_decisions': self.ai_decisions[-10:]
        }

class GlobalLogger:
    """Enhanced logger with AI decision tracking"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.entries = []
        self.ai_tracker = AIDecisionTracker()
        self.verbose = False
        self.log_to_file = False
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with level"""
        entry = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        self.entries.append(entry)
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            print(f"[{timestamp}] {level}: {message}")
            
        if self.log_to_file:
            self._write_to_file(entry)
    
    def log_ai_decision(self, decision_type: str, ai_source: str, 
                       enforced: bool, details: Dict):
        """Log an AI decision"""
        self.ai_tracker.track_decision(decision_type, ai_source, enforced, details)
        message = f"AI Decision [{ai_source}]: {decision_type} - {'ENFORCED' if enforced else 'VIOLATED'}"
        self.log(message, "AI_DECISION")
    
    def log_ai_consensus(self, consensus_type: str, ais_agreeing: List[str], 
                        decision: str):
        """Log AI consensus"""
        self.ai_tracker.track_consensus(consensus_type, ais_agreeing)
        message = f"AI Consensus ({len(ais_agreeing)}/3): {decision}"
        self.log(message, "AI_CONSENSUS")
    
    def log_optimization_start(self, num_lineups: int, field_size: str, 
                              settings: Dict):
        """Log optimization start with AI focus"""
        self.log(f"Starting AI-driven optimization: {num_lineups} lineups for {field_size}", "INFO")
        self.log(f"AI Enforcement: {settings.get('ai_enforcement', 'MANDATORY')}", "INFO")
        self.log(f"Settings: {settings}", "DEBUG")
    
    def log_lineup_generation(self, strategy: str, lineup_num: int, 
                             status: str, ai_rules_enforced: int = 0):
        """Log lineup generation with AI enforcement tracking"""
        message = f"Lineup {lineup_num} ({strategy}): {status}"
        if ai_rules_enforced > 0:
            message += f" - {ai_rules_enforced} AI rules enforced"
        self.log(message, "DEBUG" if status == "SUCCESS" else "WARNING")
    
    def get_ai_summary(self) -> Dict:
        """Get AI tracking summary"""
        return self.ai_tracker.get_summary()
    
    def display_ai_enforcement(self):
        """Display AI enforcement statistics in Streamlit"""
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
            st.metric("Consensus Decisions", stats['consensus_decisions'])
        
        # Show AI performance
        st.markdown("#### AI Performance")
        for ai_type, perf in summary['ai_performance'].items():
            if perf['suggestions'] > 0:
                usage_rate = (perf['used'] / perf['suggestions']) * 100
                st.write(f"**{ai_type.value}**: {usage_rate:.1f}% usage rate")
    
    def export_logs(self) -> str:
        """Export logs including AI decisions"""
        output = "=== OPTIMIZATION LOG ===\n\n"
        for entry in self.entries:
            timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            output += f"[{timestamp}] {entry['level']}: {entry['message']}\n"
        
        output += "\n=== AI DECISION SUMMARY ===\n"
        summary = self.get_ai_summary()
        output += f"Enforcement Rate: {summary['enforcement_rate']*100:.1f}%\n"
        output += f"Total AI Rules: {summary['stats']['total_rules']}\n"
        output += f"Enforced: {summary['stats']['enforced_rules']}\n"
        output += f"Violated: {summary['stats']['violated_rules']}\n"
        
        return output

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
        self.timers = {}
        self.counters = {}
        self.ai_metrics = {
            'ai_api_calls': 0,
            'ai_response_time': [],
            'ai_cache_hits': 0,
            'ai_enforcement_time': []
        }
    
    def start_timer(self, name: str):
        self.timers[name] = {'start': datetime.now()}
    
    def stop_timer(self, name: str) -> float:
        if name in self.timers and 'start' in self.timers[name]:
            elapsed = (datetime.now() - self.timers[name]['start']).total_seconds()
            if 'total' not in self.timers[name]:
                self.timers[name]['total'] = 0
                self.timers[name]['count'] = 0
            self.timers[name]['total'] += elapsed
            self.timers[name]['count'] += 1
            self.timers[name]['average'] = self.timers[name]['total'] / self.timers[name]['count']
            
            # Track AI-specific timings
            if 'ai' in name.lower():
                self.ai_metrics['ai_response_time'].append(elapsed)
            
            return elapsed
        return 0.0
    
    def increment_counter(self, name: str, value: int = 1):
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
        # Track AI-specific counters
        if 'ai' in name.lower():
            if 'cache' in name.lower():
                self.ai_metrics['ai_cache_hits'] += value
            elif 'api' in name.lower():
                self.ai_metrics['ai_api_calls'] += value
    
    def get_metrics(self) -> Dict:
        """Get all metrics including AI-specific ones"""
        avg_ai_response = (
            np.mean(self.ai_metrics['ai_response_time']) 
            if self.ai_metrics['ai_response_time'] else 0
        )
        
        return {
            'timers': self.timers,
            'counters': self.counters,
            'ai_metrics': {
                'api_calls': self.ai_metrics['ai_api_calls'],
                'cache_hits': self.ai_metrics['ai_cache_hits'],
                'avg_response_time': avg_ai_response,
                'cache_hit_rate': (
                    self.ai_metrics['ai_cache_hits'] / 
                    (self.ai_metrics['ai_api_calls'] + self.ai_metrics['ai_cache_hits'])
                    if (self.ai_metrics['ai_api_calls'] + self.ai_metrics['ai_cache_hits']) > 0
                    else 0
                )
            }
        }
    
    def display_metrics(self):
        """Display performance metrics with AI focus"""
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
        st.markdown("#### Optimization Performance")
        for name, stats in metrics['timers'].items():
            if 'average' in stats:
                st.write(f"**{name}**: {stats['average']:.3f}s avg ({stats['count']} ops)")

# ============================================================================
# SINGLETON GETTERS
# ============================================================================

def get_logger() -> GlobalLogger:
    """Get the global logger instance"""
    return GlobalLogger()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return PerformanceMonitor()

# ============================================================================
# AI RECOMMENDATION DATACLASS - ENHANCED
# ============================================================================

@dataclass
class AIRecommendation:
    """Enhanced AI recommendation with enforcement rules"""
    captain_targets: List[str]
    must_play: List[str]  # Players that must be included
    never_play: List[str]  # Players to avoid
    stacks: List[Dict]
    key_insights: List[str]
    confidence: float
    enforcement_rules: List[Dict]  # Specific rules to enforce
    narrative: str  # AI's narrative/reasoning
    source_ai: AIStrategistType
    
    # Optional fields for specific AI types
    contrarian_angles: Optional[List[str]] = None
    correlation_matrix: Optional[Dict] = None
    ownership_leverage: Optional[Dict] = None
    
    def get_hard_constraints(self) -> List[str]:
        """Get constraints that MUST be enforced"""
        constraints = []
        for rule in self.enforcement_rules:
            if rule.get('type') == 'hard':
                constraints.append(rule['constraint'])
        return constraints
    
    def get_soft_constraints(self) -> List[Tuple[str, float]]:
        """Get constraints with weights for soft enforcement"""
        constraints = []
        for rule in self.enforcement_rules:
            if rule.get('type') == 'soft':
                constraints.append((rule['constraint'], rule.get('weight', 1.0)))
        return constraints

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

# NFL GPP DUAL-AI OPTIMIZER - PART 4: MAIN OPTIMIZER & LINEUP GENERATION (AI-AS-CHEF VERSION)
# Version 6.0 - AI Drives Everything, Optimizer Just Executes

import pandas as pd
import numpy as np
import pulp
from typing import Dict, List, Tuple, Optional, Set, Any
import streamlit as st
import json

# ============================================================================
# AI-DRIVEN GPP OPTIMIZER
# ============================================================================

class AIChefGPPOptimizer:
    """Main optimizer where AI is the chef and optimization is just execution"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field', 
                 api_manager: ClaudeAPIManager = None):
        self.df = df
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager
        
        # Initialize the three AI strategists
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.contrarian_ai = GPPContrarianNarrativeStrategist(api_manager)
        
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
        
        # Tracking
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.ai_decisions_log = []
        self.optimization_log = []
    
    def get_triple_ai_strategies(self, use_api: bool = True) -> Dict[AIStrategistType, AIRecommendation]:
        """Get strategies from all three AIs - REQUIRED for lineup generation"""
        
        self.logger.log("Getting strategies from three AI strategists", "INFO")
        recommendations = {}
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode - automatic
            with st.spinner("🎯 Game Theory AI analyzing..."):
                self.perf_monitor.start_timer("ai_game_theory")
                recommendations[AIStrategistType.GAME_THEORY] = self.game_theory_ai.get_recommendation(
                    self.df, self.game_info, self.field_size, use_api=True
                )
                self.perf_monitor.stop_timer("ai_game_theory")
            
            with st.spinner("🔗 Correlation AI analyzing..."):
                self.perf_monitor.start_timer("ai_correlation")
                recommendations[AIStrategistType.CORRELATION] = self.correlation_ai.get_recommendation(
                    self.df, self.game_info, self.field_size, use_api=True
                )
                self.perf_monitor.stop_timer("ai_correlation")
            
            with st.spinner("🎭 Contrarian AI analyzing..."):
                self.perf_monitor.start_timer("ai_contrarian")
                recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self.contrarian_ai.get_recommendation(
                    self.df, self.game_info, self.field_size, use_api=True
                )
                self.perf_monitor.stop_timer("ai_contrarian")
        
        else:
            # Manual mode - collect all three responses
            st.subheader("📝 Triple AI Strategy Input (REQUIRED)")
            
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
            
            # Parse all responses
            recommendations[AIStrategistType.GAME_THEORY] = self.game_theory_ai.parse_response(
                responses[AIStrategistType.GAME_THEORY], self.df, self.field_size
            )
            recommendations[AIStrategistType.CORRELATION] = self.correlation_ai.parse_response(
                responses[AIStrategistType.CORRELATION], self.df, self.field_size
            )
            recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self.contrarian_ai.parse_response(
                responses[AIStrategistType.CONTRARIAN_NARRATIVE], self.df, self.field_size
            )
        
        # Check if we got valid recommendations
        valid_count = sum(1 for rec in recommendations.values() if rec.confidence > 0.3)
        
        if valid_count == 0 and OptimizerConfig.REQUIRE_AI_FOR_GENERATION:
            st.error("❌ No valid AI strategies received. Cannot generate lineups in AI-as-Chef mode.")
            st.info("Please provide valid AI responses or switch to API mode.")
            return {}
        
        # Log AI decisions
        for ai_type, rec in recommendations.items():
            self.logger.log_ai_decision(
                "strategy_received",
                ai_type.value,
                True,
                {
                    'captains': len(rec.captain_targets),
                    'must_play': len(rec.must_play),
                    'confidence': rec.confidence
                }
            )
        
        return recommendations
    
    def _get_manual_ai_input(self, ai_name: str, strategist) -> str:
        """Get manual AI input with validation"""
        with st.expander(f"View {ai_name} Prompt"):
            st.text_area(f"Copy this:", 
                       value=strategist.generate_prompt(self.df, self.game_info, self.field_size),
                       height=250, key=f"{ai_name}_prompt_display")
        
        response = st.text_area(f"Paste {ai_name} Response (JSON):", 
                               height=200, 
                               key=f"{ai_name}_response",
                               value='{}')
        
        # Validate JSON
        if response and response != '{}':
            try:
                json.loads(response)
                st.success(f"✅ Valid {ai_name} JSON")
            except:
                st.error(f"❌ Invalid {ai_name} JSON")
        
        return response
    
    def synthesize_ai_strategies(self, recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Synthesize three AI perspectives into unified strategy"""
        
        self.logger.log("Synthesizing triple AI strategies", "INFO")
        
        # Use synthesis engine
        synthesis = self.synthesis_engine.synthesize_recommendations(
            recommendations.get(AIStrategistType.GAME_THEORY),
            recommendations.get(AIStrategistType.CORRELATION),
            recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE)
        )
        
        # Create enforcement rules
        enforcement_rules = self.enforcement_engine.create_enforcement_rules(recommendations)
        
        # Validate rules are feasible
        validation = AIConfigValidator.validate_ai_requirements(enforcement_rules, self.df)
        
        if not validation['is_valid']:
            st.warning("⚠️ Some AI requirements cannot be satisfied:")
            for error in validation['errors']:
                st.write(f"  - {error}")
            
            # Adjust rules if needed
            if validation.get('adjustments'):
                for adjustment in validation['adjustments']:
                    st.info(f"Adjustment: {adjustment}")
        
        return {
            'synthesis': synthesis,
            'enforcement_rules': enforcement_rules,
            'validation': validation
        }
    
    def generate_ai_driven_lineups(self, num_lineups: int, ai_strategy: Dict) -> pd.DataFrame:
        """Generate lineups following AI directives"""
        
        self.perf_monitor.start_timer("total_optimization")
        self.logger.log_optimization_start(num_lineups, self.field_size, {
            'mode': 'AI-as-Chef',
            'enforcement': self.enforcement_engine.enforcement_level.value
        })
        
        # Extract components
        synthesis = ai_strategy['synthesis']
        enforcement_rules = ai_strategy['enforcement_rules']
        
        # Show AI consensus
        self._display_ai_consensus(synthesis)
        
        # Get player data
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Apply AI modifications to projections
        ai_adjusted_points = self._apply_ai_adjustments(points, synthesis)
        
        # Get strategy distribution based on AI consensus
        consensus_level = self._determine_consensus_level(synthesis)
        strategy_distribution = AIConfigValidator.get_ai_strategy_distribution(
            self.field_size, num_lineups, consensus_level
        )
        
        self.logger.log(f"AI Strategy distribution: {strategy_distribution}", "INFO")
        
        all_lineups = []
        used_captains = set()
        
        # Generate lineups by AI strategy
        for strategy, count in strategy_distribution.items():
            strategy_name = strategy if isinstance(strategy, str) else strategy.value
            self.logger.log(f"Generating {count} lineups with strategy: {strategy_name}", "INFO")
            
            for i in range(count):
                lineup_num = len(all_lineups) + 1
                
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
                    
                    if is_valid:
                        all_lineups.append(lineup)
                        used_captains.add(lineup['Captain'])
                        
                        self.logger.log_lineup_generation(
                            strategy_name, lineup_num, "SUCCESS", 
                            len(enforcement_rules.get('hard_constraints', []))
                        )
                    else:
                        self.logger.log(f"Lineup {lineup_num} violated AI rules: {violations}", "WARNING")
                        self.optimization_log.append(f"Lineup {lineup_num}: {violations}")
                else:
                    self.logger.log(f"Failed to generate lineup {lineup_num}", "WARNING")
        
        # Check results
        total_time = self.perf_monitor.stop_timer("total_optimization")
        self.logger.log_optimization_end(len(all_lineups), total_time)
        
        if len(all_lineups) == 0:
            st.error("❌ No valid lineups generated with AI constraints")
            self._display_optimization_issues()
            return pd.DataFrame()
        
        if len(all_lineups) < num_lineups:
            st.warning(f"Generated {len(all_lineups)}/{num_lineups} AI-compliant lineups")
        else:
            st.success(f"✅ Generated {len(all_lineups)} AI-driven lineups!")
        
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
            
            # AI-modified objective function
            player_weights = synthesis.get('player_rankings', {})
            
            objective = pulp.lpSum([
                points[p] * player_weights.get(p, 1.0) * flex[p] +
                1.5 * points[p] * player_weights.get(p, 1.0) * captain[p]
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
                salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                for p in players
            ]) <= OptimizerConfig.SALARY_CAP
            
            # Team constraint
            for team in set(teams.values()):
                team_players = [p for p in players if teams.get(p) == team]
                if team_players:
                    model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
            
            # ENFORCE AI HARD CONSTRAINTS
            for constraint in enforcement_rules.get('hard_constraints', []):
                if constraint.get('rule') == 'must_include':
                    player = constraint['player']
                    if player in players:
                        model += flex[player] + captain[player] >= 1
                
                elif constraint.get('rule') == 'must_exclude':
                    player = constraint['player']
                    if player in players:
                        model += flex[player] + captain[player] == 0
                
                elif constraint.get('rule') == 'captain_from_list':
                    valid_captains = [p for p in constraint['players'] if p in players]
                    if valid_captains:
                        model += pulp.lpSum([captain[p] for p in valid_captains]) == 1
            
            # ENFORCE AI CAPTAIN REQUIREMENTS
            captain_requirements = enforcement_rules.get('variable_locks', {}).get('captain', [])
            if captain_requirements:
                valid_ai_captains = [c for c in captain_requirements if c in players and c not in used_captains]
                if valid_ai_captains:
                    model += pulp.lpSum([captain[c] for c in valid_ai_captains]) == 1
            
            # ENFORCE AI STACKING RULES
            for stack_rule in synthesis.get('stacking_rules', [])[:2]:  # Top 2 stacks
                if stack_rule['strength'] == 'strong':
                    p1 = stack_rule.get('player1') or stack_rule.get('stack', {}).get('player1')
                    p2 = stack_rule.get('player2') or stack_rule.get('stack', {}).get('player2')
                    
                    if p1 in players and p2 in players:
                        # At least one must be in lineup
                        model += flex[p1] + captain[p1] + flex[p2] + captain[p2] >= 1
            
            # ENFORCE SOFT CONSTRAINTS WITH WEIGHTS
            # (These influence the objective but don't force compliance)
            soft_bonus = 0
            for constraint in enforcement_rules.get('soft_constraints', []):
                player = constraint.get('player')
                weight = constraint.get('weight', 0.5)
                
                if player in players:
                    if constraint.get('rule') == 'should_include':
                        soft_bonus += points[player] * weight * (flex[player] + captain[player])
                    elif constraint.get('rule') == 'should_exclude':
                        soft_bonus -= points[player] * weight * (flex[player] + captain[player])
            
            if soft_bonus != 0:
                model += objective + soft_bonus
            
            # Strategy-specific AI requirements
            if strategy == 'ai_consensus':
                # Must use consensus captain
                consensus_captains = [c for c, level in synthesis['captain_strategy'].items() 
                                    if level == 'consensus' and c in players]
                if consensus_captains:
                    model += pulp.lpSum([captain[c] for c in consensus_captains]) == 1
            
            elif strategy == 'ai_contrarian':
                # Use contrarian captain
                contrarian_captains = [c for c, level in synthesis['captain_strategy'].items() 
                                      if level == 'contrarian_narrative' and c in players]
                if contrarian_captains:
                    model += pulp.lpSum([captain[c] for c in contrarian_captains]) == 1
            
            # Unique captain constraint
            if used_captains:
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
                    total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
                    total_proj = sum(points[p] for p in flex_picks) + 1.5 * points[captain_pick]
                    total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                    
                    # Check which AI recommended this captain
                    ai_sources = []
                    for ai_type, level in synthesis['captain_strategy'].items():
                        if captain_pick == ai_type:
                            ai_sources.append(level)
                    
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
                        'AI_Sources': ai_sources,
                        'AI_Enforced': True,
                        'Confidence': synthesis.get('confidence', 0.5)
                    }
            
            return None
            
        except Exception as e:
            self.logger.log(f"Error building lineup {lineup_num}: {e}", "ERROR")
            return None
    
    def _apply_ai_adjustments(self, points: Dict, synthesis: Dict) -> Dict:
        """Apply AI-recommended adjustments to projections"""
        adjusted = points.copy()
        
        # Apply player rankings as multipliers
        rankings = synthesis.get('player_rankings', {})
        
        for player, score in rankings.items():
            if player in adjusted:
                # Convert score to multiplier (higher score = higher multiplier)
                multiplier = 1.0 + (score * 0.2)  # Max 20% boost/penalty
                adjusted[player] *= multiplier
        
        return adjusted
    
    def _determine_consensus_level(self, synthesis: Dict) -> str:
        """Determine the level of AI consensus"""
        captain_strategy = synthesis.get('captain_strategy', {})
        
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
    
    def _display_ai_consensus(self, synthesis: Dict):
        """Display AI consensus analysis"""
        st.markdown("### 🤖 AI Consensus Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            consensus_captains = len([c for c, l in synthesis['captain_strategy'].items() if l == 'consensus'])
            st.metric("Consensus Captains", consensus_captains)
        
        with col2:
            strong_stacks = len([s for s in synthesis['stacking_rules'] if s['strength'] == 'strong'])
            st.metric("Strong Stacks", strong_stacks)
        
        with col3:
            st.metric("AI Confidence", f"{synthesis['confidence']:.0%}")
        
        with col4:
            enforcement = len(synthesis.get('enforcement_rules', []))
            st.metric("Enforcement Rules", enforcement)
        
        # Show narrative
        if synthesis.get('narrative'):
            st.info(f"**AI Narrative:** {synthesis['narrative']}")
    
    def _display_optimization_issues(self):
        """Display optimization issues for debugging"""
        if self.optimization_log:
            with st.expander("⚠️ Optimization Issues", expanded=True):
                for issue in self.optimization_log[-10:]:
                    st.write(f"- {issue}")

# ============================================================================
# CAPTAIN PIVOT GENERATOR (AI-Enhanced)
# ============================================================================

class GPPCaptainPivotGenerator:
    """Generate captain pivots with AI guidance"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def generate_ai_guided_pivots(self, lineup: Dict, df: pd.DataFrame, 
                                 synthesis: Dict, max_pivots: int = 5) -> List[Dict]:
        """Generate pivots following AI recommendations"""
        
        pivots = []
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
                        'Pivot_Type': f"AI-{captain_strategy[new_captain]}",
                        'AI_Recommended': True
                    }
                    
                    pivots.append(pivot)
                    
                    if len(pivots) >= max_pivots:
                        break
        
        return pivots

# ============================================================================
# FALLBACK OPTIMIZER (When AI Fails)
# ============================================================================

class FallbackOptimizer:
    """Emergency fallback when AI is unavailable but user insists"""
    
    @staticmethod
    def generate_statistical_lineups(df: pd.DataFrame, num_lineups: int, 
                                    field_size: str) -> pd.DataFrame:
        """Generate lineups using pure statistics (no AI)"""
        
        st.error("⚠️ AI-as-Chef mode requires AI input. Using emergency statistical fallback.")
        st.warning("These lineups are NOT AI-optimized and may underperform.")
        
        # This should rarely be used in AI-as-Chef mode
        # Implementation would be similar to the original optimizer
        # but without any AI influence
        
        return pd.DataFrame()  # Return empty for now

# NFL GPP DUAL-AI OPTIMIZER - PART 5: MAIN UI AND HELPER FUNCTIONS (COMPLETE CORRECTED)
# With Performance Optimizations and Caching

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

# ============================================================================
# PERFORMANCE OPTIMIZATION - CACHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_correlations_cached(df_json: str, game_info_json: str, field_size: str) -> Dict:
    """Cache correlation calculations"""
    df = pd.read_json(df_json)
    game_info = json.loads(game_info_json)
    
    correlation_engine = GPPCorrelationEngine()
    correlations = correlation_engine.calculate_gpp_correlations(df, game_info)
    stacks = correlation_engine.identify_gpp_stacks(df, correlations, field_size)
    
    return {
        'correlations': correlations,
        'stacks': stacks
    }

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_simulations_cached(lineup_json: str, df_json: str, correlations_json: str, 
                          n_sims: int, field_size: str) -> Dict:
    """Cache simulation results"""
    lineup = json.loads(lineup_json)
    df = pd.read_json(df_json)
    correlations = json.loads(correlations_json)
    
    # Convert correlation keys back to tuples
    correlations_fixed = {}
    for key, value in correlations.items():
        if isinstance(key, str) and ',' in key:
            players = key.strip('()').split(', ')
            correlations_fixed[tuple(players)] = value
        else:
            correlations_fixed[key] = value
    
    simulator = GPPTournamentSimulator()
    return simulator.simulate_gpp_tournament(lineup, df, correlations_fixed, n_sims, field_size)

@st.cache_data(ttl=3600)
def validate_player_pool_cached(df_json: str, field_size: str) -> Dict:
    """Cache player pool validation results"""
    df = pd.read_json(df_json)
    return ConfigValidator.validate_player_pool(df, field_size)

@st.cache_data(ttl=7200)  # Cache for 2 hours
def analyze_ownership_distribution(df_json: str) -> Dict:
    """Cache ownership distribution analysis"""
    df = pd.read_json(df_json)
    bucket_manager = OwnershipBucketManager()
    buckets = bucket_manager.categorize_players(df)
    
    return {
        'buckets': buckets,
        'distribution': {k: len(v) for k, v in buckets.items()},
        'total_players': len(df)
    }

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables for persistence"""
    if 'optimization_history' not in st.session_state:
        st.session_state['optimization_history'] = []
    if 'saved_lineups' not in st.session_state:
        st.session_state['saved_lineups'] = []
    if 'api_cache' not in st.session_state:
        st.session_state['api_cache'] = {}
    if 'performance_metrics' not in st.session_state:
        st.session_state['performance_metrics'] = {}

def save_optimization_session(lineups_df: pd.DataFrame, settings: Dict):
    """Save optimization session to history"""
    session = {
        'timestamp': datetime.now(),
        'lineups': lineups_df.to_dict('records'),
        'settings': settings,
        'field_size': st.session_state.get('field_size', 'large_field')
    }
    st.session_state['optimization_history'].append(session)
    
    # Limit history to last 10 sessions
    if len(st.session_state['optimization_history']) > 10:
        st.session_state['optimization_history'].pop(0)

# ============================================================================
# DATA VALIDATION AND INTEGRITY CHECKS
# ============================================================================

def validate_lineup_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Comprehensive data validation for lineup generation with performance optimization"""
    
    # Try to use cached validation if available
    df_json = df.to_json()
    df_hash = hashlib.md5(df_json.encode()).hexdigest()
    
    # Check if we've validated this data recently
    cache_key = f"validation_{df_hash}"
    if cache_key in st.session_state.get('api_cache', {}):
        cached_result = st.session_state['api_cache'][cache_key]
        if (datetime.now() - cached_result['timestamp']).seconds < 300:  # 5 minute cache
            return cached_result['result']
    
    issues = []
    warnings = []
    data_stats = {}
    
    # Basic size check
    data_stats['total_players'] = len(df)
    if len(df) < 12:
        issues.append(f"Critical: Only {len(df)} players found - need at least 12 for valid lineups")
    elif len(df) < 20:
        warnings.append(f"Warning: Only {len(df)} players - limited lineup diversity possible")
    
    # Check for required columns
    required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Critical: Missing required columns: {missing_columns}")
        return False, issues, data_stats
    
    # Check for missing values
    for col in required_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"Critical: {null_count} missing values in {col}")
    
    # Validate data types and ranges
    if not df['Salary'].isna().all():
        salary_issues = df[df['Salary'] < OptimizerConfig.MIN_SALARY]
        if len(salary_issues) > 0:
            issues.append(f"Critical: {len(salary_issues)} players with salary below minimum (${OptimizerConfig.MIN_SALARY})")
        
        data_stats['avg_salary'] = df['Salary'].mean()
        data_stats['min_salary'] = df['Salary'].min()
        data_stats['max_salary'] = df['Salary'].max()
    
    # Validate projections
    if not df['Projected_Points'].isna().all():
        negative_projections = df[df['Projected_Points'] < 0]
        if len(negative_projections) > 0:
            issues.append(f"Critical: {len(negative_projections)} players with negative projections")
        
        zero_projections = df[df['Projected_Points'] == 0]
        if len(zero_projections) > 0:
            warnings.append(f"Warning: {len(zero_projections)} players with zero projections")
        
        data_stats['avg_projection'] = df['Projected_Points'].mean()
        data_stats['max_projection'] = df['Projected_Points'].max()
    
    # Team validation
    teams = df['Team'].unique()
    data_stats['num_teams'] = len(teams)
    
    if len(teams) != 2:
        issues.append(f"Critical: Found {len(teams)} teams - Showdown requires exactly 2 teams")
    else:
        team_counts = df['Team'].value_counts()
        for team, count in team_counts.items():
            if count < 5:
                issues.append(f"Critical: Team {team} has only {count} players - need at least 5 per team")
            data_stats[f'{team}_players'] = count
    
    # Position distribution check
    position_counts = df['Position'].value_counts()
    data_stats['positions'] = position_counts.to_dict()
    
    essential_positions = ['QB', 'RB', 'WR']
    for pos in essential_positions:
        if pos not in position_counts or position_counts[pos] == 0:
            issues.append(f"Critical: No {pos} players in pool")
        elif position_counts[pos] < 2:
            warnings.append(f"Warning: Only {position_counts[pos]} {pos} player(s)")
    
    # Ownership validation
    if 'Ownership' in df.columns:
        ownership_sum = df['Ownership'].sum()
        if ownership_sum > 0:
            if ownership_sum < 400 or ownership_sum > 800:
                warnings.append(f"Warning: Total ownership sums to {ownership_sum:.1f}% (expected ~600%)")
            
            unrealistic_high = df[df['Ownership'] > 60]
            if len(unrealistic_high) > 0:
                warnings.append(f"Warning: {len(unrealistic_high)} players with >60% ownership")
            
            data_stats['avg_ownership'] = df['Ownership'].mean()
            data_stats['total_ownership'] = ownership_sum
    
    # Salary cap feasibility check
    min_possible_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
    if min_possible_salary > OptimizerConfig.SALARY_CAP:
        issues.append(f"Critical: Impossible to create valid lineup - minimum salary {min_possible_salary} exceeds cap")
    
    # Check for duplicate players
    duplicate_players = df[df.duplicated(subset=['Player'], keep=False)]
    if len(duplicate_players) > 0:
        unique_dups = duplicate_players['Player'].unique()
        issues.append(f"Critical: {len(unique_dups)} duplicate player(s): {', '.join(unique_dups[:5])}")
    
    # Compile results
    is_valid = len(issues) == 0
    all_messages = issues + warnings
    
    # Cache the result
    result = (is_valid, all_messages, data_stats)
    if 'api_cache' not in st.session_state:
        st.session_state['api_cache'] = {}
    st.session_state['api_cache'][cache_key] = {
        'result': result,
        'timestamp': datetime.now()
    }
    
    return result

def auto_fix_common_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically fix common data issues with performance optimization"""
    logger = get_logger()
    df_fixed = df.copy()
    fixes_applied = []
    
    # Remove duplicates
    if df_fixed.duplicated(subset=['Player']).any():
        df_fixed = df_fixed.sort_values('Projected_Points', ascending=False).drop_duplicates(subset=['Player'])
        fixes_applied.append("Removed duplicate players (kept highest projection)")
        logger.log("Fixed duplicate players", "INFO")
    
    # Fill missing ownership
    if 'Ownership' in df_fixed.columns:
        missing_own = df_fixed['Ownership'].isna().sum()
        if missing_own > 0:
            df_fixed['Ownership'].fillna(OptimizerConfig.DEFAULT_OWNERSHIP, inplace=True)
            fixes_applied.append(f"Filled {missing_own} missing ownership values with {OptimizerConfig.DEFAULT_OWNERSHIP}%")
            logger.log(f"Fixed {missing_own} missing ownership values", "INFO")
    
    # Remove invalid salaries
    before_count = len(df_fixed)
    df_fixed = df_fixed[df_fixed['Salary'] >= OptimizerConfig.MIN_SALARY]
    removed = before_count - len(df_fixed)
    if removed > 0:
        fixes_applied.append(f"Removed {removed} players with invalid salaries")
        logger.log(f"Removed {removed} invalid salary players", "INFO")
    
    # Remove missing critical data
    before_count = len(df_fixed)
    df_fixed = df_fixed.dropna(subset=['Player', 'Position', 'Team', 'Salary', 'Projected_Points'])
    removed = before_count - len(df_fixed)
    if removed > 0:
        fixes_applied.append(f"Removed {removed} players with missing critical data")
        logger.log(f"Removed {removed} players with missing data", "INFO")
    
    if fixes_applied:
        st.info("Data fixes applied automatically:")
        for fix in fixes_applied:
            st.write(f"  ✓ {fix}")
    
    return df_fixed

def display_data_validation_results(is_valid: bool, messages: List[str], stats: Dict[str, Any]):
    """Display validation results with performance metrics"""
    logger = get_logger()
    
    if is_valid:
        st.success("✅ Data validation passed - ready to optimize!")
        logger.log("Data validation passed", "INFO")
    else:
        st.error("❌ Data validation failed - issues must be resolved")
        logger.log("Data validation failed", "ERROR")
    
    if messages:
        critical_issues = [msg for msg in messages if msg.startswith("Critical:")]
        warnings = [msg for msg in messages if msg.startswith("Warning:")]
        
        if critical_issues:
            with st.expander("🚨 Critical Issues (Must Fix)", expanded=True):
                for issue in critical_issues:
                    st.error(issue.replace("Critical: ", ""))
        
        if warnings:
            with st.expander("⚠️ Warnings (Should Review)", expanded=not critical_issues):
                for warning in warnings:
                    st.warning(warning.replace("Warning: ", ""))
    
    with st.expander("📊 Data Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Players", stats.get('total_players', 0))
            st.metric("Teams", stats.get('num_teams', 0))
            if 'avg_salary' in stats:
                st.metric("Avg Salary", f"${stats['avg_salary']:,.0f}")
        
        with col2:
            if 'avg_projection' in stats:
                st.metric("Avg Projection", f"{stats['avg_projection']:.1f}")
            if 'max_projection' in stats:
                st.metric("Max Projection", f"{stats['max_projection']:.1f}")
            if 'avg_ownership' in stats:
                st.metric("Avg Ownership", f"{stats['avg_ownership']:.1f}%")
        
        with col3:
            if 'positions' in stats:
                st.write("**Position Counts:**")
                for pos, count in stats['positions'].items():
                    st.write(f"{pos}: {count}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate CSV with comprehensive checks and caching"""
    logger = get_logger()
    perf = get_performance_monitor()
    
    perf.start_timer("data_loading")
    logger.log("Starting data load and validation", "INFO")
    
    df = pd.read_csv(uploaded_file)
    
    # Check required columns
    required_cols = ['first_name', 'last_name', 'position', 'team', 'salary', 'ppg_projection']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Required columns: first_name, last_name, position, team, salary, ppg_projection")
        logger.log(f"Missing columns: {missing_cols}", "ERROR")
        st.stop()
    
    # Create player names
    df['first_name'] = df['first_name'].fillna('')
    df['last_name'] = df['last_name'].fillna('')
    df['Player'] = (df['first_name'] + ' ' + df['last_name']).str.strip()
    
    # Remove empty players
    df = df[df['Player'].str.len() > 0]
    
    # Rename columns
    df = df.rename(columns={
        'position': 'Position',
        'team': 'Team',
        'salary': 'Salary',
        'ppg_projection': 'Projected_Points'
    })
    
    # Ensure numeric types
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    df['Projected_Points'] = pd.to_numeric(df['Projected_Points'], errors='coerce')
    
    # Add GPP value column
    df['Value'] = np.where(
        df['Salary'] > 0,
        df['Projected_Points'] / (df['Salary'] / 1000),
        0
    )
    
    # Run comprehensive validation
    is_valid, messages, stats = validate_lineup_data(df)
    
    # Try auto-fixing if there are issues
    if not is_valid:
        st.warning("Attempting to auto-fix common issues...")
        df = auto_fix_common_issues(df)
        
        # Re-validate after fixes
        is_valid, messages, stats = validate_lineup_data(df)
    
    # Display validation results
    display_data_validation_results(is_valid, messages, stats)
    
    # Stop if still invalid after auto-fix
    if not is_valid:
        st.error("Cannot proceed with optimization. Please fix the critical issues above.")
        logger.log("Data validation failed after auto-fix attempts", "ERROR")
        st.stop()
    
    elapsed = perf.stop_timer("data_loading")
    logger.log(f"Data loaded and validated in {elapsed:.2f}s", "INFO")
    perf.increment_counter("successful_data_loads")
    
    return df

def display_gpp_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame, field_size: str):
    """Display GPP-specific lineup analysis with performance optimization"""
    
    if lineups_df.empty:
        st.warning("No lineups to analyze")
        return
    
    logger = get_logger()
    perf = get_performance_monitor()
    
    perf.start_timer("lineup_analysis")
    logger.log("Generating lineup analysis visualizations", "DEBUG")
    
    # Create GPP analysis visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Strategy Distribution for GPP
    ax1 = axes[0, 0]
    if 'Strategy' in lineups_df.columns:
        strategy_counts = lineups_df['Strategy'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#98D8C8']
        ax1.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.0f%%',
               colors=colors[:len(strategy_counts)], startangle=90)
        ax1.set_title('GPP Strategy Distribution')
    else:
        ax1.text(0.5, 0.5, 'No Strategy Data', ha='center', va='center')
        ax1.set_title('Strategy Distribution')
    
    # 2. Ownership vs Ceiling (GPP Focus)
    ax2 = axes[0, 1]
    ceiling_col = 'Ceiling_99th' if 'Ceiling_99th' in lineups_df.columns else 'Projected'
    if 'Total_Ownership' in lineups_df.columns and ceiling_col in lineups_df.columns:
        scatter = ax2.scatter(lineups_df['Total_Ownership'], lineups_df[ceiling_col],
                            c=lineups_df.get('GPP_Score', lineups_df['Projected']), 
                            cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add optimal ownership zone
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(field_size, (60, 90))
        ax2.axvspan(min_own, max_own, alpha=0.2, color='green', label=f'Optimal ({min_own}-{max_own}%)')
        
        ax2.set_xlabel('Total Ownership %')
        ax2.set_ylabel('99th Percentile Points')
        ax2.set_title(f'GPP Ownership vs Ceiling ({field_size})')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='GPP Score')
    
    # 3. Captain Ownership Distribution
    ax3 = axes[0, 2]
    if 'Captain_Own%' in lineups_df.columns:
        captain_owns = lineups_df['Captain_Own%'].values
        ax3.hist(captain_owns, bins=min(20, len(set(captain_owns))), alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax3.axvline(15, color='red', linestyle='--', label='15% Threshold')
        ax3.set_xlabel('Captain Ownership %')
        ax3.set_ylabel('Number of Lineups')
        ax3.set_title('GPP Captain Ownership Distribution')
        ax3.legend()
    
    # 4. Leverage Score Distribution
    ax4 = axes[1, 0]
    if 'Leverage_Score' in lineups_df.columns:
        ax4.hist(lineups_df['Leverage_Score'], bins=15, alpha=0.7, color='#45B7D1', edgecolor='black')
        ax4.axvline(lineups_df['Leverage_Score'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {lineups_df['Leverage_Score'].mean():.1f}")
        ax4.set_xlabel('GPP Leverage Score')
        ax4.set_ylabel('Number of Lineups')
        ax4.set_title('Leverage Distribution')
        ax4.legend()
    
    # 5. Ship Equity vs Tournament EV
    ax5 = axes[1, 1]
    if 'Ship_Equity' in lineups_df.columns and 'Tournament_EV' in lineups_df.columns:
        ax5.scatter(lineups_df['Ship_Equity'], lineups_df['Tournament_EV'],
                   c=lineups_df.get('Total_Ownership', 50), cmap='RdYlGn_r',
                   s=80, alpha=0.7)
        ax5.set_xlabel('Ship Equity (Win Probability)')
        ax5.set_ylabel('Tournament EV')
        ax5.set_title('GPP Tournament Equity Analysis')
    elif 'Total_Ownership' in lineups_df.columns and 'GPP_Score' in lineups_df.columns:
        ax5.scatter(lineups_df['Total_Ownership'], lineups_df['GPP_Score'], alpha=0.6)
        ax5.set_xlabel('Total Ownership %')
        ax5.set_ylabel('GPP Score')
        ax5.set_title('GPP Score Distribution')
    
    # 6. Top Captains for GPP
    ax6 = axes[1, 2]
    if 'Captain' in lineups_df.columns:
        captain_counts = lineups_df['Captain'].value_counts().head(10)
        
        if 'Captain_Own%' in lineups_df.columns:
            captain_ownership = lineups_df.groupby('Captain')['Captain_Own%'].first()
            colors = ['#FF6B6B' if captain_ownership.get(cap, 15) > 20 else 
                     '#4ECDC4' if captain_ownership.get(cap, 15) > 10 else '#45B7D1' 
                     for cap in captain_counts.index]
        else:
            colors = ['#4ECDC4'] * len(captain_counts)
        
        y_pos = np.arange(len(captain_counts))
        ax6.barh(y_pos, captain_counts.values, color=colors)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(captain_counts.index, fontsize=8)
        ax6.set_xlabel('Times Used as Captain')
        ax6.set_title('Top 10 GPP Captains')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    elapsed = perf.stop_timer("lineup_analysis")
    logger.log(f"Analysis visualizations generated in {elapsed:.2f}s", "DEBUG")
    
    # Ownership Tier Distribution with caching
    st.markdown("### 🎯 GPP Ownership Tier Analysis")
    
    if 'Captain' in lineups_df.columns and 'FLEX' in lineups_df.columns:
        # Use cached ownership distribution if available
        df_json = df.to_json()
        ownership_dist = analyze_ownership_distribution(df_json)
        
        tier_data = []
        for idx, row in lineups_df.head(20).iterrows():
            bucket_counts = {'mega_chalk': 0, 'chalk': 0, 'pivot': 0, 'leverage': 0, 'super_leverage': 0}
            lineup_players = [row['Captain']] + (row['FLEX'] if isinstance(row['FLEX'], list) else [])
            
            for player in lineup_players:
                if player in df['Player'].values:
                    player_own = df[df['Player'] == player]['Ownership'].values[0]
                    bucket = OwnershipBucketManager.get_bucket(player_own)
                    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            
            tier_data.append(list(bucket_counts.values()))
        
        if tier_data:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            cmap = plt.cm.RdYlGn_r
            im = ax.imshow(tier_data, cmap=cmap, aspect='auto', vmin=0, vmax=6)
            
            ax.set_xticks(range(5))
            ax.set_xticklabels(['Mega\nChalk\n(35%+)', 'Chalk\n(20-35%)', 'Pivot\n(10-20%)', 
                               'Leverage\n(5-10%)', 'Super\nLeverage\n(<5%)'])
            ax.set_yticks(range(len(tier_data)))
            ax.set_yticklabels([f'#{i+1}' for i in range(len(tier_data))])
            ax.set_title(f'GPP Ownership Distribution - {field_size.upper()} Field')
            ax.set_xlabel('Ownership Tier')
            ax.set_ylabel('Lineup Rank')
            
            # Add text annotations
            for i in range(len(tier_data)):
                for j in range(5):
                    value = tier_data[i][j]
                    color = "white" if value > 3 else "black"
                    if j < 2 and value > 2:  # Too much chalk for GPP
                        color = "red"
                    elif j >= 3 and value >= 2:  # Good leverage
                        color = "green"
                    text = ax.text(j, i, value, ha="center", va="center",
                                 color=color, fontweight='bold' if j >= 3 else 'normal')
            
            plt.colorbar(im, ax=ax, label='Player Count')
            st.pyplot(fig)

# ============================================================================
# MAIN STREAMLIT UI
# ============================================================================

# Initialize session state
init_session_state()

with st.sidebar:
    st.header("🏆 GPP Tournament Settings")
    
    # Debug panel
    with st.expander("🐛 Debug & Logging"):
        verbose_logging = st.checkbox("Verbose Logging", value=False)
        log_to_file = st.checkbox("Log to File", value=False)
        
        # Update logger settings
        logger = get_logger()
        logger.verbose = verbose_logging
        logger.log_to_file = log_to_file
        
        if st.button("Show Log Summary"):
            logger.display_log_summary()
        
        if st.button("Show Performance Metrics"):
            perf = get_performance_monitor()
            perf.display_metrics()
        
        if st.button("Export Logs"):
            log_text = logger.export_logs()
            st.download_button(
                "Download Logs",
                data=log_text,
                file_name=f"optimizer_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.session_state['api_cache'] = {}
            st.success("Cache cleared!")
    
    st.markdown("### 🎯 Contest Type")
    contest_type = st.selectbox(
        "Select GPP Type",
        list(OptimizerConfig.FIELD_SIZES.keys()),
        index=2,
        help="Different GPP types require different strategies"
    )
    field_size = OptimizerConfig.FIELD_SIZES[contest_type]
    
    # Display field size strategy
    if field_size == 'milly_maker':
        st.info("💎 **Milly Maker Strategy:**\nUltra-contrarian, <80% total ownership, zero chalk")
    elif field_size == 'large_field':
        st.info("🎯 **Large Field Strategy:**\n60-90% ownership, maximize leverage")
    elif field_size == 'medium_field':
        st.info("🔄 **Medium Field Strategy:**\n70-100% ownership, balanced approach")
    else:
        st.info("⚖️ **Small Field Strategy:**\n80-120% ownership, slight contrarian")
    
    st.markdown("---")
    
    st.markdown("### 🤖 AI Configuration")
    api_mode = st.radio(
        "Connection Mode",
        ["Manual (Free)", "API (Automated)"],
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
            if st.button("🔌 Connect API"):
                api_manager = ClaudeAPIManager(api_key)
                use_api = bool(api_manager.client)
        
        if use_api:
            st.success("✅ API Connected")
    else:
        st.info("📋 Manual mode: Copy prompts to Claude")
    
    st.markdown("---")
    
    # GPP-Specific Settings
    with st.expander("⚙️ GPP Advanced Settings"):
        st.markdown("### Optimization Parameters")
        
        force_unique_captains = st.checkbox(
            "Force Unique Captains", 
            value=True,
            help="Each lineup gets different captain (recommended for GPP)"
        )
        
        min_leverage_players = st.slider(
            "Min Sub-10% Players", 0, 4, 2,
            help="Minimum low-owned players per lineup"
        )
        
        st.markdown("### Simulation Settings")
        num_sims = st.slider(
            "Monte Carlo Simulations", 
            1000, 10000, 5000, 1000,
            help="More sims = better accuracy"
        )
        
        st.markdown("### Captain Settings")
        max_captain_ownership = st.slider(
            "Max Captain Ownership %", 
            5, 30, 15, 5,
            help="Maximum ownership for captains"
        )
        
        st.markdown("### Export Options")
        include_pivots = st.checkbox(
            "Generate Captain Pivots", 
            value=True,
            help="Create leverage captain swaps"
        )
        
        export_top_n = st.number_input(
            "Export Top N Lineups", 
            5, 150, 20, 5,
            help="Number of lineups to export"
        )

# Main Content Area
st.markdown("## 🎮 GPP Tournament Optimizer")

# Display current settings
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Contest Type", contest_type)
with col2:
    ownership_range = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
    st.metric("Target Own%", f"{ownership_range[0]}-{ownership_range[1]}%")
with col3:
    st.metric("Field Size", field_size.replace('_', ' ').title())
with col4:
    st.metric("Mode", "🏆 GPP Only")

st.markdown("---")
st.markdown("## 📁 Data Upload & Game Configuration")

uploaded_file = st.file_uploader(
    "Upload DraftKings CSV",
    type="csv",
    help="Export player pool from DraftKings Showdown contest"
)

if uploaded_file is not None:
    # Load and validate data with caching
    df = load_and_validate_data(uploaded_file)
    
    # Game Configuration
    st.markdown("### ⚙️ Game Setup")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        teams = st.text_input("Teams", "BUF vs MIA", help="Team matchup")
    with col2:
        total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5,
                               help="Higher totals = more correlation plays")
    with col3:
        spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5,
                                help="Large spreads = leverage on dogs")
    with col4:
        weather = st.selectbox("Weather", ["Clear", "Wind", "Rain", "Snow"],
                             help="Weather impacts volatility")
    
    game_info = {
        'teams': teams,
        'total': total,
        'spread': spread,
        'weather': weather,
        'field_size': field_size
    }
    
    # GPP-Specific Player Adjustments
    st.markdown("### 🎯 GPP Player Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Ownership Projections (CRITICAL FOR GPP)")
        ownership_text = st.text_area(
            "Format: Player: %",
            height=150,
            placeholder="Josh Allen: 45\nStefon Diggs: 35\nTyreek Hill: 40\nJaylen Waddle: 25",
            help="Accurate ownership projections are essential for GPP success"
        )
        
        # Parse ownership
        ownership_dict = {}
        for line in ownership_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        player = parts[0].strip()
                        own_pct = float(parts[1].strip())
                        if 0 <= own_pct <= 100:
                            ownership_dict[player] = own_pct
                    except:
                        pass
        
        # Apply ownership
        df['Ownership'] = df['Player'].map(ownership_dict).fillna(OptimizerConfig.DEFAULT_OWNERSHIP)
        
        # Show ownership distribution
        if ownership_dict:
            high_owned = len(df[df['Ownership'] > 30])
            low_owned = len(df[df['Ownership'] < 10])
            st.info(f"Chalk (30%+): {high_owned} | Leverage (<10%): {low_owned}")
    
    with col2:
        st.markdown("#### 🎲 GPP Ceiling Boosts")
        boost_text = st.text_area(
            "High Ceiling Players (Format: Player: Multiplier)",
            height=150,
            placeholder="Tyreek Hill: 1.2\nJosh Allen: 1.15\nRaheem Mostert: 1.25",
            help="Boost projections for boom/bust players"
        )
        
        # Apply ceiling boosts
        for line in boost_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        player = parts[0].strip()
                        multiplier = float(parts[1].strip())
                        if player in df['Player'].values and 0.5 <= multiplier <= 2.0:
                            df.loc[df['Player'] == player, 'Projected_Points'] *= multiplier
                            df.loc[df['Player'] == player, 'Ceiling_Boost'] = multiplier
                    except:
                        pass
    
    # Add GPP-specific columns
    df['Bucket'] = df['Ownership'].apply(OwnershipBucketManager.get_bucket)
    df['GPP_Value'] = np.where(
        df['Salary'] > 0,
        (df['Projected_Points'] / (df['Salary'] / 1000)) * (20 / (df['Ownership'] + 5)),
        0
    )
    df['Leverage_Score'] = df.apply(
        lambda x: 3 if x['Ownership'] < 5 else 2 if x['Ownership'] < 10 else 1 if x['Ownership'] < 15 else -1 if x['Ownership'] > 30 else 0, 
        axis=1
    )
    
    # Display GPP player pool
    st.markdown("### 💎 GPP Player Pool Analysis")
    
    # Use cached ownership distribution
    df_json = df.to_json()
    ownership_dist = analyze_ownership_distribution(df_json)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**GPP Ownership Tiers:**")
        tier_emojis = {
            'mega_chalk': '🔴',
            'chalk': '🟠', 
            'pivot': '🟡',
            'leverage': '🟢',
            'super_leverage': '💎'
        }
        
        for bucket in ['mega_chalk', 'chalk', 'pivot', 'leverage', 'super_leverage']:
            count = ownership_dist['distribution'].get(bucket, 0)
            emoji = tier_emojis.get(bucket, '')
            percentage = (count / ownership_dist['total_players'] * 100) if ownership_dist['total_players'] > 0 else 0
            st.write(f"{emoji} {bucket}: {count} ({percentage:.0f}%)")
        
        # GPP recommendations
        if ownership_dist['distribution'].get('super_leverage', 0) < 5:
            st.warning("⚠️ Limited super leverage plays available")
        if ownership_dist['distribution'].get('mega_chalk', 0) > 10:
            st.info("📊 Heavy chalk slate - focus on leverage")
    
    with col2:
        # Show player pool
        display_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 
                       'Ownership', 'Bucket', 'GPP_Value', 'Leverage_Score']
        
        def highlight_gpp_rows(row):
            if row['Bucket'] == 'super_leverage':
                return ['background-color: #90EE90'] * len(row)
            elif row['Bucket'] == 'leverage':
                return ['background-color: #98FB98'] * len(row)
            elif row['Bucket'] == 'mega_chalk':
                return ['background-color: #FFB6C1'] * len(row)
            return [''] * len(row)
        
        styled_df = df[display_cols].sort_values('GPP_Value', ascending=False).head(20).style.apply(
            highlight_gpp_rows, axis=1
        )
        
        st.dataframe(
            styled_df,
            height=400,
            use_container_width=True
        )
    
    # Optimization Section
    st.markdown("---")
    st.markdown("## 🚀 GPP Lineup Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lineups = st.number_input(
            "Number of Lineups", 
            5, 150, 20, 5,
            help="More lineups = better coverage"
        )
    
    with col2:
        correlation_emphasis = st.slider(
            "Correlation Focus", 
            0, 100, 50,
            help="Higher = more stacks"
        )
    
    with col3:
        if st.button("🎯 Generate GPP Lineups", type="primary", use_container_width=True):
            logger = get_logger()
            perf = get_performance_monitor()
            
            # Save current settings
            current_settings = {
                'num_lineups': num_lineups,
                'field_size': field_size,
                'force_unique_captains': force_unique_captains,
                'num_sims': num_sims,
                'correlation_emphasis': correlation_emphasis
            }
            
            # Initialize optimizer
            optimizer = GPPDualAIOptimizer(df, game_info, field_size, api_manager)
            
            # Get AI strategies
            with st.spinner("Getting GPP AI strategies..."):
                rec1, rec2 = optimizer.get_ai_strategies(use_api=use_api)
            
            # Display AI insights
            with st.expander("🧠 GPP AI Strategic Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Game Theory AI")
                    st.metric("Confidence", f"{rec1.confidence:.0%}")
                    
                    if rec1.captain_targets:
                        st.markdown("**Leverage Captains:**")
                        for captain in rec1.captain_targets[:5]:
                            own = df[df['Player'] == captain]['Ownership'].values[0] if captain in df['Player'].values else 5
                            emoji = "💎" if own < 5 else "🟢" if own < 10 else "🟡" if own < 15 else "⚠️"
                            st.write(f"{emoji} {captain} ({own:.0f}%)")
                    
                    if rec1.fades:
                        st.markdown("**Fade Targets (Chalk):**")
                        for fade in rec1.fades[:3]:
                            st.write(f"🔴 {fade}")
                    
                    if rec1.key_insights:
                        st.markdown("**GPP Insights:**")
                        for insight in rec1.key_insights[:3]:
                            st.info(insight)
                
                with col2:
                    st.markdown("### 🔗 Correlation AI")
                    st.metric("Confidence", f"{rec2.confidence:.0%}")
                    
                    if rec2.stacks:
                        st.markdown("**GPP Stacks:**")
                        for stack in rec2.stacks[:5]:
                            if isinstance(stack, dict):
                                p1 = stack.get('player1', '')
                                p2 = stack.get('player2', '')
                                stack_type = stack.get('type', '')
                                if p1 and p2:
                                    emoji = "💎" if 'leverage' in stack_type else "🔗"
                                    st.write(f"{emoji} {p1} + {p2}")
                    
                    if rec2.key_insights:
                        st.markdown("**Correlation Insights:**")
                        for insight in rec2.key_insights[:3]:
                            st.info(insight)
            
            # Generate GPP lineups
            with st.spinner(f"Generating {num_lineups} GPP lineups for {field_size}..."):
                lineups_df = optimizer.generate_gpp_lineups(
                    num_lineups, rec1, rec2, force_unique_captains=force_unique_captains
                )
            
            if lineups_df.empty:
                st.error("❌ No valid lineups generated. Try adjusting constraints.")
            else:
                st.success(f"✅ Generated {len(lineups_df)} GPP lineups!")
                
                # Calculate correlations with caching
                df_json = df.to_json()
                game_info_json = json.dumps(game_info)
                correlation_data = calculate_correlations_cached(df_json, game_info_json, field_size)
                correlations = correlation_data['correlations']
                
                # Run GPP simulations with caching
                with st.spinner("Running GPP tournament simulations..."):
                    perf.start_timer("simulations")
                    
                    for idx, row in lineups_df.iterrows():
                        lineup_json = json.dumps(row.to_dict())
                        correlations_json = json.dumps({str(k): v for k, v in correlations.items()})
                        
                        sim_results = run_simulations_cached(
                            lineup_json, df_json, correlations_json, num_sims, field_size
                        )
                        
                        for key, value in sim_results.items():
                            lineups_df.loc[idx, key] = value
                    
                    perf.stop_timer("simulations")
                
                # Calculate GPP scores
                lineups_df = calculate_gpp_scores(lineups_df, field_size)
                
                # Sort by GPP score
                lineups_df = lineups_df.sort_values('GPP_Score', ascending=False).reset_index(drop=True)
                
                # Store in session state
                st.session_state['lineups_df'] = lineups_df
                st.session_state['df'] = df
                st.session_state['correlations'] = correlations
                st.session_state['field_size'] = field_size
                
                # Save optimization session
                save_optimization_session(lineups_df, current_settings)
                
                # Generate captain pivots if enabled
                if include_pivots:
                    with st.spinner("Generating GPP captain pivots..."):
                        all_pivots = []
                        for i in range(min(3, len(lineups_df))):
                            pivots = optimizer.pivot_generator.find_optimal_gpp_pivots(
                                lineups_df.iloc[i].to_dict(), df, field_size
                            )
                            all_pivots.extend(pivots)
                        st.session_state['pivots_df'] = all_pivots
    
    # Display results if lineups exist
    if 'lineups_df' in st.session_state:
        lineups_df = st.session_state['lineups_df']
        df = st.session_state['df']
        field_size = st.session_state.get('field_size', 'large_field')
        
        st.markdown("---")
        st.markdown("## 📊 GPP Optimization Results")
        
        # GPP Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Lineups", len(lineups_df))
        with col2:
            ceiling_col = 'Ceiling_99th' if 'Ceiling_99th' in lineups_df else 'Projected'
            st.metric("Avg 99th%", f"{lineups_df[ceiling_col].mean():.1f}")
        with col3:
            st.metric("Avg Own%", f"{lineups_df['Total_Ownership'].mean():.1f}%")
        with col4:
            st.metric("Avg Leverage", f"{lineups_df['Leverage_Score'].mean():.1f}")
        with col5:
            unique_captains = lineups_df['Captain'].nunique()
            st.metric("Unique CPT", f"{unique_captains}/{len(lineups_df)}")
        with col6:
            elite_lineups = len(lineups_df[lineups_df['Total_Ownership'] < 70])
            st.metric("Elite (<70%)", elite_lineups)
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 GPP Lineups", "🔄 Captain Pivots", "📈 GPP Analysis", 
            "💎 Leverage Plays", "📊 Simulations", "💾 Export"
        ])
        
        with tab1:
            st.markdown("### 🏆 Top GPP Lineups")
            
            # Display options
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                display_format = st.radio(
                    "Display Format",
                    ["GPP Summary", "Detailed", "Compact"],
                    horizontal=True
                )
            with col2:
                max_display = len(lineups_df) if len(lineups_df) > 0 else 150
                default_display = min(10, max_display)
                show_top_n = st.number_input("Show Top", 5, max_display, default_display, 5)
            with col3:
                min_leverage = st.number_input("Min Leverage", 0, 20, 5)
                filtered_df = lineups_df[lineups_df['Leverage_Score'] >= min_leverage]
                if len(filtered_df) == 0:
                    filtered_df = lineups_df
            
            if display_format == "GPP Summary":
                display_cols = ['Lineup', 'Strategy', 'Captain', 'Captain_Own%', 'Projected', 
                              'Ceiling_99th', 'Total_Ownership', 'Leverage_Score', 
                              'GPP_Score', 'Ship_Equity']
                
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                
                st.dataframe(
                    filtered_df[display_cols].head(show_top_n),
                    use_container_width=True
                )
            
            elif display_format == "Detailed":
                for i, (idx, lineup) in enumerate(filtered_df.head(show_top_n).iterrows(), 1):
                    tier_emoji = "💎" if lineup['Total_Ownership'] < 60 else "🟢" if lineup['Total_Ownership'] < 80 else "🟡"
                    
                    with st.expander(f"{tier_emoji} Lineup #{i} - {lineup['Strategy']} - GPP Score: {lineup.get('GPP_Score', 0):.1f}"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("**Roster:**")
                            captain_own = lineup.get('Captain_Own%', 0)
                            st.write(f"🎯 **Captain:** {lineup['Captain']} ({captain_own:.1f}%)")
                            st.write("**FLEX:**")
                            flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else []
                            for player in flex_players:
                                if player in df['Player'].values:
                                    pos = df[df['Player'] == player]['Position'].values[0]
                                    own = df[df['Player'] == player]['Ownership'].values[0]
                                    emoji = "💎" if own < 5 else "🟢" if own < 10 else ""
                                    st.write(f"{emoji} {player} ({pos}) - {own:.0f}%")
                        
                        with col2:
                            st.markdown("**GPP Projections:**")
                            st.metric("99th %ile", f"{lineup.get('Ceiling_99th', lineup['Projected']*1.8):.1f}")
                            st.metric("99.9th %ile", f"{lineup.get('Ceiling_99_9th', lineup['Projected']*2):.1f}")
                            st.metric("Ship Rate", f"{lineup.get('Ship_Rate', 0.1):.3f}%")
                            st.metric("Boom Rate", f"{lineup.get('Boom_Rate', 5):.1f}%")
                        
                        with col3:
                            st.markdown("**GPP Metrics:**")
                            st.write(f"💰 Salary: ${lineup['Salary']:,}")
                            st.write(f"📊 Total Own: {lineup['Total_Ownership']:.1f}%")
                            st.write(f"🎯 Leverage: {lineup['Leverage_Score']:.1f}")
                            st.write(f"🏆 GPP Score: {lineup.get('GPP_Score', 0):.1f}")
                        
                        with col4:
                            st.markdown("**Stack Info:**")
                            if lineup['Has_Stack']:
                                st.success(f"✅ {lineup['Stack_Details']}")
                            else:
                                st.info("No primary stack")
                            st.write(f"Field: {lineup['Field_Size']}")
                            st.write(f"Tier: {lineup.get('Ownership_Tier', 'Unknown')}")
            
            else:  # Compact view
                for i, (idx, lineup) in enumerate(filtered_df.head(show_top_n).iterrows(), 1):
                    emoji = "💎" if lineup['Total_Ownership'] < 60 else "🟢" if lineup['Total_Ownership'] < 80 else "🟡"
                    captain_own = lineup.get('Captain_Own%', 0)
                    flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else []
                    flex_preview = ', '.join(flex_players[:3]) + ('...' if len(flex_players) > 3 else '')
                    st.write(f"{emoji} **#{i}:** CPT: {lineup['Captain']} ({captain_own:.0f}%) | FLEX: {flex_preview} | Own: {lineup['Total_Ownership']:.0f}% | GPP: {lineup.get('GPP_Score', 0):.0f}")
        
        with tab2:
            st.markdown("### 🔄 GPP Captain Pivots")
            
            if 'pivots_df' in st.session_state and st.session_state['pivots_df']:
                pivots_df = st.session_state['pivots_df']
                
                st.info(f"Generated {len(pivots_df)} GPP captain pivot variations")
                
                for i, pivot in enumerate(pivots_df[:10], 1):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**#{i}:** {pivot['Original_Captain']} → {pivot['Captain']}")
                    with col2:
                        st.write(f"Captain Own: {pivot['Captain_Own%']:.1f}%")
                    with col3:
                        st.write(f"Leverage: +{pivot['Leverage_Gain']:.1f}")
                    with col4:
                        st.write(f"{pivot['Pivot_Type']}")
            else:
                st.info("Enable captain pivots in settings to generate variations")
        
        with tab3:
            st.markdown("### 📈 GPP Tournament Analysis")
            display_gpp_lineup_analysis(lineups_df, df, field_size)
        
        with tab4:
            st.markdown("### 💎 GPP Leverage Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Low-Owned Captains (<15%)")
                if 'Captain_Own%' in lineups_df.columns:
                    low_captains = lineups_df[lineups_df['Captain_Own%'] < 15]['Captain'].value_counts()
                else:
                    low_captains = pd.Series()
                    
                for player, count in low_captains.items():
                    if player in df['Player'].values:
                        own = df[df['Player'] == player]['Ownership'].values[0]
                        emoji = "💎" if own < 5 else "🟢" if own < 10 else "🟡"
                        pct = count / len(lineups_df) * 100
                        st.write(f"{emoji} {player} ({own:.0f}%) - {count} lineups ({pct:.0f}%)")
            
            with col2:
                st.markdown("#### 🔗 Leverage Stacks")
                leverage_stacks = []
                
                for idx, row in lineups_df.iterrows():
                    if row['Has_Stack'] and row['Total_Ownership'] < 80:
                        stacks = row['Stack_Details'].split(', ')
                        for stack in stacks:
                            if stack != 'None':
                                leverage_stacks.append(stack)
                
                if leverage_stacks:
                    stack_counts = Counter(leverage_stacks)
                    for stack, count in stack_counts.most_common(10):
                        pct = count / len(lineups_df) * 100
                        st.write(f"🔗 {stack} - {count} lineups ({pct:.0f}%)")
            
            st.markdown("#### 💎 Super Leverage Plays (<5% ownership)")
            
            player_usage = defaultdict(int)
            for idx, row in lineups_df.iterrows():
                lineup_players = [row['Captain']] + (row['FLEX'] if isinstance(row['FLEX'], list) else [])
                for player in lineup_players:
                    player_usage[player] += 1
            
            super_leverage = []
            for player, usage in player_usage.items():
                if player in df['Player'].values:
                    own = df[df['Player'] == player]['Ownership'].values[0]
                    if own < 5 and usage >= 3:
                        super_leverage.append((player, own, usage))
            
            if super_leverage:
                super_leverage.sort(key=lambda x: x[2], reverse=True)
                for player, own, usage in super_leverage[:10]:
                    pct = usage / len(lineups_df) * 100
                    st.write(f"💎 {player} ({own:.0f}%) - {usage} lineups ({pct:.0f}%)")
        
        with tab5:
            st.markdown("### 📊 GPP Simulation Results")
            
            sim_cols = ['Lineup', 'Captain', 'Total_Ownership', 'Mean', 
                       'Ceiling_95th', 'Ceiling_99th', 'Ceiling_99_9th',
                       'Ship_Rate', 'Elite_Rate', 'Boom_Rate']
            
            sim_cols = [col for col in sim_cols if col in lineups_df.columns]
            
            if sim_cols:
                st.dataframe(
                    lineups_df[sim_cols].head(20),
                    use_container_width=True
                )
            
            if 'Ship_Rate' in lineups_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(lineups_df['Ship_Rate'], bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax.set_xlabel('Ship Rate (%)')
                ax.set_ylabel('Number of Lineups')
                ax.set_title(f'Tournament Win Probability Distribution - {field_size}')
                st.pyplot(fig)
        
        with tab6:
            st.markdown("### 💾 Export GPP Lineups")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### DraftKings Upload Format")
                
                dk_lineups = []
                export_df = lineups_df.head(export_top_n)
                
                for idx, row in export_df.iterrows():
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
                
                st.write(f"Preview (first 5 of {len(dk_df)}):")
                st.dataframe(dk_df.head(), use_container_width=True)
                
                csv = dk_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download DK CSV ({len(dk_df)} lineups)",
                    data=csv,
                    file_name=f"dk_gpp_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("#### GPP Analysis Export")
                
                export_analysis = export_df.copy()
                export_analysis['FLEX'] = export_analysis['FLEX'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
                
                gpp_export_cols = ['Lineup', 'Strategy', 'Captain', 'Captain_Own%', 'FLEX', 
                                  'Projected', 'Salary', 'Total_Ownership', 'Leverage_Score',
                                  'Ceiling_95th', 'Ceiling_99th', 'Ceiling_99_9th',
                                  'Ship_Rate', 'Elite_Rate', 'Boom_Rate',
'GPP_Score', 'Tournament_EV', 'Has_Stack']
                
                gpp_export_cols = [col for col in gpp_export_cols if col in export_analysis.columns]
                
                final_export = export_analysis[gpp_export_cols]
                
                st.write(f"Preview (first 5 of {len(final_export)}):")
                st.dataframe(final_export.head(), use_container_width=True)
                
                csv_full = final_export.to_csv(index=False)
                st.download_button(
                    label=f"📊 Download GPP Analysis ({len(final_export)} lineups)",
                    data=csv_full,
                    file_name=f"gpp_analysis_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            # Captain pivot export
            if include_pivots and 'pivots_df' in st.session_state and st.session_state['pivots_df']:
                st.markdown("#### 🔄 Captain Pivots Export")
                
                pivots_list = st.session_state['pivots_df']
                pivot_export = pd.DataFrame(pivots_list)
                
                csv_pivots = pivot_export.to_csv(index=False)
                st.download_button(
                    label=f"🔄 Download Captain Pivots ({len(pivot_export)} pivots)",
                    data=csv_pivots,
                    file_name=f"captain_pivots_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            # Session history export
            st.markdown("#### 📜 Session History")
            if st.session_state.get('optimization_history'):
                st.info(f"You have {len(st.session_state['optimization_history'])} saved sessions")
                
                if st.button("Export Session History"):
                    history_data = []
                    for session in st.session_state['optimization_history']:
                        history_data.append({
                            'Timestamp': session['timestamp'],
                            'Field Size': session['field_size'],
                            'Lineups Generated': len(session['lineups']),
                            'Settings': str(session['settings'])
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    csv_history = history_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📜 Download Session History",
                        data=csv_history,
                        file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

# Footer with GPP-specific tips
st.markdown("---")
st.markdown("""
### 🏆 NFL GPP Tournament Optimizer - Professional Edition v5.1

**New Performance Features:**
- ⚡ **Cached Calculations**: Correlations and simulations are cached for faster re-runs
- 💾 **Session History**: Automatically saves your optimization sessions
- 📊 **Performance Monitoring**: Track operation timings and bottlenecks
- 🔄 **Smart Caching**: Reuses validation and analysis results when possible

**GPP-Specific Features:**
- 💎 **Super Leverage Detection**: Identifies <5% owned tournament winners
- 🎯 **Field-Size Optimization**: Tailored strategies for different GPP types
- 🔄 **Captain Pivot Engine**: Creates unique lineups with leverage captains
- 📊 **Ship Equity Calculator**: Tournament win probability analysis
- 🤖 **Dual AI System**: Game theory + correlation for maximum edge

**GPP Strategy Tips:**
- **Milly Maker**: Target <80% total ownership with zero chalk tolerance
- **Large Field**: 60-90% ownership with 2+ leverage plays minimum
- **Small Field**: Can use slightly higher ownership (80-120%) 
- **Captain Selection**: Prioritize <15% owned captains for differentiation
- **Stacking**: Low-owned game stacks in projected shootouts (50+ total)

**Ownership Tier Guide:**
- 💎 Super Leverage (<5%): Maximum tournament equity
- 🟢 Leverage (5-10%): Strong GPP plays
- 🟡 Pivot (10-20%): Balanced risk/reward
- 🟠 Chalk (20-35%): Use sparingly
- 🔴 Mega Chalk (35%+): Avoid in large field GPPs

**Performance Tips:**
- Enable caching for repeated optimizations with same data
- Use session history to track successful configurations
- Check debug panel for performance metrics and bottlenecks
- Clear cache if experiencing issues with stale data

**Version:** 5.1 GPP Edition | **Focus:** Tournament Winning Upside with Performance Optimization

*Maximize your tournament equity with blazing fast optimization!* 🚀
""")

# Display performance summary if available
if 'lineups_df' in st.session_state:
    perf = get_performance_monitor()
    metrics = perf.get_metrics()
    
    if metrics['timers']:
        with st.expander("⚡ Performance Summary"):
            total_time = sum(timer['total'] for timer in metrics['timers'].values())
            st.write(f"**Total Time:** {total_time:.2f}s")
            
            for name, stats in metrics['timers'].items():
                st.write(f"- {name}: {stats['average']:.3f}s avg ({stats['count']} operations)")
            
            if metrics['counters']:
                st.write("**Operation Counts:**")
                for name, count in metrics['counters'].items():
                    st.write(f"- {name}: {count}")

# Display current timestamp and field size
if 'field_size' in st.session_state:
    current_field = st.session_state['field_size']
    st.caption(f"GPP Optimizer last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Field: {current_field} | Cache Status: Active")
else:
    st.caption(f"GPP Optimizer ready: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | No active session")

