# NFL GPP DUAL-AI OPTIMIZER - PART 1: IMPORTS AND CONFIGURATION
# Version 5.0 - GPP Tournament Focus

import streamlit as st
import pandas as pd
import pulp
import numpy as np
from scipy.stats import multivariate_normal, gamma, norm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set
import json
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict
import hashlib
from datetime import datetime, timedelta

# For Google Colab compatibility
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.warning("Anthropic package not installed. Install with: pip install anthropic")

st.set_page_config(page_title="NFL GPP Optimizer Pro", page_icon="🏈", layout="wide")
st.title('🏈 NFL GPP Tournament Optimizer - Professional Edition')

# ============================================================================
# GPP-FOCUSED CONFIGURATION
# ============================================================================

class OptimizerConfig:
    """GPP-optimized configuration settings"""
    SALARY_CAP = 50000
    ROSTER_SIZE = 6  # 1 Captain + 5 FLEX
    MAX_PLAYERS_PER_TEAM = 4
    CAPTAIN_MULTIPLIER = 1.5
    DEFAULT_OWNERSHIP = 5
    MIN_SALARY = 1000
    
    # GPP-FOCUSED VOLATILITY (increased for more variance)
    BASE_VOLATILITY = 0.30  # Increased from 0.25
    HIGH_VOLATILITY = 0.45  # Increased from 0.35
    INJURY_RATE = 0.08  # Increased from 0.05
    BOOM_RATE = 0.08  # Increased from 0.05
    
    # Simulation parameters
    NUM_SIMS = 5000
    FIELD_SIZE = 100000
    
    # API Configuration
    CLAUDE_MODEL = "claude-3-haiku-20240307"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # GPP-FOCUSED OWNERSHIP BUCKETS
    OWNERSHIP_BUCKETS = {
        'mega_chalk': (35, 100),      # Lowered from 40%
        'chalk': (20, 35),             # Lowered from 25%
        'pivot': (10, 20),             # Tightened range
        'leverage': (5, 10),           # 5-10% ownership
        'super_leverage': (0, 5)       # <5% ownership
    }
    
    # GPP-ONLY BUCKET RULES
    BUCKET_RULES = {
        'gpp_balanced': {
            'mega_chalk': (0, 2),      # Max 2 mega chalk
            'chalk': (0, 3),           # Max 3 chalk
            'pivot': (1, 6),           # At least 1 pivot
            'leverage': (1, 6),        # At least 1 leverage
            'super_leverage': (0, 6)   # Any super leverage
        },
        'contrarian': {
            'mega_chalk': (0, 0),      # NO mega chalk
            'chalk': (0, 1),           # Max 1 chalk
            'pivot': (2, 6),           # At least 2 pivots
            'leverage': (2, 6),        # At least 2 leverage
            'super_leverage': (1, 6)   # At least 1 super leverage
        },
        'super_contrarian': {  # NEW for GPP
            'mega_chalk': (0, 0),      
            'chalk': (0, 0),           # NO chalk at all
            'pivot': (1, 6),           
            'leverage': (2, 6),        # At least 2 leverage
            'super_leverage': (2, 6)   # At least 2 super leverage
        }
    }
    
    # GPP-FOCUSED TARGET OWNERSHIP BY FIELD SIZE
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (80, 120),    # Small GPP/Single Entry
        'medium_field': (70, 100),   # Medium GPP
        'large_field': (60, 90),     # Large GPP (default)
        'milly_maker': (50, 80)      # Massive GPP
    }
    
    # GPP FIELD SIZE DEFINITIONS
    FIELD_SIZES = {
        'Single Entry': 'small_field',
        '3-Max': 'small_field',
        '20-Max': 'medium_field',
        '150-Max': 'large_field',
        'Milly Maker': 'milly_maker'
    }
    
    # Correlation values adjusted for GPP
    QB_PASS_CATCHER_CORR = 0.50  # Increased from 0.45
    QB_RB_CORR = -0.20  # More negative
    SAME_TEAM_WR_CORR = 0.10  # Decreased to avoid doubling up
    OPPOSING_QB_CORR = 0.35  # Increased for shootouts
    DST_OPPOSING_CORR = -0.40  # More negative
    
    # GPP Strategy Weights by Field Size
    GPP_STRATEGY_WEIGHTS = {
        'small_field': {
            'leverage': 0.25,
            'contrarian': 0.15,
            'game_stack': 0.25,
            'stars_scrubs': 0.15,
            'correlation': 0.20
        },
        'medium_field': {
            'leverage': 0.30,
            'contrarian': 0.20,
            'game_stack': 0.20,
            'stars_scrubs': 0.15,
            'correlation': 0.15
        },
        'large_field': {
            'leverage': 0.35,
            'contrarian': 0.25,
            'game_stack': 0.20,
            'stars_scrubs': 0.15,
            'correlation': 0.05
        },
        'milly_maker': {
            'leverage': 0.40,
            'contrarian': 0.35,
            'game_stack': 0.15,
            'stars_scrubs': 0.10,
            'correlation': 0.00
        }
    }

class StrategyType(Enum):
    """GPP-focused strategies only"""
    LEVERAGE = "leverage"
    CONTRARIAN = "contrarian"
    SUPER_CONTRARIAN = "super_contrarian"
    GAME_STACK = "game_stack"
    STARS_SCRUBS = "stars_scrubs"
    CORRELATION = "correlation"

@dataclass
class AIRecommendation:
    """Structure for AI strategist recommendations"""
    strategist_name: str
    confidence: float
    captain_targets: List[str]
    stacks: List[Dict[str, str]]
    fades: List[str]
    boosts: List[str]
    strategy_weights: Dict[StrategyType, float]
    key_insights: List[str]
    gpp_specific_rules: Dict[str, any]  # New field for GPP rules

@dataclass
class PlayerProjection:
    """Enhanced player projection with GPP metrics"""
    player: str
    projection: float
    floor: float
    ceiling: float
    volatility: float
    boom_probability: float  # New for GPP
    bust_probability: float  # New for GPP

# ============================================================================
# CLAUDE API MANAGER
# ============================================================================

class ClaudeAPIManager:
    """Manages Claude API interactions with caching"""
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.client = None
        self.cache = {}  # Cache API responses
        
        if ANTHROPIC_AVAILABLE and api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                st.success("✅ Claude API connected successfully")
            except Exception as e:
                st.error(f"Failed to initialize Claude API: {e}")
                self.client = None
    
    def get_ai_response(self, prompt: str, system_prompt: str = None, use_cache: bool = True) -> str:
        """Get response from Claude API with caching"""
        if not self.client:
            return "{}"
        
        # Check cache first
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        if use_cache and prompt_hash in self.cache:
            st.info("Using cached AI response")
            return self.cache[prompt_hash]
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if system_prompt:
                system = system_prompt
            else:
                system = """You are an expert DFS analyst specializing in GPP tournaments. 
                           Provide strategic recommendations in valid JSON format only, no markdown or extra text.
                           Focus on tournament-winning strategies with emphasis on ceiling and leverage."""
            
            response = self.client.messages.create(
                model=OptimizerConfig.CLAUDE_MODEL,
                max_tokens=OptimizerConfig.MAX_TOKENS,
                temperature=OptimizerConfig.TEMPERATURE,
                system=system,
                messages=messages
            )
            
            response_text = response.content[0].text.strip()
            
            # Clean JSON formatting
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            # Cache the response
            self.cache[prompt_hash] = response_text
            return response_text
            
        except Exception as e:
            st.error(f"API call failed: {e}")
            return "{}"

# ============================================================================
# OWNERSHIP BUCKET MANAGER WITH GPP FOCUS
# ============================================================================

class OwnershipBucketManager:
    """GPP-focused ownership bucketing and analysis"""
    
    @staticmethod
    def get_bucket(ownership: float) -> str:
        """Determine which bucket a player belongs to"""
        for bucket_name, (min_own, max_own) in OptimizerConfig.OWNERSHIP_BUCKETS.items():
            if min_own <= ownership < max_own:
                return bucket_name
        return 'super_leverage'
    
    @staticmethod
    def categorize_players(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize all players into ownership buckets"""
        buckets = defaultdict(list)
        for _, row in df.iterrows():
            bucket = OwnershipBucketManager.get_bucket(row['Ownership'])
            buckets[bucket].append(row['Player'])
        return dict(buckets)
    
    @staticmethod
    def validate_gpp_lineup(lineup_players: List[str], df: pd.DataFrame, 
                           field_size: str = 'large_field') -> Tuple[bool, str]:
        """Validate lineup meets GPP bucket constraints"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        bucket_counts = defaultdict(int)
        total_ownership = 0
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            bucket = OwnershipBucketManager.get_bucket(ownership)
            bucket_counts[bucket] += 1
            total_ownership += ownership
        
        # Get appropriate rules based on field size
        if field_size == 'milly_maker':
            rules = OptimizerConfig.BUCKET_RULES['super_contrarian']
        elif field_size == 'large_field':
            rules = OptimizerConfig.BUCKET_RULES['contrarian']
        else:
            rules = OptimizerConfig.BUCKET_RULES['gpp_balanced']
        
        violations = []
        
        # Check bucket constraints
        for bucket_name, (min_count, max_count) in rules.items():
            count = bucket_counts.get(bucket_name, 0)
            if count < min_count:
                violations.append(f"Need at least {min_count} {bucket_name}")
            if count > max_count:
                violations.append(f"Too many {bucket_name} ({count}/{max_count})")
        
        # Check total ownership for field size
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
        if total_ownership < min_own:
            violations.append(f"Total ownership too low ({total_ownership:.1f}% < {min_own}%)")
        if total_ownership > max_own:
            violations.append(f"Total ownership too high ({total_ownership:.1f}% > {max_own}%)")
        
        if violations:
            return False, "; ".join(violations)
        return True, "Valid GPP lineup"
    
    @staticmethod
    def get_gpp_summary(lineup_players: List[str], df: pd.DataFrame, field_size: str) -> str:
        """Get GPP-specific summary of lineup's ownership profile"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        bucket_counts = defaultdict(int)
        total_ownership = 0
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            bucket = OwnershipBucketManager.get_bucket(ownership)
            bucket_counts[bucket] += 1
            total_ownership += ownership
        
        # GPP-specific emoji indicators
        bucket_emojis = {
            'mega_chalk': '🔴',  # Bad for GPP
            'chalk': '🟠',       # Caution
            'pivot': '🟡',       # OK
            'leverage': '🟢',    # Good
            'super_leverage': '💎'  # Excellent
        }
        
        # Determine GPP quality
        gpp_quality = "💎 Elite" if total_ownership < 70 else "✅ Good" if total_ownership < 100 else "⚠️ Chalky"
        
        summary = f"Own: {total_ownership:.1f}% [{gpp_quality}] | "
        summary += " ".join([f"{bucket_emojis.get(k, '')} {k}:{v}" 
                           for k, v in bucket_counts.items() if v > 0])
        return summary
    
    @staticmethod
    def calculate_gpp_leverage(lineup_players: List[str], df: pd.DataFrame) -> float:
        """Calculate GPP-specific leverage score"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        leverage_score = 0
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            if ownership < 3:
                leverage_score += 5  # Huge bonus for super low owned
            elif ownership < 5:
                leverage_score += 3
            elif ownership < 10:
                leverage_score += 2
            elif ownership < 15:
                leverage_score += 1
            elif ownership > 35:
                leverage_score -= 2  # Penalty for mega chalk
            elif ownership > 25:
                leverage_score -= 1
        
        return leverage_score

# NFL GPP DUAL-AI OPTIMIZER - PART 2: CORE COMPONENTS (COMPLETE)
# Captain Pivots, Correlations, and Tournament Simulation

# ============================================================================
# GPP CAPTAIN PIVOT GENERATOR
# ============================================================================

class GPPCaptainPivotGenerator:
    """GPP-focused captain pivot generation"""
    
    @staticmethod
    def generate_gpp_pivots(lineup: Dict, df: pd.DataFrame, max_pivots: int = 5,
                           target_ownership: float = 15.0) -> List[Dict]:
        """Generate GPP-optimal captain pivot variations"""
        captain = lineup['Captain']
        flex_players = lineup['FLEX']
        
        salaries = df.set_index('Player')['Salary'].to_dict()
        points = df.set_index('Player')['Projected_Points'].to_dict()
        ownership = df.set_index('Player')['Ownership'].to_dict()
        positions = df.set_index('Player')['Position'].to_dict()
        
        pivot_lineups = []
        
        # Prioritize low-owned FLEX players for captain pivots
        flex_candidates = []
        for p in flex_players:
            own = ownership.get(p, 5)
            if own < 20:  # Only consider sub-20% for GPP pivots
                flex_candidates.append((p, own))
        
        # Sort by ownership (lowest first for max leverage)
        flex_candidates.sort(key=lambda x: x[1])
        
        for new_captain, captain_own in flex_candidates[:max_pivots]:
            old_captain_salary = salaries.get(captain, 0)
            new_captain_salary = salaries.get(new_captain, 0)
            
            salary_freed = old_captain_salary * 0.5
            salary_needed = new_captain_salary * 0.5
            
            if salary_freed >= salary_needed - 200:  # Allow small salary overrun for GPP
                new_flex = [p for p in flex_players if p != new_captain] + [captain]
                
                pivot_lineup = lineup.copy()
                pivot_lineup['Captain'] = new_captain
                pivot_lineup['FLEX'] = new_flex
                
                # GPP-specific pivot types
                if captain_own < 5:
                    pivot_type = '💎 Super Leverage'
                elif captain_own < 10:
                    pivot_type = '🟢 Leverage'
                elif captain_own < 15:
                    pivot_type = '🟡 Contrarian'
                else:
                    pivot_type = '⚠️ Standard'
                
                pivot_lineup['Pivot_Type'] = pivot_type
                pivot_lineup['Original_Captain'] = captain
                
                # Calculate GPP metrics
                total_proj = points.get(new_captain, 0) * 1.5 + sum(points.get(p, 0) for p in new_flex)
                total_own = ownership.get(new_captain, 5) * 1.5 + sum(ownership.get(p, 5) for p in new_flex)
                
                # GPP leverage calculation
                leverage_gain = (ownership.get(captain, 20) - captain_own) * 1.5
                
                pivot_lineup['Projected'] = round(total_proj, 2)
                pivot_lineup['Total_Ownership'] = round(total_own, 1)
                pivot_lineup['Captain_Own%'] = round(captain_own, 1)
                pivot_lineup['Leverage_Gain'] = round(leverage_gain, 1)
                pivot_lineup['GPP_Score'] = round(total_proj * (1 + leverage_gain/50), 1)
                pivot_lineup['Captain_Position'] = positions.get(new_captain, 'Unknown')
                pivot_lineup['Is_Elite_GPP'] = total_own < 70  # Elite GPP lineup threshold
                
                pivot_lineups.append(pivot_lineup)
        
        # Sort by GPP score
        pivot_lineups.sort(key=lambda x: x['GPP_Score'], reverse=True)
        return pivot_lineups
    
    @staticmethod
    def find_optimal_gpp_pivots(lineup: Dict, df: pd.DataFrame, 
                               field_size: str = 'large_field') -> List[Dict]:
        """Find pivots optimized for specific GPP field size"""
        
        # Get ownership targets for field size
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
        target_ownership = (min_own + max_own) / 2
        
        pivots = GPPCaptainPivotGenerator.generate_gpp_pivots(
            lineup, df, max_pivots=7, target_ownership=target_ownership
        )
        
        # Filter pivots by field size criteria
        if field_size == 'milly_maker':
            # Only super leverage pivots for Milly
            pivots = [p for p in pivots if p['Captain_Own%'] < 10]
        elif field_size == 'large_field':
            # Sub-15% captains for large field
            pivots = [p for p in pivots if p['Captain_Own%'] < 15]
        
        return pivots[:5]  # Return top 5

# ============================================================================
# GPP CORRELATION ENGINE
# ============================================================================

class GPPCorrelationEngine:
    """GPP-focused correlation calculations"""
    
    @staticmethod
    def calculate_gpp_correlations(df: pd.DataFrame, game_context: Dict) -> Dict[Tuple[str, str], float]:
        """Calculate GPP-specific correlations with emphasis on ceiling"""
        correlations = {}
        
        players = df['Player'].tolist()
        positions = df.set_index('Player')['Position'].to_dict()
        teams = df.set_index('Player')['Team'].to_dict()
        ownership = df.set_index('Player')['Ownership'].to_dict()
        
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                team1, team2 = teams.get(p1), teams.get(p2)
                pos1, pos2 = positions.get(p1), positions.get(p2)
                own1, own2 = ownership.get(p1, 5), ownership.get(p2, 5)
                
                correlation = 0
                leverage_bonus = 0  # GPP-specific bonus for low-owned correlations
                
                if team1 == team2:
                    # Same team correlations
                    if pos1 == 'QB' and pos2 in ['WR', 'TE']:
                        correlation = OptimizerConfig.QB_PASS_CATCHER_CORR
                        # GPP bonus for low-owned stacks
                        if own1 < 15 and own2 < 10:
                            leverage_bonus = 0.15
                        # Extra correlation in projected shootouts
                        if game_context.get('total', 48) > 52:
                            correlation += 0.15
                    elif pos1 == 'QB' and pos2 == 'RB':
                        correlation = OptimizerConfig.QB_RB_CORR
                        # More negative in blowouts (bad for GPP)
                        if abs(game_context.get('spread', 0)) > 7:
                            correlation -= 0.15
                    elif pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                        correlation = OptimizerConfig.SAME_TEAM_WR_CORR
                        # Reduce correlation for GPP (avoid doubling up on pass catchers)
                        if own1 > 20 or own2 > 20:
                            correlation -= 0.05
                    elif pos1 == 'RB' and pos2 == 'RB':
                        correlation = -0.6  # Very negative for GPP
                
                else:
                    # Opposing team correlations (game stacks)
                    if pos1 == 'QB' and pos2 == 'QB':
                        correlation = OptimizerConfig.OPPOSING_QB_CORR
                        # Much higher in projected shootouts
                        if game_context.get('total', 48) > 54:
                            correlation += 0.25
                        # GPP leverage bonus for low-owned QB stacks
                        if own1 < 20 and own2 < 20:
                            leverage_bonus = 0.20
                    elif pos1 == 'QB' and pos2 in ['WR', 'TE']:
                        # Bring-back correlation
                        correlation = 0.15
                        if game_context.get('total', 48) > 52:
                            correlation += 0.10
                    elif pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                        # Opposing pass catchers in shootout
                        if game_context.get('total', 48) > 52:
                            correlation = 0.10
                    elif 'DST' in [pos1, pos2]:
                        correlation = OptimizerConfig.DST_OPPOSING_CORR
                        # Even more negative in high-scoring games
                        if game_context.get('total', 48) > 50:
                            correlation -= 0.15
                
                # Apply leverage bonus
                final_correlation = correlation + leverage_bonus
                
                if abs(final_correlation) > 0.05:  # Only store meaningful correlations
                    correlations[(p1, p2)] = final_correlation
        
        return correlations
    
    @staticmethod
    def identify_gpp_stacks(df: pd.DataFrame, correlations: Dict, 
                           field_size: str = 'large_field') -> List[Dict]:
        """Identify GPP-optimal stacking opportunities"""
        stacks = []
        
        # Get ownership targets for field size
        if field_size == 'milly_maker':
            max_combined_own = 30
        elif field_size == 'large_field':
            max_combined_own = 40
        else:
            max_combined_own = 50
        
        # Find all positive correlations
        for (p1, p2), corr in correlations.items():
            if corr > 0.15:  # Meaningful positive correlation for GPP
                ownership1 = df[df['Player'] == p1]['Ownership'].values[0] if len(df[df['Player'] == p1]) > 0 else 5
                ownership2 = df[df['Player'] == p2]['Ownership'].values[0] if len(df[df['Player'] == p2]) > 0 else 5
                combined_own = ownership1 + ownership2
                
                # Calculate GPP stack score
                leverage_score = max(0, 50 - combined_own)
                gpp_score = corr * 100 + leverage_score
                
                # Determine stack type
                if combined_own < 20:
                    stack_type = '💎 Elite Leverage'
                elif combined_own < 30:
                    stack_type = '🟢 Leverage'
                elif combined_own < 40:
                    stack_type = '🟡 Contrarian'
                else:
                    stack_type = '⚠️ Chalky'
                
                if combined_own <= max_combined_own:  # Only include if meets field size criteria
                    stacks.append({
                        'player1': p1,
                        'player2': p2,
                        'correlation': corr,
                        'combined_ownership': combined_own,
                        'leverage': leverage_score,
                        'gpp_score': gpp_score,
                        'stack_type': stack_type,
                        'is_game_stack': df[df['Player'] == p1]['Team'].values[0] != df[df['Player'] == p2]['Team'].values[0]
                    })
        
        # Sort by GPP score
        stacks.sort(key=lambda x: x['gpp_score'], reverse=True)
        return stacks[:20]  # Return top 20 stacks

# ============================================================================
# GPP TOURNAMENT SIMULATOR WITH ROBUST ERROR HANDLING
# ============================================================================

class GPPTournamentSimulator:
    """GPP-specific tournament simulation with emphasis on ceiling"""
    
    @staticmethod
    def _ensure_positive_semidefinite(cov_matrix: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive semi-definite"""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Fix negative eigenvalues
        min_eigenvalue = np.min(eigenvalues)
        if min_eigenvalue < 0:
            # Add to diagonal to make positive semi-definite
            adjustment = abs(min_eigenvalue) + 1e-6
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * adjustment
        
        return cov_matrix
    
    @staticmethod
    def simulate_gpp_tournament(lineup: Dict, df: pd.DataFrame, 
                               correlations: Dict, n_sims: int = 5000,
                               field_size: str = 'large_field') -> Dict[str, float]:
        """GPP-focused simulation with higher variance and ceiling emphasis"""
        captain = lineup['Captain']
        flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else lineup['FLEX']
        all_players = [captain] + list(flex_players)
        
        projections = df.set_index('Player')['Projected_Points'].to_dict()
        ownership = df.set_index('Player')['Ownership'].to_dict()
        
        # Build correlation matrix
        n_players = len(all_players)
        means = np.array([projections.get(p, 0) for p in all_players])
        
        # GPP variance adjustment based on ownership and field size
        variance_multiplier = 1.0
        if field_size == 'milly_maker':
            variance_multiplier = 1.3  # Much higher variance for Milly
        elif field_size == 'large_field':
            variance_multiplier = 1.2
        
        variances = []
        for p in all_players:
            own = ownership.get(p, 5)
            # Lower ownership = higher variance (more GPP upside)
            ownership_mult = 1.0 + max(0, (20 - own) / 30)
            var = (OptimizerConfig.HIGH_VOLATILITY * variance_multiplier * ownership_mult) ** 2
            variances.append(var)
        
        # Create covariance matrix with adjusted variances
        cov_matrix = np.diag(variances)
        
        # Add correlations with safety bounds
        for i in range(n_players):
            for j in range(i+1, n_players):
                p1, p2 = all_players[i], all_players[j]
                pair = tuple(sorted([p1, p2]))
                if pair in correlations:
                    # Cap correlation to prevent matrix issues while maintaining GPP dynamics
                    raw_corr = correlations[pair]
                    # Keep high correlations for GPP but cap at safe levels
                    correlation = np.clip(raw_corr, -0.85, 0.85)
                    cov_value = correlation * np.sqrt(variances[i] * variances[j])
                    cov_matrix[i, j] = cov_value
                    cov_matrix[j, i] = cov_value
        
        # Ensure positive semi-definite
        cov_matrix = GPPTournamentSimulator._ensure_positive_semidefinite(cov_matrix)
        
        try:
            # Primary simulation method - multivariate normal with full correlations
            sims = multivariate_normal(mean=means, cov=cov_matrix, allow_singular=True).rvs(n_sims)
            sims = np.maximum(0, sims)  # No negative scores
            
        except np.linalg.LinAlgError:
            # Backup method 1: Reduce correlation strength slightly
            try:
                cov_matrix_reduced = np.diag(variances)
                for i in range(n_players):
                    for j in range(i+1, n_players):
                        p1, p2 = all_players[i], all_players[j]
                        pair = tuple(sorted([p1, p2]))
                        if pair in correlations:
                            # Reduce correlations by 20% for stability
                            correlation = np.clip(correlations[pair] * 0.8, -0.7, 0.7)
                            cov_value = correlation * np.sqrt(variances[i] * variances[j])
                            cov_matrix_reduced[i, j] = cov_value
                            cov_matrix_reduced[j, i] = cov_value
                
                cov_matrix_reduced = GPPTournamentSimulator._ensure_positive_semidefinite(cov_matrix_reduced)
                sims = multivariate_normal(mean=means, cov=cov_matrix_reduced, allow_singular=True).rvs(n_sims)
                sims = np.maximum(0, sims)
                
            except:
                # Backup method 2: Cholesky decomposition approach
                try:
                    # Use Cholesky decomposition for more stable generation
                    L = np.linalg.cholesky(cov_matrix + np.eye(n_players) * 1e-6)
                    z = np.random.standard_normal((n_sims, n_players))
                    sims = means + z @ L.T
                    sims = np.maximum(0, sims)
                    
                except:
                    # Final fallback: Correlated normal with simplified approach
                    sims = np.zeros((n_sims, n_players))
                    
                    # Generate base simulations
                    for i in range(n_players):
                        sims[:, i] = np.random.normal(means[i], np.sqrt(variances[i]), n_sims)
                    
                    # Apply correlations through mixing
                    for i in range(n_players):
                        for j in range(i+1, n_players):
                            p1, p2 = all_players[i], all_players[j]
                            pair = tuple(sorted([p1, p2]))
                            if pair in correlations:
                                corr = min(0.6, max(-0.6, correlations[pair]))
                                if abs(corr) > 0.1:
                                    # Mix signals to create correlation
                                    if corr > 0:
                                        sims[:, j] = (1 - abs(corr)) * sims[:, j] + abs(corr) * sims[:, i]
                                    else:
                                        sims[:, j] = (1 - abs(corr)) * sims[:, j] - abs(corr) * sims[:, i]
                    
                    sims = np.maximum(0, sims)
        
        # Apply GPP-specific volatility events (same for all methods)
        for i in range(n_players):
            own = ownership.get(all_players[i], 5)
            
            # Injury risk (higher for chalky players in GPP)
            injury_rate = OptimizerConfig.INJURY_RATE
            if own > 30:  # Higher injury risk for chalk
                injury_rate *= 1.5
            injury_mask = np.random.random(n_sims) < injury_rate
            sims[injury_mask, i] *= np.random.uniform(0, 0.15, injury_mask.sum())
            
            # Boom potential (much higher for low-owned players) - KEY FOR GPP
            if own < 5:
                boom_rate = OptimizerConfig.BOOM_RATE * 3
                boom_range = (3.0, 5.0)  # Massive booms for super leverage
            elif own < 10:
                boom_rate = OptimizerConfig.BOOM_RATE * 2
                boom_range = (2.5, 4.0)
            elif own < 15:
                boom_rate = OptimizerConfig.BOOM_RATE * 1.5
                boom_range = (2.0, 3.0)
            else:
                boom_rate = OptimizerConfig.BOOM_RATE
                boom_range = (1.5, 2.5)
            
            boom_mask = np.random.random(n_sims) < boom_rate
            sims[boom_mask, i] *= np.random.uniform(boom_range[0], boom_range[1], boom_mask.sum())
        
        # Calculate lineup scores
        captain_scores = sims[:, 0] * OptimizerConfig.CAPTAIN_MULTIPLIER
        flex_scores = np.sum(sims[:, 1:], axis=1)
        total_scores = captain_scores + flex_scores
        
        # GPP-specific metrics - comprehensive for tournament play
        return {
            'Mean': round(np.mean(total_scores), 2),
            'Std': round(np.std(total_scores), 2),
            'Floor_10th': round(np.percentile(total_scores, 10), 2),
            'Floor_25th': round(np.percentile(total_scores, 25), 2),
            'Median': round(np.percentile(total_scores, 50), 2),
            'Ceiling_75th': round(np.percentile(total_scores, 75), 2),
            'Ceiling_90th': round(np.percentile(total_scores, 90), 2),
            'Ceiling_95th': round(np.percentile(total_scores, 95), 2),
            'Ceiling_98th': round(np.percentile(total_scores, 98), 2),
            'Ceiling_99th': round(np.percentile(total_scores, 99), 2),
            'Ceiling_99_9th': round(np.percentile(total_scores, 99.9), 2),
            'Boom_Rate': round(np.mean(total_scores > np.percentile(total_scores, 95)) * 100, 1),
            'Elite_Rate': round(np.mean(total_scores > np.percentile(total_scores, 99)) * 100, 2),
            'Ship_Rate': round(np.mean(total_scores > np.percentile(total_scores, 99.9)) * 100, 3),
            'Bust_Rate': round(np.mean(total_scores < np.percentile(total_scores, 25)) * 100, 1),
            'Max_Score': round(np.max(total_scores), 2),
            'Top_1pct_Avg': round(np.mean(np.sort(total_scores)[-int(n_sims*0.01):]), 2)
        }
    
    @staticmethod
    def calculate_gpp_win_probability(lineup_scores: np.ndarray, 
                                     field_size_num: int = 100000) -> Dict[str, float]:
        """Calculate GPP tournament win probabilities"""
        # Model GPP field scores with right-skewed distribution
        field_mean = 85
        field_shape = 3.5  # Lower shape = more right skew for GPP
        field_scale = field_mean / field_shape
        
        # Simulate field scores
        field_scores = gamma.rvs(field_shape, scale=field_scale, size=field_size_num)
        
        # Add some extreme scores (GPP has more variance)
        num_elite = int(field_size_num * 0.001)  # Top 0.1% are elite
        elite_scores = gamma.rvs(5, scale=30, size=num_elite) + 100
        field_scores[:num_elite] = elite_scores
        
        win_prob = 0
        top_10_prob = 0
        top_100_prob = 0
        top_1pct_prob = 0
        min_cash = int(field_size_num * 0.2)  # Top 20% cash
        cash_prob = 0
        
        sample_size = min(1000, len(lineup_scores))
        for score in np.random.choice(lineup_scores, sample_size):
            placement = np.sum(score > field_scores)
            
            if placement == field_size_num - 1:
                win_prob += 1
            if placement >= field_size_num - 10:
                top_10_prob += 1
            if placement >= field_size_num - 100:
                top_100_prob += 1
            if placement >= field_size_num * 0.99:
                top_1pct_prob += 1
            if placement >= field_size_num - min_cash:
                cash_prob += 1
        
        return {
            'Win_Prob': round(win_prob / sample_size * 100, 4),
            'Top_10_Prob': round(top_10_prob / sample_size * 100, 3),
            'Top_100_Prob': round(top_100_prob / sample_size * 100, 2),
            'Top_1pct_Prob': round(top_1pct_prob / sample_size * 100, 2),
            'Min_Cash_Prob': round(cash_prob / sample_size * 100, 1)
        }

# ============================================================================
# GPP SCORING CALCULATOR
# ============================================================================

def calculate_gpp_scores(lineups_df: pd.DataFrame, field_size: str = 'large_field') -> pd.DataFrame:
    """Calculate GPP-specific scores with heavy ceiling emphasis"""
    
    # GPP scoring weights based on field size
    if field_size == 'small_field':
        weights = {
            'ceiling_95': 0.25,
            'ceiling_99': 0.20,
            'ceiling_99_9': 0.10,
            'ownership': 0.20,
            'leverage': 0.15,
            'correlation': 0.10
        }
    elif field_size == 'medium_field':
        weights = {
            'ceiling_95': 0.20,
            'ceiling_99': 0.25,
            'ceiling_99_9': 0.15,
            'ownership': 0.20,
            'leverage': 0.15,
            'correlation': 0.05
        }
    elif field_size == 'large_field':
        weights = {
            'ceiling_95': 0.15,
            'ceiling_99': 0.30,
            'ceiling_99_9': 0.20,
            'ownership': 0.15,
            'leverage': 0.15,
            'correlation': 0.05
        }
    else:  # milly_maker
        weights = {
            'ceiling_95': 0.10,
            'ceiling_99': 0.25,
            'ceiling_99_9': 0.30,
            'ownership': 0.15,
            'leverage': 0.20,
            'correlation': 0.00
        }
    
    # Calculate GPP score
    lineups_df['GPP_Score'] = 0
    
    # Ceiling components
    if 'Ceiling_95th' in lineups_df.columns:
        lineups_df['GPP_Score'] += weights.get('ceiling_95', 0.2) * lineups_df['Ceiling_95th'] * 0.5
    
    if 'Ceiling_99th' in lineups_df.columns:
        lineups_df['GPP_Score'] += weights.get('ceiling_99', 0.25) * lineups_df['Ceiling_99th'] * 0.6
    
    if 'Ceiling_99_9th' in lineups_df.columns:
        lineups_df['GPP_Score'] += weights.get('ceiling_99_9', 0.15) * lineups_df['Ceiling_99_9th'] * 0.7
    
    # Ownership component (lower is better for GPP)
    optimal_ownership = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size][0] + 10
    lineups_df['GPP_Score'] += weights.get('ownership', 0.15) * (
        150 - np.abs(lineups_df['Total_Ownership'] - optimal_ownership)
    ) * 0.5
    
    # Leverage score component
    lineups_df['GPP_Score'] += weights.get('leverage', 0.15) * lineups_df['Leverage_Score'] * 8
    
    # Correlation bonus
    lineups_df['GPP_Score'] += weights.get('correlation', 0.05) * lineups_df['Has_Stack'].astype(int) * 40
    
    # Additional GPP metrics
    if 'Ship_Rate' in lineups_df.columns:
        lineups_df['Ship_Equity'] = lineups_df['Ship_Rate'] * 1000
    
    if 'Elite_Rate' in lineups_df.columns:
        lineups_df['Elite_Equity'] = lineups_df['Elite_Rate'] * 100
    
    # Tournament equity score (combines all factors)
    lineups_df['Tournament_EV'] = (
        lineups_df['GPP_Score'] * 0.6 +
        lineups_df.get('Ship_Equity', 0) * 0.25 +
        lineups_df.get('Elite_Equity', 0) * 0.15
    )
    
    return lineups_df

# NFL GPP DUAL-AI OPTIMIZER - PART 3: AI STRATEGISTS
# Game Theory and Correlation AI with GPP Focus

# ============================================================================
# GPP GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist:
    """AI Strategist 1: GPP Game Theory and ownership leverage"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate GPP-focused game theory prompt"""
        
        bucket_manager = OwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)
        
        # Get ownership targets for field size
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
        
        # Get key players by GPP categories
        mega_chalk = df[df['Player'].isin(buckets.get('mega_chalk', []))].nlargest(5, 'Ownership')[
            ['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        super_leverage = df[df['Player'].isin(buckets.get('super_leverage', []))].nlargest(
            10, 'Projected_Points')[['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        leverage_plays = df[df['Player'].isin(buckets.get('leverage', []))].nlargest(
            10, 'Projected_Points')[['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        # Calculate GPP value plays (high ceiling, low ownership)
        df['GPP_Value'] = (df['Projected_Points'] / (df['Salary'] / 1000)) * (20 / (df['Ownership'] + 5))
        top_gpp_values = df.nlargest(10, 'GPP_Value')[['Player', 'Position', 'GPP_Value', 'Ownership']]
        
        # Field size strategy
        field_strategy = {
            'small_field': "Focus on slightly contrarian plays with 80-120% cumulative ownership",
            'medium_field': "Target 70-100% ownership with at least 2 leverage plays",
            'large_field': "Maximize leverage with 60-90% ownership, fade mega chalk",
            'milly_maker': "Ultra-contrarian with 50-80% ownership, zero chalk tolerance"
        }
        
        return f"""
        As an expert GPP tournament strategist, analyze this NFL Showdown slate for {field_size.upper()} tournaments:
        
        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        Field Type: {field_size} ({field_strategy.get(field_size, 'Standard GPP')})
        
        MEGA CHALK (35%+ ownership - AVOID IN GPP):
        {mega_chalk.to_string() if not mega_chalk.empty else 'None'}
        
        SUPER LEVERAGE (<5% ownership - GPP GOLD):
        {super_leverage.to_string() if not super_leverage.empty else 'None'}
        
        LEVERAGE PLAYS (5-10% ownership - GPP TARGETS):
        {leverage_plays.to_string() if not leverage_plays.empty else 'None'}
        
        TOP GPP VALUE PLAYS (Ceiling + Low Ownership):
        {top_gpp_values.to_string()}
        
        OWNERSHIP DISTRIBUTION:
        - Mega Chalk (35%+): {len(buckets.get('mega_chalk', []))} players
        - Chalk (20-35%): {len(buckets.get('chalk', []))} players  
        - Pivot (10-20%): {len(buckets.get('pivot', []))} players
        - Leverage (5-10%): {len(buckets.get('leverage', []))} players
        - Super Leverage (<5%): {len(buckets.get('super_leverage', []))} players
        
        GPP STRATEGY REQUIREMENTS:
        - Target cumulative ownership: {min_own}-{max_own}%
        - Prioritize ceiling over floor
        - Identify tournament-winning leverage
        
        Provide GPP-specific recommendations:
        1. Which chalk is actually bad chalk (low ceiling despite ownership)?
        2. Which low-owned players have tournament-winning upside?
        3. Optimal captain leverage plays (sub-15% ownership)?
        4. How to differentiate lineups while maintaining projection?
        5. Contrarian game theory for this specific field size?
        
        Return ONLY valid JSON:
        {{
            "gpp_captain_targets": ["leverage_captain1", "leverage_captain2", "leverage_captain3"],
            "super_leverage_plays": ["gem1", "gem2", "gem3"],
            "must_fade_chalk": ["overowned1", "overowned2"],
            "tournament_winners": ["upside1", "upside2"],
            "gpp_construction_rules": {{
                "max_chalk_players": 1,
                "min_sub10_ownership": 2,
                "target_total_ownership": {{"min": {min_own}, "max": {max_own}}},
                "required_leverage_captain": true
            }},
            "leverage_stacks": [
                {{"player1": "low_own1", "player2": "low_own2", "combined_ownership": 15}}
            ],
            "differentiation_strategy": "specific approach for uniqueness",
            "field_size_adjustments": "specific tweaks for {field_size}",
            "gpp_insights": [
                "Key tournament insight 1",
                "Key tournament insight 2"
            ],
            "win_probability_boosters": ["factor1", "factor2"],
            "confidence_score": 0.85
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse and validate GPP game theory response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Extract GPP construction rules
        rules = data.get('gpp_construction_rules', {})
        max_chalk = rules.get('max_chalk_players', 1)
        min_leverage = rules.get('min_sub10_ownership', 2)
        
        # Determine GPP strategy weights based on field size
        if field_size == 'milly_maker':
            strategy_weights = {
                StrategyType.SUPER_CONTRARIAN: 0.4,
                StrategyType.LEVERAGE: 0.35,
                StrategyType.CONTRARIAN: 0.2,
                StrategyType.STARS_SCRUBS: 0.05,
                StrategyType.CORRELATION: 0.0,
                StrategyType.GAME_STACK: 0.0
            }
        elif field_size == 'large_field':
            strategy_weights = {
                StrategyType.LEVERAGE: 0.35,
                StrategyType.CONTRARIAN: 0.25,
                StrategyType.SUPER_CONTRARIAN: 0.15,
                StrategyType.GAME_STACK: 0.15,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.CORRELATION: 0.0
            }
        else:
            strategy_weights = {
                StrategyType.LEVERAGE: 0.25,
                StrategyType.CONTRARIAN: 0.15,
                StrategyType.GAME_STACK: 0.25,
                StrategyType.CORRELATION: 0.2,
                StrategyType.STARS_SCRUBS: 0.15,
                StrategyType.SUPER_CONTRARIAN: 0.0
            }
        
        # GPP-specific rules
        gpp_rules = {
            'max_chalk': max_chalk,
            'min_leverage': min_leverage,
            'force_leverage_captain': rules.get('required_leverage_captain', True),
            'ownership_targets': rules.get('target_total_ownership', 
                                         OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size])
        }
        
        return AIRecommendation(
            strategist_name="GPP Game Theory AI",
            confidence=data.get('confidence_score', 0.80),
            captain_targets=data.get('gpp_captain_targets', []),
            stacks=[{'player1': s.get('player1'), 'player2': s.get('player2')} 
                   for s in data.get('leverage_stacks', []) if isinstance(s, dict)],
            fades=data.get('must_fade_chalk', []),
            boosts=data.get('super_leverage_plays', []) + data.get('tournament_winners', []),
            strategy_weights=strategy_weights,
            key_insights=data.get('gpp_insights', ["Using default GPP strategy"]),
            gpp_specific_rules=gpp_rules
        )

# ============================================================================
# GPP CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist:
    """AI Strategist 2: GPP correlation and game stack specialist"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate GPP correlation analysis prompt"""
        
        teams = df['Team'].unique()[:2]
        team_breakdown = {}
        
        for team in teams:
            team_df = df[df['Team'] == team]
            
            # Get players with ownership context for GPP
            qbs = team_df[team_df['Position'] == 'QB'][
                ['Player', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            pass_catchers = team_df[team_df['Position'].isin(['WR', 'TE'])][
                ['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']
            ].sort_values('Ownership').to_dict('records')  # Sort by ownership for GPP
            rbs = team_df[team_df['Position'] == 'RB'][
                ['Player', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            
            team_breakdown[team] = {
                'QB': qbs,
                'Pass_Catchers': pass_catchers[:6],  # Include more for GPP variety
                'RB': rbs[:3]
            }
        
        # Determine correlation strategy based on total
        if game_info.get('total', 48) > 54:
            correlation_focus = "SHOOTOUT: Prioritize game stacks with 4+ players from game"
        elif game_info.get('total', 48) > 50:
            correlation_focus = "MODERATE: Mix of game stacks and single team stacks"
        else:
            correlation_focus = "LOW TOTAL: Focus on single team stacks, leverage RBs"
        
        return f"""
        As an expert GPP Correlation strategist, identify tournament-winning stacks for {field_size} GPPs:
        
        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        Correlation Strategy: {correlation_focus}
        
        TEAM ROSTERS WITH OWNERSHIP:
        {json.dumps(team_breakdown, indent=2)}
        
        GPP CORRELATION PRIORITIES:
        1. Leverage stacks (combined ownership < 30%)
        2. Game stacks for ceiling in projected shootouts
        3. Contrarian bring-backs (opposing team correlation)
        4. Low-owned QB stacks (QB < 20% ownership)
        5. Secondary stacks that differentiate lineups
        6. Negative correlations to avoid (RB-RB, etc.)
        
        Field Size Considerations:
        - Small field: Can use one higher-owned stack
        - Large field: Need multiple low-owned correlations
        - Milly Maker: Only super-leverage stacks
        
        Return ONLY valid JSON:
        {{
            "primary_leverage_stacks": [
                {{"qb": "name", "receiver": "name", "correlation": 0.6, "combined_ownership": 20, "gpp_score": 85}}
            ],
            "game_stacks": [
                {{"players": ["p1", "p2", "p3", "p4"], "scenario": "shootout", "total_ownership": 60, "ceiling_boost": "high"}}
            ],
            "super_leverage_stacks": [
                {{"player1": "sub5own", "player2": "sub10own", "combined_ownership": 12, "tournament_equity": "elite"}}
            ],
            "contrarian_bringbacks": [
                {{"primary": "team1_qb", "bringback": "team2_wr2", "leverage": "high"}}
            ],
            "avoid_correlations": [
                {{"player1": "rb1", "player2": "rb2", "reason": "negative correlation"}}
            ],
            "gpp_game_script": {{
                "most_likely": "scenario",
                "leverage_scenario": "contrarian_scenario",
                "boost_players": ["player1", "player2"],
                "fade_players": ["chalk1", "chalk2"]
            }},
            "field_specific_stacks": {{
                "small_field": ["balanced_stack"],
                "large_field": ["leverage_stack"],
                "milly_maker": ["super_leverage_stack"]
            }},
            "correlation_insights": [
                "GPP correlation insight 1",
                "GPP correlation insight 2"
            ],
            "stack_differentiation": "how to make stacks unique",
            "confidence": 0.85
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse and validate GPP correlation response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Build comprehensive GPP stacks list
        all_stacks = []
        
        # Primary leverage stacks
        for stack in data.get('primary_leverage_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('qb'),
                    'player2': stack.get('receiver'),
                    'type': 'leverage_primary',
                    'correlation': stack.get('correlation', 0.5),
                    'gpp_score': stack.get('gpp_score', 50)
                })
        
        # Game stacks
        for stack in data.get('game_stacks', []):
            if isinstance(stack, dict):
                players = stack.get('players', [])
                if len(players) >= 2:
                    all_stacks.append({
                        'player1': players[0],
                        'player2': players[1],
                        'type': 'game_stack',
                        'correlation': 0.35,
                        'additional_players': players[2:] if len(players) > 2 else []
                    })
        
        # Super leverage stacks
        for stack in data.get('super_leverage_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('player1'),
                    'player2': stack.get('player2'),
                    'type': 'super_leverage',
                    'correlation': 0.3,
                    'tournament_equity': stack.get('tournament_equity', 'high')
                })
        
        # Determine strategy weights based on game script and field size
        game_script = data.get('gpp_game_script', {}).get('most_likely', 'balanced')
        
        if field_size == 'milly_maker':
            strategy_weights = {
                StrategyType.SUPER_CONTRARIAN: 0.35,
                StrategyType.LEVERAGE: 0.35,
                StrategyType.GAME_STACK: 0.2 if game_script == 'shootout' else 0.1,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.CONTRARIAN: 0.0,
                StrategyType.CORRELATION: 0.0
            }
        elif game_script == 'shootout':
            strategy_weights = {
                StrategyType.GAME_STACK: 0.35,
                StrategyType.CORRELATION: 0.25,
                StrategyType.LEVERAGE: 0.2,
                StrategyType.CONTRARIAN: 0.15,
                StrategyType.STARS_SCRUBS: 0.05,
                StrategyType.SUPER_CONTRARIAN: 0.0
            }
        else:
            strategy_weights = {
                StrategyType.LEVERAGE: 0.3,
                StrategyType.CONTRARIAN: 0.25,
                StrategyType.CORRELATION: 0.2,
                StrategyType.GAME_STACK: 0.15,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.SUPER_CONTRARIAN: 0.0
            }
        
        # Extract captain targets from low-owned QBs
        captain_targets = []
        for stack in data.get('primary_leverage_stacks', []):
            if isinstance(stack, dict) and stack.get('qb'):
                qb = stack['qb']
                if qb in df['Player'].values:
                    qb_own = df[df['Player'] == qb]['Ownership'].values[0]
                    if qb_own < 20:  # Only low-owned QBs for GPP
                        captain_targets.append(qb)
        
        # GPP-specific rules
        gpp_rules = {
            'leverage_scenario': data.get('gpp_game_script', {}).get('leverage_scenario'),
            'field_stacks': data.get('field_specific_stacks', {}).get(field_size, []),
            'avoid_together': data.get('avoid_correlations', [])
        }
        
        return AIRecommendation(
            strategist_name="GPP Correlation AI",
            confidence=data.get('confidence', 0.80),
            captain_targets=captain_targets,
            stacks=all_stacks,
            fades=data.get('gpp_game_script', {}).get('fade_players', []),
            boosts=data.get('gpp_game_script', {}).get('boost_players', []),
            strategy_weights=strategy_weights,
            key_insights=data.get('correlation_insights', ["Using GPP correlation strategy"]),
            gpp_specific_rules=gpp_rules
        )

# NFL GPP DUAL-AI OPTIMIZER - PART 4: MAIN OPTIMIZER & LINEUP GENERATION (CORRECTED)

# ============================================================================
# GPP DUAL AI OPTIMIZER
# ============================================================================

class GPPDualAIOptimizer:
    """Main GPP optimizer with dual AI strategy integration"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field', 
                 api_manager: ClaudeAPIManager = None):
        self.df = df
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.bucket_manager = OwnershipBucketManager()
        self.pivot_generator = GPPCaptainPivotGenerator()
        self.correlation_engine = GPPCorrelationEngine()
        self.tournament_sim = GPPTournamentSimulator()
        self.optimization_logger = []  # Track optimization issues
    
    def safe_api_call(self, prompt: str, strategist_name: str, fallback_response: str = '{}') -> str:
        """Safely make API calls with fallback"""
        if not self.api_manager or not self.api_manager.client:
            return fallback_response
        
        try:
            response = self.api_manager.get_ai_response(prompt)
            # Validate it's actual JSON
            json.loads(response)
            return response
        except json.JSONDecodeError as e:
            st.warning(f"{strategist_name}: Invalid JSON response. Using fallback strategy.")
            self.optimization_logger.append(f"JSON decode error: {e}")
            return fallback_response
        except Exception as e:
            st.warning(f"{strategist_name}: API call failed. Using fallback strategy.")
            self.optimization_logger.append(f"API error: {e}")
            return fallback_response
    
    def get_ai_strategies(self, use_api: bool = True) -> Tuple[AIRecommendation, AIRecommendation]:
        """Get strategies from both GPP AIs with error handling"""
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode - automatic with safe calls
            with st.spinner("🎯 GPP Game Theory AI analyzing..."):
                gt_prompt = self.game_theory_ai.generate_prompt(self.df, self.game_info, self.field_size)
                gt_response = self.safe_api_call(gt_prompt, "Game Theory AI")
            
            with st.spinner("🔗 GPP Correlation AI analyzing..."):
                corr_prompt = self.correlation_ai.generate_prompt(self.df, self.game_info, self.field_size)
                corr_response = self.safe_api_call(corr_prompt, "Correlation AI")
        else:
            # Manual mode
            st.subheader("📝 Manual AI Strategy Input")
            
            tab1, tab2 = st.tabs(["🎯 Game Theory AI", "🔗 Correlation AI"])
            
            with tab1:
                with st.expander("View GPP Game Theory Prompt"):
                    st.text_area("Copy this prompt:", 
                               value=self.game_theory_ai.generate_prompt(self.df, self.game_info, self.field_size),
                               height=300, key="gt_prompt_display")
                gt_response = st.text_area("Paste Game Theory Response (JSON):", 
                                          height=200, key="gt_manual_input",
                                          value='{}')
            
            with tab2:
                with st.expander("View GPP Correlation Prompt"):
                    st.text_area("Copy this prompt:", 
                               value=self.correlation_ai.generate_prompt(self.df, self.game_info, self.field_size),
                               height=300, key="corr_prompt_display")
                corr_response = st.text_area("Paste Correlation Response (JSON):", 
                                            height=200, key="corr_manual_input",
                                            value='{}')
        
        rec1 = self.game_theory_ai.parse_response(gt_response, self.df, self.field_size)
        rec2 = self.correlation_ai.parse_response(corr_response, self.df, self.field_size)
        
        return rec1, rec2
    
    def combine_gpp_recommendations(self, rec1: AIRecommendation, rec2: AIRecommendation) -> Dict:
        """Combine recommendations with GPP-specific weighting"""
        
        total_confidence = rec1.confidence + rec2.confidence
        w1 = rec1.confidence / total_confidence if total_confidence > 0 else 0.5
        w2 = rec2.confidence / total_confidence if total_confidence > 0 else 0.5
        
        # Combine captain targets with GPP scoring
        all_captains = set(rec1.captain_targets + rec2.captain_targets)
        captain_scores = {}
        ownership_dict = self.df.set_index('Player')['Ownership'].to_dict()
        
        for captain in all_captains:
            score = 0
            ownership = ownership_dict.get(captain, 5)
            
            # Base score from AI recommendations
            if captain in rec1.captain_targets:
                score += w1
            if captain in rec2.captain_targets:
                score += w2
            
            # GPP leverage bonus (lower ownership = higher score)
            if ownership < 5:
                score *= 2.0
            elif ownership < 10:
                score *= 1.5
            elif ownership < 15:
                score *= 1.2
            elif ownership > 30:
                score *= 0.5
            elif ownership > 20:
                score *= 0.7
            
            captain_scores[captain] = score
        
        # Get GPP strategy weights for field size
        strategy_weights = OptimizerConfig.GPP_STRATEGY_WEIGHTS[self.field_size]
        
        # Merge GPP-specific rules
        gpp_rules = {}
        if hasattr(rec1, 'gpp_specific_rules'):
            gpp_rules.update(rec1.gpp_specific_rules)
        if hasattr(rec2, 'gpp_specific_rules'):
            gpp_rules.update(rec2.gpp_specific_rules)
        
        return {
            'captain_scores': captain_scores,
            'strategy_weights': strategy_weights,
            'consensus_fades': list(set(rec1.fades) & set(rec2.fades)) if rec1.fades and rec2.fades else [],
            'all_boosts': list(set(rec1.boosts) | set(rec2.boosts)) if rec1.boosts or rec2.boosts else [],
            'combined_stacks': rec1.stacks + rec2.stacks,
            'confidence': (rec1.confidence + rec2.confidence) / 2,
            'insights': rec1.key_insights + rec2.key_insights,
            'gpp_rules': gpp_rules,
            'field_size': self.field_size
        }
    
    def validate_lineup_constraints(self, captain: str, flex_players: List[str], 
                                  salaries: Dict, ownership: Dict) -> Tuple[bool, str]:
        """Validate that a lineup meets all constraints"""
        # Check basic counts
        if not captain or len(flex_players) != 5:
            return False, "Invalid lineup structure"
        
        # Check salary
        total_salary = salaries.get(captain, 0) * 1.5 + sum(salaries.get(p, 0) for p in flex_players)
        if total_salary > OptimizerConfig.SALARY_CAP:
            return False, f"Salary {total_salary} exceeds cap"
        
        # Check ownership for field size
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[self.field_size]
        total_ownership = ownership.get(captain, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_players)
        
        if total_ownership < min_own - 10:  # Allow some flexibility
            return False, f"Ownership {total_ownership:.1f}% below minimum {min_own}%"
        if total_ownership > max_own + 10:
            return False, f"Ownership {total_ownership:.1f}% above maximum {max_own}%"
        
        return True, "Valid"
    
    def generate_gpp_lineups(self, num_lineups: int, rec1: AIRecommendation, 
                            rec2: AIRecommendation, force_unique_captains: bool = True) -> pd.DataFrame:
        """Generate GPP-optimized lineups with improved constraint handling"""
        
        combined = self.combine_gpp_recommendations(rec1, rec2)
        
        # Get data
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Validate we have enough players
        if len(players) < 6:
            st.error(f"Not enough players ({len(players)}) to create lineups")
            return pd.DataFrame()
        
        # Apply GPP AI adjustments more aggressively
        adjusted_points = points.copy()
        for player in combined['consensus_fades']:
            if player in adjusted_points and ownership.get(player, 5) > 30:
                adjusted_points[player] *= 0.70  # Heavy fade for GPP chalk
        
        for player in combined['all_boosts']:
            if player in adjusted_points and ownership.get(player, 5) < 10:
                adjusted_points[player] *= 1.25  # Strong boost for leverage
        
        # Get ownership buckets
        player_buckets = {}
        for player in players:
            player_buckets[player] = self.bucket_manager.get_bucket(
                ownership.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            )
        
        # Get ownership targets for field size
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[self.field_size]
        
        # Distribute lineups across GPP strategies
        strategy_weights = combined['strategy_weights']
        lineups_per_strategy = {}
        for strategy in StrategyType:
            if strategy in strategy_weights:
                weight = strategy_weights.get(strategy, 0)
                lineups_per_strategy[strategy] = max(0, int(num_lineups * weight))
        
        # Ensure we hit target number
        total_assigned = sum(lineups_per_strategy.values())
        if total_assigned < num_lineups:
            lineups_per_strategy[StrategyType.LEVERAGE] = lineups_per_strategy.get(
                StrategyType.LEVERAGE, 0) + (num_lineups - total_assigned)
        
        all_lineups = []
        used_captains = set()
        lineup_num = 0
        failed_attempts = 0
        max_failures = 50  # Prevent infinite loops
        
        for strategy, count in lineups_per_strategy.items():
            strategy_attempts = 0
            strategy_max_attempts = count * 3  # Allow 3x attempts per lineup
            
            for i in range(count):
                if strategy_attempts > strategy_max_attempts:
                    self.optimization_logger.append(f"Max attempts reached for {strategy.value}")
                    break
                    
                lineup_num += 1
                
                model = pulp.LpProblem(f"GPP_Lineup_{lineup_num}_{strategy.value}", pulp.LpMaximize)
                flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
                
                # GPP-SPECIFIC OBJECTIVE FUNCTIONS
                if strategy == StrategyType.LEVERAGE:
                    # Maximum leverage for GPP
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 + max(0, 20 - ownership.get(p, 10))/20) +
                        1.5 * adjusted_points[p] * captain[p] * (1 + max(0, 15 - ownership.get(p, 10))/15)
                        for p in players
                    ])
                
                elif strategy == StrategyType.CONTRARIAN:
                    # Contrarian GPP approach
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 - ownership.get(p, 5)/80) +
                        1.5 * adjusted_points[p] * captain[p] * (1 - ownership.get(p, 5)/60)
                        for p in players
                    ])
                
                elif strategy == StrategyType.SUPER_CONTRARIAN:
                    # Ultra-contrarian for large GPP
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 - ownership.get(p, 5)/50) +
                        1.5 * adjusted_points[p] * captain[p] * (1 - ownership.get(p, 5)/40)
                        for p in players
                    ])
                
                elif strategy == StrategyType.GAME_STACK:
                    # Game stack with correlation bonus
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 + (self.game_info.get('total', 48) > 52) * 0.1) +
                        1.5 * adjusted_points[p] * captain[p]
                        for p in players
                    ])
                
                elif strategy == StrategyType.STARS_SCRUBS:
                    # High variance construction
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 + (salaries[p] < 3500 or salaries[p] > 9000) * 0.25) +
                        1.5 * adjusted_points[p] * captain[p]
                        for p in players
                    ])
                
                else:  # CORRELATION
                    # Standard GPP objective
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] + 1.5 * adjusted_points[p] * captain[p]
                        for p in players
                    ])
                
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
                for team in self.df['Team'].unique():
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
                
                # GPP-SPECIFIC CONSTRAINTS (FIXED)
                
                # CRITICAL FIX: Actually constrain total ownership
                ownership_expr = pulp.lpSum([
                    ownership.get(p, 5) * (flex[p] + 0.5 * captain[p])
                    for p in players
                ])
                
                # Add flexibility for edge cases but still constrain
                model += ownership_expr >= max(0, min_own - 10)
                model += ownership_expr <= max_own + 10
                
                # Field-size specific chalk constraints
                if self.field_size == 'milly_maker':
                    # Ultra strict for Milly
                    high_ownership_players = [p for p in players if ownership.get(p, 5) > 25]
                    if high_ownership_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in high_ownership_players]) <= 0
                    
                    # Force super leverage
                    super_low_owned = [p for p in players if ownership.get(p, 5) < 5]
                    if super_low_owned and len(super_low_owned) >= 2:
                        model += pulp.lpSum([flex[p] + captain[p] for p in super_low_owned]) >= 2
                
                elif self.field_size == 'large_field':
                    # Strict for large field
                    high_ownership_players = [p for p in players if ownership.get(p, 5) > 30]
                    if high_ownership_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in high_ownership_players]) <= 1
                    
                    low_owned = [p for p in players if ownership.get(p, 5) < 10]
                    if low_owned and len(low_owned) >= 2:
                        model += pulp.lpSum([flex[p] + captain[p] for p in low_owned]) >= 2
                
                else:
                    # More flexible for smaller fields
                    high_ownership_players = [p for p in players if ownership.get(p, 5) > 35]
                    if high_ownership_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in high_ownership_players]) <= 2
                
                # Force unique captains if enabled
                if force_unique_captains and used_captains:
                    for prev_captain in used_captains:
                        if prev_captain in players:
                            model += captain[prev_captain] == 0
                
                # Force leverage captain for certain strategies
                if strategy in [StrategyType.LEVERAGE, StrategyType.SUPER_CONTRARIAN]:
                    leverage_captains = [p for p in players if ownership.get(p, 5) < 15]
                    if leverage_captains:
                        model += pulp.lpSum([captain[p] for p in leverage_captains]) == 1
                
                # Game stack constraints
                if strategy == StrategyType.GAME_STACK and combined['combined_stacks']:
                    # Try to force a recommended stack
                    stack_forced = False
                    for stack in combined['combined_stacks'][:3]:
                        if isinstance(stack, dict) and not stack_forced:
                            p1, p2 = stack.get('player1'), stack.get('player2')
                            if p1 in players and p2 in players:
                                # Soft constraint - at least one of the stack
                                model += flex[p1] + captain[p1] + flex[p2] + captain[p2] >= 1
                                stack_forced = True
                
                # Stars and scrubs specific
                if strategy == StrategyType.STARS_SCRUBS:
                    min_salary_players = [p for p in players if salaries[p] <= 3000]
                    max_salary_players = [p for p in players if salaries[p] >= 9000]
                    
                    if min_salary_players and len(min_salary_players) >= 2:
                        model += pulp.lpSum([flex[p] for p in min_salary_players]) >= 2
                    if max_salary_players and len(max_salary_players) >= 2:
                        model += pulp.lpSum([flex[p] + captain[p] for p in max_salary_players]) >= 2
                
                # Diversity constraint - ensure lineup uniqueness
                if lineup_num > 1 and all_lineups:
                    for prev_lineup in all_lineups[-min(3, len(all_lineups)):]:  # Check last 3
                        prev_players = [prev_lineup['Captain']] + prev_lineup['FLEX']
                        model += pulp.lpSum([flex[p] + captain[p] for p in prev_players]) <= 5
                
                # Solve
                model.solve(pulp.PULP_CBC_CMD(msg=0))  # Suppress solver output
                
                if pulp.LpStatus[model.status] == 'Optimal':
                    captain_pick = None
                    flex_picks = []
                    
                    for p in players:
                        if captain[p].value() == 1:
                            captain_pick = p
                            used_captains.add(p)
                        if flex[p].value() == 1:
                            flex_picks.append(p)
                    
                    if captain_pick and len(flex_picks) == 5:
                        # Validate the lineup
                        is_valid, reason = self.validate_lineup_constraints(
                            captain_pick, flex_picks, salaries, ownership
                        )
                        
                        if is_valid:
                            total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
                            total_proj = sum(points[p] for p in flex_picks) + 1.5 * points[captain_pick]
                            total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                            
                            lineup_players = [captain_pick] + flex_picks
                            
                            # Check for stacks
                            has_stack = False
                            stack_details = []
                            for stack in combined['combined_stacks']:
                                if isinstance(stack, dict):
                                    p1, p2 = stack.get('player1'), stack.get('player2')
                                    if p1 in lineup_players and p2 in lineup_players:
                                        has_stack = True
                                        stack_details.append(f"{p1}-{p2}")
                            
                            # Determine ownership tier
                            if total_ownership < 60:
                                ownership_tier = '💎 Elite'
                            elif total_ownership < 80:
                                ownership_tier = '🟢 Optimal'
                            elif total_ownership < 100:
                                ownership_tier = '🟡 Balanced'
                            else:
                                ownership_tier = '⚠️ Chalky'
                            
                            all_lineups.append({
                                'Lineup': len(all_lineups) + 1,  # Use actual count
                                'Strategy': strategy.value,
                                'Captain': captain_pick,
                                'Captain_Own%': ownership.get(captain_pick, 5),
                                'FLEX': flex_picks,
                                'Projected': round(total_proj, 2),
                                'Salary': int(total_salary),
                                'Salary_Remaining': int(OptimizerConfig.SALARY_CAP - total_salary),
                                'Total_Ownership': round(total_ownership, 1),
                                'Ownership_Tier': ownership_tier,
                                'GPP_Summary': self.bucket_manager.get_gpp_summary(lineup_players, self.df, self.field_size),
                                'Leverage_Score': self.bucket_manager.calculate_gpp_leverage(lineup_players, self.df),
                                'Has_Stack': has_stack,
                                'Stack_Details': ', '.join(stack_details) if stack_details else 'None',
                                'Field_Size': self.field_size,
                                'Unique_Captain': captain_pick not in [l['Captain'] for l in all_lineups[:lineup_num-1]] if all_lineups else True,
                                'AI1_Captain': captain_pick in rec1.captain_targets,
                                'AI2_Captain': captain_pick in rec2.captain_targets
                            })
                        else:
                            self.optimization_logger.append(f"Invalid lineup: {reason}")
                            strategy_attempts += 1
                else:
                    failed_attempts += 1
                    strategy_attempts += 1
                    self.optimization_logger.append(f"No solution for {strategy.value} lineup {i+1}")
                    
                    if failed_attempts > max_failures:
                        st.warning(f"Optimization struggling. Generated {len(all_lineups)}/{num_lineups} lineups")
                        break
        
        # If we didn't get enough lineups, report issues
        if len(all_lineups) < num_lineups:
            st.warning(f"Only generated {len(all_lineups)}/{num_lineups} valid lineups")
            if self.optimization_logger:
                with st.expander("Optimization Issues"):
                    for log in self.optimization_logger[-10:]:  # Show last 10 issues
                        st.write(f"- {log}")
        
        return pd.DataFrame(all_lineups)

# NFL GPP DUAL-AI OPTIMIZER - PART 5: MAIN UI AND HELPER FUNCTIONS (COMPLETE & CORRECTED)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate CSV with GPP-specific checks"""
    df = pd.read_csv(uploaded_file)
    
    # Check required columns
    required_cols = ['first_name', 'last_name', 'position', 'team', 'salary', 'ppg_projection']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
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
    
    # Filter valid data
    df = df[df['Salary'] >= OptimizerConfig.MIN_SALARY]
    df = df[df['Projected_Points'] > 0]
    df = df.dropna(subset=['Salary', 'Projected_Points'])
    
    # Add GPP value column (projection * leverage potential)
    df['Value'] = df['Projected_Points'] / (df['Salary'] / 1000)
    
    # Data summary
    st.success(f"✅ Loaded {len(df)} valid players for GPP optimization")
    
    with st.expander("📊 GPP Player Pool Analysis"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Players", len(df))
            st.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
            st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
        
        with col2:
            st.metric("Teams", len(df['Team'].unique()))
            st.metric("Min Salary", f"${df['Salary'].min():,.0f}")
            st.metric("Max Salary", f"${df['Salary'].max():,.0f}")
        
        with col3:
            positions = df['Position'].value_counts()
            st.write("**Position Distribution:**")
            for pos, count in positions.items():
                st.write(f"{pos}: {count}")
    
    return df

def display_gpp_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame, field_size: str):
    """Display GPP-specific lineup analysis and visualizations"""
    
    # Create GPP analysis visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Strategy Distribution for GPP
    ax1 = axes[0, 0]
    strategy_counts = lineups_df['Strategy'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#98D8C8']
    ax1.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.0f%%',
           colors=colors[:len(strategy_counts)], startangle=90)
    ax1.set_title('GPP Strategy Distribution')
    
    # 2. Ownership vs Ceiling (GPP Focus)
    ax2 = axes[0, 1]
    ceiling_col = 'Ceiling_99th' if 'Ceiling_99th' in lineups_df else 'Projected'
    scatter = ax2.scatter(lineups_df['Total_Ownership'], lineups_df[ceiling_col],
                        c=lineups_df.get('GPP_Score', lineups_df['Projected']), 
                        cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add optimal ownership zone
    min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
    ax2.axvspan(min_own, max_own, alpha=0.2, color='green', label=f'Optimal ({min_own}-{max_own}%)')
    
    ax2.set_xlabel('Total Ownership %')
    ax2.set_ylabel('99th Percentile Points')
    ax2.set_title(f'GPP Ownership vs Ceiling ({field_size})')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='GPP Score')
    
    # Annotate top 3 GPP lineups
    top_3 = lineups_df.nlargest(3, 'GPP_Score') if 'GPP_Score' in lineups_df else lineups_df.head(3)
    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
        ax2.annotate(f'#{i}', (row['Total_Ownership'], row.get(ceiling_col, row['Projected'])),
                   fontsize=12, fontweight='bold', color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # 3. Captain Ownership Distribution
    ax3 = axes[0, 2]
    if 'Captain_Own%' in lineups_df:
        captain_owns = lineups_df['Captain_Own%'].values
    else:
        captain_owns = np.random.normal(15, 8, len(lineups_df))  # Fallback data
    ax3.hist(captain_owns, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
    ax3.axvline(15, color='red', linestyle='--', label='15% Threshold')
    ax3.set_xlabel('Captain Ownership %')
    ax3.set_ylabel('Number of Lineups')
    ax3.set_title('GPP Captain Ownership Distribution')
    ax3.legend()
    
    # 4. Leverage Score Distribution
    ax4 = axes[1, 0]
    ax4.hist(lineups_df['Leverage_Score'], bins=15, alpha=0.7, color='#45B7D1', edgecolor='black')
    ax4.axvline(lineups_df['Leverage_Score'].mean(), color='red', linestyle='--', 
               label=f"Mean: {lineups_df['Leverage_Score'].mean():.1f}")
    ax4.set_xlabel('GPP Leverage Score')
    ax4.set_ylabel('Number of Lineups')
    ax4.set_title('Leverage Distribution')
    ax4.legend()
    
    # 5. Ship Equity vs Tournament EV
    ax5 = axes[1, 1]
    if 'Ship_Equity' in lineups_df and 'Tournament_EV' in lineups_df:
        ax5.scatter(lineups_df['Ship_Equity'], lineups_df['Tournament_EV'],
                   c=lineups_df['Total_Ownership'], cmap='RdYlGn_r',
                   s=80, alpha=0.7)
        ax5.set_xlabel('Ship Equity (Win Probability)')
        ax5.set_ylabel('Tournament EV')
        ax5.set_title('GPP Tournament Equity Analysis')
    else:
        # Fallback plot
        ax5.scatter(lineups_df['Total_Ownership'], lineups_df.get('GPP_Score', lineups_df['Projected']),
                   alpha=0.6)
        ax5.set_xlabel('Total Ownership %')
        ax5.set_ylabel('GPP Score')
        ax5.set_title('GPP Score Distribution')
    
    # 6. Top Captains for GPP
    ax6 = axes[1, 2]
    captain_counts = lineups_df.groupby('Captain').agg({
        'Lineup': 'count',
        'Captain_Own%': 'first' if 'Captain_Own%' in lineups_df else lambda x: 15
    }).sort_values('Lineup', ascending=False).head(10)
    
    if 'Captain_Own%' in captain_counts.columns:
        colors = ['#FF6B6B' if own > 20 else '#4ECDC4' if own > 10 else '#45B7D1' 
                 for own in captain_counts['Captain_Own%']]
    else:
        colors = ['#4ECDC4'] * len(captain_counts)
    
    y_pos = np.arange(len(captain_counts))
    bars = ax6.barh(y_pos, captain_counts['Lineup'], color=colors)
    ax6.set_yticks(y_pos)
    if 'Captain_Own%' in captain_counts.columns:
        ax6.set_yticklabels([f"{name} ({own:.0f}%)" 
                             for name, own in zip(captain_counts.index, captain_counts['Captain_Own%'])],
                            fontsize=8)
    else:
        ax6.set_yticklabels(captain_counts.index, fontsize=8)
    ax6.set_xlabel('Times Used as Captain')
    ax6.set_title('Top 10 GPP Captains')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Ownership Tier Distribution
    st.markdown("### 🎯 GPP Ownership Tier Analysis")
    
    tier_data = []
    for idx, row in lineups_df.head(20).iterrows():
        bucket_counts = {'mega_chalk': 0, 'chalk': 0, 'pivot': 0, 'leverage': 0, 'super_leverage': 0}
        lineup_players = [row['Captain']] + row['FLEX']
        
        for player in lineup_players:
            player_own = df[df['Player'] == player]['Ownership'].values
            if len(player_own) > 0:
                bucket = OwnershipBucketManager.get_bucket(player_own[0])
                bucket_counts[bucket] += 1
        
        tier_data.append(list(bucket_counts.values()))
    
    if tier_data:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use GPP-optimized colormap (green for leverage, red for chalk)
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
        
        # Add text annotations with color coding
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

with st.sidebar:
    st.header("🏆 GPP Tournament Settings")
    
    st.markdown("### 🎯 Contest Type")
    contest_type = st.selectbox(
        "Select GPP Type",
        list(OptimizerConfig.FIELD_SIZES.keys()),
        index=2,  # Default to 150-Max
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
    # Load and validate data
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
        
        # Parse ownership with GPP validation
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
        
        # Apply ownership with GPP default
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
    df['GPP_Value'] = (df['Projected_Points'] / (df['Salary'] / 1000)) * (20 / (df['Ownership'] + 5))
    df['Leverage_Score'] = df.apply(
        lambda x: 3 if x['Ownership'] < 5 else 2 if x['Ownership'] < 10 else 1 if x['Ownership'] < 15 else -1 if x['Ownership'] > 30 else 0, 
        axis=1
    )
    
    # Display GPP player pool
    st.markdown("### 💎 GPP Player Pool Analysis")
    
    # Bucket distribution visualization
    bucket_counts = df['Bucket'].value_counts()
    
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
            count = bucket_counts.get(bucket, 0)
            emoji = tier_emojis.get(bucket, '')
            percentage = (count / len(df) * 100) if len(df) > 0 else 0
            st.write(f"{emoji} {bucket}: {count} ({percentage:.0f}%)")
        
        # GPP recommendations
        if bucket_counts.get('super_leverage', 0) < 5:
            st.warning("⚠️ Limited super leverage plays available")
        if bucket_counts.get('mega_chalk', 0) > 10:
            st.info("📊 Heavy chalk slate - focus on leverage")
    
    with col2:
        # Show player pool with GPP metrics
        display_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 
                       'Ownership', 'Bucket', 'GPP_Value', 'Leverage_Score']
        
        # Color code by ownership tier
        def highlight_gpp_rows(row):
            if row['Bucket'] == 'super_leverage':
                return ['background-color: #90EE90'] * len(row)  # Light green
            elif row['Bucket'] == 'leverage':
                return ['background-color: #98FB98'] * len(row)  # Pale green
            elif row['Bucket'] == 'mega_chalk':
                return ['background-color: #FFB6C1'] * len(row)  # Light red
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
            
            # Initialize GPP optimizer
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
                
                # Calculate correlations
                correlations = optimizer.correlation_engine.calculate_gpp_correlations(df, game_info)
                
                # Run GPP simulations
                with st.spinner("Running GPP tournament simulations..."):
                    for idx, row in lineups_df.iterrows():
                        sim_results = optimizer.tournament_sim.simulate_gpp_tournament(
                            row.to_dict(), df, correlations, n_sims=num_sims, field_size=field_size
                        )
                        for key, value in sim_results.items():
                            lineups_df.loc[idx, key] = value
                
                # Calculate GPP scores
                lineups_df = calculate_gpp_scores(lineups_df, field_size)
                
                # Sort by GPP score
                lineups_df = lineups_df.sort_values('GPP_Score', ascending=False)
                
                # Store in session state
                st.session_state['lineups_df'] = lineups_df
                st.session_state['df'] = df
                st.session_state['correlations'] = correlations
                st.session_state['field_size'] = field_size
                
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
            st.metric("Avg 99th%", f"{lineups_df.get('Ceiling_99th', lineups_df['Projected']).mean():.1f}")
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
                    filtered_df = lineups_df  # Fall back to all lineups if filter is too strict
            
            if display_format == "GPP Summary":
                display_cols = ['Lineup', 'Strategy', 'Captain', 'Captain_Own%', 'Projected', 
                              'Ceiling_99th', 'Total_Ownership', 'Leverage_Score', 
                              'GPP_Score', 'Ship_Equity']
                
                # Remove columns that might not exist
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                
                st.dataframe(
                    filtered_df[display_cols].head(show_top_n),
                    use_container_width=True,
                    column_config={
                        "Captain_Own%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Projected": st.column_config.NumberColumn(format="%.1f"),
                        "Ceiling_99th": st.column_config.NumberColumn(format="%.1f"),
                        "Total_Ownership": st.column_config.NumberColumn(format="%.1f%%"),
                        "GPP_Score": st.column_config.NumberColumn(format="%.1f"),
                        "Ship_Equity": st.column_config.NumberColumn(format="%.2f")
                    }
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
                            for player in lineup['FLEX']:
                                pos = df[df['Player'] == player]['Position'].values[0] if player in df['Player'].values else '??'
                                own = df[df['Player'] == player]['Ownership'].values[0] if player in df['Player'].values else 5
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
                    flex_preview = ', '.join(lineup['FLEX'][:3]) + ('...' if len(lineup['FLEX']) > 3 else '')
                    st.write(f"{emoji} **#{i}:** CPT: {lineup['Captain']} ({captain_own:.0f}%) | FLEX: {flex_preview} | Own: {lineup['Total_Ownership']:.0f}% | GPP: {lineup.get('GPP_Score', 0):.0f}")
        
        with tab2:
            st.markdown("### 🔄 GPP Captain Pivots")
            
            if 'pivots_df' in st.session_state and st.session_state['pivots_df']:
                pivots_df = st.session_state['pivots_df']
                
                st.info(f"Generated {len(pivots_df)} GPP captain pivot variations")
                
                # Display pivots
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
                if 'Captain_Own%' in lineups_df:
                    low_captains = lineups_df[lineups_df['Captain_Own%'] < 15]['Captain'].value_counts()
                else:
                    low_captains = pd.Series()
                    
                for player, count in low_captains.items():
                    own = df[df['Player'] == player]['Ownership'].values[0] if player in df['Player'].values else 5
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
                    from collections import Counter
                    stack_counts = Counter(leverage_stacks)
                    for stack, count in stack_counts.most_common(10):
                        pct = count / len(lineups_df) * 100
                        st.write(f"🔗 {stack} - {count} lineups ({pct:.0f}%)")
            
            st.markdown("#### 💎 Super Leverage Plays (<5% ownership)")
            
            player_usage = defaultdict(int)
            for idx, row in lineups_df.iterrows():
                for player in [row['Captain']] + row['FLEX']:
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
            
            # Simulation metrics comparison
            sim_cols = ['Lineup', 'Captain', 'Total_Ownership', 'Mean', 
                       'Ceiling_95th', 'Ceiling_99th', 'Ceiling_99_9th',
                       'Ship_Rate', 'Elite_Rate', 'Boom_Rate']
            
            sim_cols = [col for col in sim_cols if col in lineups_df.columns]
            
            st.dataframe(
                lineups_df[sim_cols].head(20),
                use_container_width=True,
                column_config={
                    "Total_Ownership": st.column_config.NumberColumn(format="%.1f%%"),
                    "Ship_Rate": st.column_config.NumberColumn(format="%.3f%%"),
                    "Elite_Rate": st.column_config.NumberColumn(format="%.2f%%"),
                    "Boom_Rate": st.column_config.NumberColumn(format="%.1f%%")
                }
            )
            
            # Distribution chart
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
                
                # Create DraftKings export
                dk_lineups = []
                export_df = lineups_df.head(export_top_n)
                
                for idx, row in export_df.iterrows():
                    flex_players = row['FLEX']
                    dk_lineups.append({
                        'CPT': row['Captain'],
                        'FLEX 1': flex_players[0] if len(flex_players) > 0 else '',
                        'FLEX 2': flex_players[1] if len(flex_players) > 1 else '',
                        'FLEX 3': flex_players[2] if len(flex_players) > 2 else '',
                        'FLEX 4': flex_players[3] if len(flex_players) > 3 else '',
                        'FLEX 5': flex_players[4] if len(flex_players) > 4 else ''
                    })
                
                dk_df = pd.DataFrame(dk_lineups)
                
                # Preview
                st.write(f"Preview (first 5 of {len(dk_df)}):")
                st.dataframe(dk_df.head(), use_container_width=True)
                
                # Download button
                csv = dk_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download DK CSV ({len(dk_df)} lineups)",
                    data=csv,
                    file_name=f"dk_gpp_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("#### GPP Analysis Export")
                
                # Prepare GPP export
                export_analysis = export_df.copy()
                export_analysis['FLEX'] = export_analysis['FLEX'].apply(lambda x: ', '.join(x))
                
                # Select GPP-specific columns
                gpp_export_cols = ['Lineup', 'Strategy', 'Captain', 'Captain_Own%', 'FLEX', 
                                  'Projected', 'Salary', 'Total_Ownership', 'Leverage_Score',
                                  'Ceiling_95th', 'Ceiling_99th', 'Ceiling_99_9th',
                                  'Ship_Rate', 'Elite_Rate', 'Boom_Rate',
                                  'GPP_Score', 'Tournament_EV', 'Has_Stack']
                
                # Filter to existing columns
                gpp_export_cols = [col for col in gpp_export_cols if col in export_analysis.columns]
                
                final_export = export_analysis[gpp_export_cols]
                
                # Preview
                st.write(f"Preview (first 5 of {len(final_export)}):")
                st.dataframe(final_export.head(), use_container_width=True)
                
                # Download button
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

# Footer with GPP-specific tips
st.markdown("---")
st.markdown("""
### 🏆 NFL GPP Tournament Optimizer - Professional Edition

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

**Version:** 5.0 GPP Edition | **Focus:** Tournament Winning Upside

*Maximize your tournament equity!* 🚀
""")

# Display current field size if available
if 'field_size' in st.session_state:
    current_field = st.session_state['field_size']
    st.caption(f"GPP Optimizer last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Field: {current_field}")
else:
    st.caption(f"GPP Optimizer last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Field: Not Selected")
