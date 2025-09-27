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

st.set_page_config(page_title="NFL Dual-AI Optimizer Pro", page_icon="🏈", layout="wide")
st.title('🏈 NFL Showdown Optimizer - Professional Edition')

# Enhanced Configuration
class OptimizerConfig:
    SALARY_CAP = 50000
    ROSTER_SIZE = 6  # 1 Captain + 5 FLEX
    MAX_PLAYERS_PER_TEAM = 4
    CAPTAIN_MULTIPLIER = 1.5
    DEFAULT_OWNERSHIP = 5
    MIN_SALARY = 1000
    
    # Simulation parameters
    BASE_VOLATILITY = 0.25
    HIGH_VOLATILITY = 0.35
    NUM_SIMS = 5000
    FIELD_SIZE = 100000
    
    # API Configuration
    CLAUDE_MODEL = "claude-3-haiku-20240307"  # Most reliable
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # Ownership buckets
    OWNERSHIP_BUCKETS = {
        'mega_chalk': (40, 100),      # 40%+ ownership
        'chalk': (25, 40),             # 25-40% ownership
        'pivot': (10, 25),             # 10-25% ownership
        'leverage': (5, 10),           # 5-10% ownership
        'super_leverage': (0, 5)       # <5% ownership
    }
    
    # Flexible bucket rules based on strategy
    BUCKET_RULES = {
        'balanced': {
            'mega_chalk': (0, 3),      # Allow up to 3 mega chalk
            'chalk': (0, 4),           # Flexible chalk
            'pivot': (0, 6),           # Any pivots
            'leverage': (0, 6),        # Any leverage
            'super_leverage': (0, 6)   # Any super leverage
        },
        'contrarian': {
            'mega_chalk': (0, 1),      # Max 1 mega chalk
            'chalk': (0, 2),           # Max 2 chalk
            'pivot': (1, 6),           # At least 1 pivot
            'leverage': (1, 6),        # At least 1 leverage
            'super_leverage': (0, 6)   # Any super leverage
        },
        'leverage': {
            'mega_chalk': (0, 0),      # No mega chalk
            'chalk': (0, 1),           # Max 1 chalk
            'pivot': (1, 6),           # At least 1 pivot
            'leverage': (2, 6),        # At least 2 leverage
            'super_leverage': (1, 6)   # At least 1 super leverage
        }
    }
    
    # Correlation defaults
    QB_PASS_CATCHER_CORR = 0.45
    QB_RB_CORR = -0.15
    SAME_TEAM_WR_CORR = 0.15
    OPPOSING_QB_CORR = 0.25
    DST_OPPOSING_CORR = -0.35

class StrategyType(Enum):
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"
    GAME_STACK = "game_stack"
    STARS_SCRUBS = "stars_scrubs"

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

@dataclass
class PlayerProjection:
    """Enhanced player projection with confidence intervals"""
    player: str
    projection: float
    floor: float
    ceiling: float
    volatility: float

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
                system = """You are an expert DFS analyst. Provide strategic recommendations 
                           in valid JSON format only, no markdown or extra text."""
            
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

class OwnershipBucketManager:
    """Advanced ownership bucketing and analysis"""
    
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
    def validate_lineup_buckets(lineup_players: List[str], df: pd.DataFrame, 
                               strategy: str = 'balanced') -> Tuple[bool, str]:
        """Check if lineup meets bucket constraints"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        bucket_counts = defaultdict(int)
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            bucket = OwnershipBucketManager.get_bucket(ownership)
            bucket_counts[bucket] += 1
        
        rules = OptimizerConfig.BUCKET_RULES.get(strategy, OptimizerConfig.BUCKET_RULES['balanced'])
        
        violations = []
        for bucket_name, (min_count, max_count) in rules.items():
            count = bucket_counts.get(bucket_name, 0)
            if count < min_count:
                violations.append(f"Need at least {min_count} {bucket_name}")
            if count > max_count:
                violations.append(f"Too many {bucket_name} ({count}/{max_count})")
        
        if violations:
            return False, "; ".join(violations)
        return True, "Valid"
    
    @staticmethod
    def get_bucket_summary(lineup_players: List[str], df: pd.DataFrame) -> str:
        """Get a detailed summary of lineup's ownership profile"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        bucket_counts = defaultdict(int)
        total_ownership = 0
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            bucket = OwnershipBucketManager.get_bucket(ownership)
            bucket_counts[bucket] += 1
            total_ownership += ownership
        
        # Create emoji indicators
        bucket_emojis = {
            'mega_chalk': '🔴',
            'chalk': '🟠',
            'pivot': '🟢',
            'leverage': '🔵',
            'super_leverage': '🟣'
        }
        
        summary = f"Total: {total_ownership:.1f}% | "
        summary += " ".join([f"{bucket_emojis.get(k, '')} {k}:{v}" 
                           for k, v in bucket_counts.items() if v > 0])
        return summary
    
    @staticmethod
    def calculate_lineup_leverage(lineup_players: List[str], df: pd.DataFrame) -> float:
        """Calculate overall leverage score for a lineup"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        leverage_score = 0
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            if ownership < 5:
                leverage_score += 3
            elif ownership < 10:
                leverage_score += 2
            elif ownership < 25:
                leverage_score += 1
            elif ownership > 40:
                leverage_score -= 1
        
        return leverage_score

class CaptainPivotGenerator:
    """Advanced captain pivot generation with salary optimization"""
    
    @staticmethod
    def generate_pivots(lineup: Dict, df: pd.DataFrame, max_pivots: int = 3) -> List[Dict]:
        """Generate captain pivot variations with enhanced metrics"""
        captain = lineup['Captain']
        flex_players = lineup['FLEX']
        
        salaries = df.set_index('Player')['Salary'].to_dict()
        points = df.set_index('Player')['Projected_Points'].to_dict()
        ownership = df.set_index('Player')['Ownership'].to_dict()
        positions = df.set_index('Player')['Position'].to_dict()
        
        pivot_lineups = []
        
        # Sort FLEX players by leverage potential
        flex_leverage = [(p, ownership.get(p, 5)) for p in flex_players]
        flex_leverage.sort(key=lambda x: x[1])  # Lower ownership first
        
        for new_captain, captain_own in flex_leverage[:max_pivots]:
            old_captain_salary = salaries.get(captain, 0)
            new_captain_salary = salaries.get(new_captain, 0)
            
            salary_freed = old_captain_salary * 0.5
            salary_needed = new_captain_salary * 0.5
            
            if salary_freed >= salary_needed - 100:  # Allow small salary overrun
                new_flex = [p for p in flex_players if p != new_captain] + [captain]
                
                pivot_lineup = lineup.copy()
                pivot_lineup['Captain'] = new_captain
                pivot_lineup['FLEX'] = new_flex
                pivot_lineup['Pivot_Type'] = 'Leverage Swap' if captain_own < 10 else 'Standard Swap'
                pivot_lineup['Original_Captain'] = captain
                
                # Calculate metrics
                total_proj = points.get(new_captain, 0) * 1.5 + sum(points.get(p, 0) for p in new_flex)
                total_own = ownership.get(new_captain, 5) * 1.5 + sum(ownership.get(p, 5) for p in new_flex)
                
                pivot_lineup['Projected'] = round(total_proj, 2)
                pivot_lineup['Total_Ownership'] = round(total_own, 1)
                pivot_lineup['Ownership_Delta'] = round(total_own - lineup.get('Total_Ownership', 100), 1)
                pivot_lineup['Leverage_Gain'] = round((lineup.get('Total_Ownership', 100) - total_own) / 10, 1)
                pivot_lineup['Captain_Position'] = positions.get(new_captain, 'Unknown')
                
                pivot_lineups.append(pivot_lineup)
        
        return pivot_lineups
    
    @staticmethod
    def find_optimal_pivots(lineup: Dict, df: pd.DataFrame, 
                           target_ownership: float = 80) -> List[Dict]:
        """Find pivots that hit a target ownership level"""
        pivots = CaptainPivotGenerator.generate_pivots(lineup, df, max_pivots=5)
        
        # Sort by distance to target ownership
        for pivot in pivots:
            pivot['Target_Distance'] = abs(pivot['Total_Ownership'] - target_ownership)
        
        pivots.sort(key=lambda x: x['Target_Distance'])
        return pivots[:3]

class CorrelationEngine:
    """Advanced correlation calculations for stacking"""
    
    @staticmethod
    def calculate_dynamic_correlations(df: pd.DataFrame, game_context: Dict) -> Dict[Tuple[str, str], float]:
        """Calculate correlations based on positions, teams, and game context"""
        correlations = {}
        
        players = df['Player'].tolist()
        positions = df.set_index('Player')['Position'].to_dict()
        teams = df.set_index('Player')['Team'].to_dict()
        
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                team1, team2 = teams.get(p1), teams.get(p2)
                pos1, pos2 = positions.get(p1), positions.get(p2)
                
                correlation = 0
                
                if team1 == team2:
                    # Same team correlations
                    if pos1 == 'QB' and pos2 in ['WR', 'TE']:
                        correlation = OptimizerConfig.QB_PASS_CATCHER_CORR
                        # Adjust for game total
                        if game_context.get('total', 48) > 52:
                            correlation += 0.1  # Higher correlation in shootouts
                    elif pos1 == 'QB' and pos2 == 'RB':
                        correlation = OptimizerConfig.QB_RB_CORR
                        # More negative in likely blowouts
                        if abs(game_context.get('spread', 0)) > 7:
                            correlation -= 0.1
                    elif pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                        correlation = OptimizerConfig.SAME_TEAM_WR_CORR
                    elif pos1 == 'RB' and pos2 == 'RB':
                        correlation = -0.5  # RBs rarely both succeed
                
                else:
                    # Opposing team correlations
                    if pos1 == 'QB' and pos2 == 'QB':
                        correlation = OptimizerConfig.OPPOSING_QB_CORR
                        # Higher in projected shootouts
                        if game_context.get('total', 48) > 52:
                            correlation += 0.15
                    elif pos1 == 'QB' and pos2 in ['WR', 'TE']:
                        # Opposing QB-WR has slight positive correlation (shootouts)
                        correlation = 0.1
                    elif 'DST' in [pos1, pos2]:
                        correlation = OptimizerConfig.DST_OPPOSING_CORR
                
                if correlation != 0:
                    correlations[(p1, p2)] = correlation
        
        return correlations
    
    @staticmethod
    def identify_optimal_stacks(df: pd.DataFrame, correlations: Dict) -> List[Dict]:
        """Identify the best stacking opportunities"""
        stacks = []
        
        # Find all positive correlations
        for (p1, p2), corr in correlations.items():
            if corr > 0.2:  # Meaningful positive correlation
                ownership1 = df[df['Player'] == p1]['Ownership'].values[0] if len(df[df['Player'] == p1]) > 0 else 5
                ownership2 = df[df['Player'] == p2]['Ownership'].values[0] if len(df[df['Player'] == p2]) > 0 else 5
                
                stacks.append({
                    'player1': p1,
                    'player2': p2,
                    'correlation': corr,
                    'combined_ownership': ownership1 + ownership2,
                    'leverage': max(0, 50 - (ownership1 + ownership2))  # Lower ownership = more leverage
                })
        
        # Sort by correlation * leverage
        stacks.sort(key=lambda x: x['correlation'] * (1 + x['leverage']/100), reverse=True)
        return stacks[:10]

class TournamentSimulator:
    """Advanced tournament simulation with realistic distributions"""
    
    @staticmethod
    def simulate_with_correlation(lineup: Dict, df: pd.DataFrame, 
                                 correlations: Dict, n_sims: int = 5000) -> Dict[str, float]:
        """Run correlated tournament simulations"""
        captain = lineup['Captain']
        flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else lineup['FLEX']
        all_players = [captain] + list(flex_players)
        
        projections = df.set_index('Player')['Projected_Points'].to_dict()
        
        # Build correlation matrix
        n_players = len(all_players)
        means = np.array([projections.get(p, 0) for p in all_players])
        
        # Create covariance matrix
        cov_matrix = np.eye(n_players) * (OptimizerConfig.BASE_VOLATILITY ** 2)
        
        for i in range(n_players):
            for j in range(i+1, n_players):
                p1, p2 = all_players[i], all_players[j]
                pair = tuple(sorted([p1, p2]))
                if pair in correlations:
                    correlation = correlations[pair]
                    cov_matrix[i, j] = correlation * OptimizerConfig.BASE_VOLATILITY ** 2
                    cov_matrix[j, i] = cov_matrix[i, j]
        
        try:
            # Use gamma distribution for more realistic DFS scoring
            sims = multivariate_normal(mean=means, cov=cov_matrix, allow_singular=True).rvs(n_sims)
            
            # Apply realistic constraints
            sims = np.maximum(0, sims)  # No negative scores
            
            # Add injury/boom-bust volatility
            for i in range(n_players):
                # 5% chance of injury (score < 25% of projection)
                injury_mask = np.random.random(n_sims) < 0.05
                sims[injury_mask, i] *= np.random.uniform(0, 0.25, injury_mask.sum())
                
                # 5% chance of boom (score > 200% of projection)
                boom_mask = np.random.random(n_sims) < 0.05
                sims[boom_mask, i] *= np.random.uniform(2.0, 3.0, boom_mask.sum())
            
            # Calculate lineup scores
            captain_scores = sims[:, 0] * OptimizerConfig.CAPTAIN_MULTIPLIER
            flex_scores = np.sum(sims[:, 1:], axis=1)
            total_scores = captain_scores + flex_scores
            
            return {
                'Mean': round(np.mean(total_scores), 2),
                'Std': round(np.std(total_scores), 2),
                'Floor_10th': round(np.percentile(total_scores, 10), 2),
                'Median': round(np.percentile(total_scores, 50), 2),
                'Ceiling_75th': round(np.percentile(total_scores, 75), 2),
                'Ceiling_90th': round(np.percentile(total_scores, 90), 2),
                'Ceiling_95th': round(np.percentile(total_scores, 95), 2),
                'Ceiling_99th': round(np.percentile(total_scores, 99), 2),
                'Boom_Rate': round(np.mean(total_scores > np.percentile(total_scores, 95)) * 100, 1)
            }
            
        except Exception as e:
            st.warning(f"Simulation error: {e}")
            # Fallback to simple calculation
            total_proj = projections.get(captain, 0) * 1.5 + sum(projections.get(p, 0) for p in flex_players)
            return {
                'Mean': round(total_proj, 2),
                'Std': round(total_proj * 0.25, 2),
                'Floor_10th': round(total_proj * 0.7, 2),
                'Median': round(total_proj, 2),
                'Ceiling_75th': round(total_proj * 1.15, 2),
                'Ceiling_90th': round(total_proj * 1.3, 2),
                'Ceiling_95th': round(total_proj * 1.4, 2),
                'Ceiling_99th': round(total_proj * 1.6, 2),
                'Boom_Rate': 5.0
            }
    
    @staticmethod
    def calculate_win_probability(lineup_score_distribution: np.ndarray, 
                                 field_size: int = 100000) -> Dict[str, float]:
        """Calculate tournament win probabilities"""
        # Model field scores with gamma distribution (right-skewed)
        field_mean = 90
        field_shape = 4
        field_scale = field_mean / field_shape
        
        # Simulate field scores
        field_scores = gamma.rvs(field_shape, scale=field_scale, size=field_size)
        
        win_prob = 0
        top_10_prob = 0
        top_100_prob = 0
        cash_prob = 0  # Top 20%
        
        for score in lineup_score_distribution[:1000]:  # Sample for speed
            placement = np.sum(score > field_scores)
            percentile = placement / field_size
            
            if placement == field_size:
                win_prob += 1
            if placement >= field_size - 10:
                top_10_prob += 1
            if placement >= field_size - 100:
                top_100_prob += 1
            if percentile >= 0.8:
                cash_prob += 1
        
        n_samples = min(1000, len(lineup_score_distribution))
        
        return {
            'Win_Prob': round(win_prob / n_samples * 100, 3),
            'Top_10_Prob': round(top_10_prob / n_samples * 100, 2),
            'Top_100_Prob': round(top_100_prob / n_samples * 100, 2),
            'Cash_Prob': round(cash_prob / n_samples * 100, 1)
        }

class GameTheoryStrategist:
    """AI Strategist 1: Advanced game theory and ownership leverage"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict) -> str:
        """Generate comprehensive game theory prompt"""
        
        bucket_manager = OwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)
        
        # Get key players by category
        mega_chalk = df[df['Player'].isin(buckets.get('mega_chalk', []))].nlargest(5, 'Ownership')[
            ['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        leverage_plays = df[df['Player'].isin(buckets.get('leverage', []) + buckets.get('super_leverage', []))].nlargest(
            10, 'Projected_Points')[['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        # Calculate value plays
        df['Value'] = df['Projected_Points'] / (df['Salary'] / 1000)
        top_values = df.nlargest(5, 'Value')[['Player', 'Position', 'Value', 'Ownership']]
        
        return f"""
        As an expert DFS Game Theory strategist, analyze this NFL Showdown slate:
        
        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        
        MEGA CHALK PLAYERS (40%+ ownership):
        {mega_chalk.to_string() if not mega_chalk.empty else 'None'}
        
        LEVERAGE OPPORTUNITIES (<10% ownership with upside):
        {leverage_plays.to_string() if not leverage_plays.empty else 'None'}
        
        TOP VALUE PLAYS:
        {top_values.to_string()}
        
        OWNERSHIP DISTRIBUTION:
        - Mega Chalk (40%+): {len(buckets.get('mega_chalk', []))} players
        - Chalk (25-40%): {len(buckets.get('chalk', []))} players
        - Pivot (10-25%): {len(buckets.get('pivot', []))} players
        - Leverage (<10%): {len(buckets.get('leverage', []) + buckets.get('super_leverage', []))} players
        
        Provide strategic recommendations considering:
        1. Which chalk plays are actually -EV due to ownership?
        2. Which low-owned players have hidden ceiling?
        3. Optimal cumulative ownership targets?
        4. Game theory optimal captain selections?
        5. Contrarian roster constructions that maintain projection?
        
        Return ONLY valid JSON:
        {{
            "leverage_captains": ["player1", "player2", "player3"],
            "must_fades": ["overowned1", "overowned2"],
            "must_plays": ["leverage1", "leverage2"],
            "hidden_gems": ["gem1", "gem2"],
            "construction_rules": {{
                "max_mega_chalk": 1,
                "min_leverage": 2,
                "target_ownership": {{"min": 70, "max": 110}}
            }},
            "contrarian_stacks": [
                {{"player1": "name1", "player2": "name2", "combined_own": 15}}
            ],
            "game_theory_insights": [
                "Key strategic insight 1",
                "Key strategic insight 2"
            ],
            "confidence_score": 0.85
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse and validate game theory response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Extract construction rules
        rules = data.get('construction_rules', {})
        max_chalk = rules.get('max_mega_chalk', 2)
        min_leverage = rules.get('min_leverage', 1)
        
        # Determine strategy weights based on rules
        if max_chalk <= 1 and min_leverage >= 2:
            strategy_weights = {
                StrategyType.LEVERAGE: 0.4,
                StrategyType.CONTRARIAN: 0.3,
                StrategyType.STARS_SCRUBS: 0.2,
                StrategyType.BALANCED: 0.1,
                StrategyType.CORRELATION: 0.0,
                StrategyType.GAME_STACK: 0.0
            }
        else:
            strategy_weights = {
                StrategyType.BALANCED: 0.3,
                StrategyType.LEVERAGE: 0.25,
                StrategyType.CONTRARIAN: 0.2,
                StrategyType.CORRELATION: 0.15,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.GAME_STACK: 0.0
            }
        
        return AIRecommendation(
            strategist_name="Game Theory AI",
            confidence=data.get('confidence_score', 0.75),
            captain_targets=data.get('leverage_captains', []),
            stacks=[{'player1': s.get('player1'), 'player2': s.get('player2')} 
                   for s in data.get('contrarian_stacks', []) if isinstance(s, dict)],
            fades=data.get('must_fades', []),
            boosts=data.get('must_plays', []) + data.get('hidden_gems', []),
            strategy_weights=strategy_weights,
            key_insights=data.get('game_theory_insights', ["Using default game theory strategy"])
        )

class CorrelationStrategist:
    """AI Strategist 2: Advanced correlation and stacking analysis"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict) -> str:
        """Generate comprehensive correlation analysis prompt"""
        
        teams = df['Team'].unique()[:2]  # Focus on main game
        team_breakdown = {}
        
        for team in teams:
            team_df = df[df['Team'] == team]
            
            # Get key players by position
            qbs = team_df[team_df['Position'] == 'QB'][['Player', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            pass_catchers = team_df[team_df['Position'].isin(['WR', 'TE'])][
                ['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            rbs = team_df[team_df['Position'] == 'RB'][['Player', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            
            team_breakdown[team] = {
                'QB': qbs,
                'Pass_Catchers': pass_catchers[:5],  # Top 5
                'RB': rbs[:3]  # Top 3
            }
        
        return f"""
        As an expert DFS Correlation strategist, identify optimal stacking patterns:
        
        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        
        TEAM ROSTERS:
        {json.dumps(team_breakdown, indent=2)}
        
        Analyze for:
        1. Primary stacks (QB + pass catcher combinations)
        2. Game stacks (correlated players from both teams)
        3. Leverage stacks (low combined ownership with correlation)
        4. Negative correlations to avoid
        5. Game script dependent correlations
        6. Secondary stacks (RB+DEF, TE+TE, etc.)
        
        Consider:
        - Target share and red zone role
        - Game script scenarios (blowout vs shootout)
        - Historical correlation data
        - Injury/weather impacts on correlation
        
        Return ONLY valid JSON:
        {{
            "primary_stacks": [
                {{"qb": "name", "receiver": "name", "correlation": 0.6, "stack_ownership": 25, "reasoning": "why"}}
            ],
            "game_stacks": [
                {{"team1_player": "name1", "team2_player": "name2", "scenario": "shootout", "correlation": 0.4}}
            ],
            "leverage_stacks": [
                {{"player1": "name1", "player2": "name2", "combined_own": 15, "upside": "high"}}
            ],
            "avoid_together": [
                {{"player1": "name1", "player2": "name2", "reason": "negative correlation"}}
            ],
            "game_script_correlations": {{
                "blowout_home": {{"boost": ["player1", "player2"], "fade": ["player3"]}},
                "shootout": {{"boost": ["player4", "player5"], "fade": ["player6"]}},
                "low_scoring": {{"boost": ["def1", "rb1"], "fade": ["wr1", "wr2"]}}
            }},
            "secondary_stacks": [
                {{"type": "RB+DEF", "player1": "rb1", "player2": "def1", "scenario": "leading script"}}
            ],
            "correlation_insights": [
                "Key correlation insight 1",
                "Key correlation insight 2"
            ],
            "recommended_game_script": "shootout",
            "confidence": 0.85
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse and validate correlation response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Build comprehensive stacks list
        all_stacks = []
        
        # Primary stacks
        for stack in data.get('primary_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('qb'),
                    'player2': stack.get('receiver'),
                    'type': 'primary',
                    'correlation': stack.get('correlation', 0.5)
                })
        
        # Game stacks
        for stack in data.get('game_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('team1_player'),
                    'player2': stack.get('team2_player'),
                    'type': 'game',
                    'correlation': stack.get('correlation', 0.3)
                })
        
        # Leverage stacks
        for stack in data.get('leverage_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('player1'),
                    'player2': stack.get('player2'),
                    'type': 'leverage',
                    'correlation': 0.3
                })
        
        # Determine strategy weights based on game script
        game_script = data.get('recommended_game_script', 'balanced')
        
        if game_script == 'shootout':
            strategy_weights = {
                StrategyType.GAME_STACK: 0.4,
                StrategyType.CORRELATION: 0.35,
                StrategyType.BALANCED: 0.15,
                StrategyType.LEVERAGE: 0.1,
                StrategyType.CONTRARIAN: 0.0,
                StrategyType.STARS_SCRUBS: 0.0
            }
        elif game_script in ['blowout', 'low_scoring']:
            strategy_weights = {
                StrategyType.CONTRARIAN: 0.3,
                StrategyType.LEVERAGE: 0.3,
                StrategyType.BALANCED: 0.2,
                StrategyType.STARS_SCRUBS: 0.2,
                StrategyType.CORRELATION: 0.0,
                StrategyType.GAME_STACK: 0.0
            }
        else:
            strategy_weights = {
                StrategyType.CORRELATION: 0.3,
                StrategyType.BALANCED: 0.25,
                StrategyType.GAME_STACK: 0.2,
                StrategyType.LEVERAGE: 0.15,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.CONTRARIAN: 0.0
            }
        
        # Extract captain targets from QB positions in stacks
        captain_targets = []
        for stack in data.get('primary_stacks', []):
            if isinstance(stack, dict) and stack.get('qb'):
                captain_targets.append(stack['qb'])
        
        return AIRecommendation(
            strategist_name="Correlation AI",
            confidence=data.get('confidence', 0.75),
            captain_targets=captain_targets,
            stacks=all_stacks,
            fades=[],
            boosts=[],
            strategy_weights=strategy_weights,
            key_insights=data.get('correlation_insights', ["Using default correlation strategy"])
        )

class DualAIOptimizer:
    """Main optimizer with dual AI strategy integration"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict, api_manager: ClaudeAPIManager = None):
        self.df = df
        self.game_info = game_info
        self.api_manager = api_manager
        self.game_theory_ai = GameTheoryStrategist(api_manager)
        self.correlation_ai = CorrelationStrategist(api_manager)
        self.bucket_manager = OwnershipBucketManager()
        self.pivot_generator = CaptainPivotGenerator()
        self.correlation_engine = CorrelationEngine()
        self.tournament_sim = TournamentSimulator()
    
    def get_ai_strategies(self, use_api: bool = True) -> Tuple[AIRecommendation, AIRecommendation]:
        """Get strategies from both AIs"""
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode - automatic
            with st.spinner("🤖 Game Theory AI analyzing..."):
                gt_prompt = self.game_theory_ai.generate_prompt(self.df, self.game_info)
                gt_response = self.api_manager.get_ai_response(gt_prompt)
            
            with st.spinner("🔗 Correlation AI analyzing..."):
                corr_prompt = self.correlation_ai.generate_prompt(self.df, self.game_info)
                corr_response = self.api_manager.get_ai_response(corr_prompt)
        else:
            # Manual mode
            st.subheader("📝 Manual AI Strategy Input")
            
            tab1, tab2 = st.tabs(["🎯 Game Theory AI", "🔗 Correlation AI"])
            
            with tab1:
                with st.expander("View Game Theory Prompt"):
                    st.text_area("Copy this prompt:", 
                               value=self.game_theory_ai.generate_prompt(self.df, self.game_info),
                               height=300, key="gt_prompt_display")
                gt_response = st.text_area("Paste Game Theory Response (JSON):", 
                                          height=200, key="gt_manual_input",
                                          value='{}')
            
            with tab2:
                with st.expander("View Correlation Prompt"):
                    st.text_area("Copy this prompt:", 
                               value=self.correlation_ai.generate_prompt(self.df, self.game_info),
                               height=300, key="corr_prompt_display")
                corr_response = st.text_area("Paste Correlation Response (JSON):", 
                                            height=200, key="corr_manual_input",
                                            value='{}')
        
        rec1 = self.game_theory_ai.parse_response(gt_response, self.df)
        rec2 = self.correlation_ai.parse_response(corr_response, self.df)
        
        return rec1, rec2
    
    def combine_recommendations(self, rec1: AIRecommendation, rec2: AIRecommendation) -> Dict:
        """Combine recommendations with weighted consensus"""
        
        total_confidence = rec1.confidence + rec2.confidence
        w1 = rec1.confidence / total_confidence if total_confidence > 0 else 0.5
        w2 = rec2.confidence / total_confidence if total_confidence > 0 else 0.5
        
        # Combine captain targets with scoring
        all_captains = set(rec1.captain_targets + rec2.captain_targets)
        captain_scores = {}
        for captain in all_captains:
            score = 0
            if captain in rec1.captain_targets:
                score += w1
            if captain in rec2.captain_targets:
                score += w2
            captain_scores[captain] = score
        
        # Combine strategy weights
        combined_weights = {}
        for strategy in StrategyType:
            combined_weights[strategy] = (
                rec1.strategy_weights.get(strategy, 0) * w1 +
                rec2.strategy_weights.get(strategy, 0) * w2
            )
        
        # Normalize weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return {
            'captain_scores': captain_scores,
            'strategy_weights': combined_weights,
            'consensus_fades': list(set(rec1.fades) & set(rec2.fades)) if rec1.fades and rec2.fades else [],
            'all_boosts': list(set(rec1.boosts) | set(rec2.boosts)) if rec1.boosts or rec2.boosts else [],
            'combined_stacks': rec1.stacks + rec2.stacks,
            'confidence': (rec1.confidence + rec2.confidence) / 2,
            'insights': rec1.key_insights + rec2.key_insights
        }
    
    def generate_optimized_lineups(self, num_lineups: int, rec1: AIRecommendation, 
                                  rec2: AIRecommendation, enforce_buckets: bool = False) -> pd.DataFrame:
        """Generate lineups with advanced optimization"""
        
        combined = self.combine_recommendations(rec1, rec2)
        
        # Get data
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Apply AI adjustments
        adjusted_points = points.copy()
        for player in combined['consensus_fades']:
            if player in adjusted_points:
                adjusted_points[player] *= 0.85
        
        for player in combined['all_boosts']:
            if player in adjusted_points:
                adjusted_points[player] *= 1.12
        
        # Get ownership buckets
        player_buckets = {}
        for player in players:
            player_buckets[player] = self.bucket_manager.get_bucket(
                ownership.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            )
        
        # Distribute lineups across strategies
        lineups_per_strategy = {}
        for strategy, weight in combined['strategy_weights'].items():
            lineups_per_strategy[strategy] = max(1, int(num_lineups * weight))
        
        # Adjust to match target
        total_assigned = sum(lineups_per_strategy.values())
        if total_assigned < num_lineups:
            lineups_per_strategy[StrategyType.BALANCED] = lineups_per_strategy.get(StrategyType.BALANCED, 0) + (num_lineups - total_assigned)
        
        all_lineups = []
        lineup_num = 0
        
        for strategy, count in lineups_per_strategy.items():
            for i in range(count):
                lineup_num += 1
                
                model = pulp.LpProblem(f"Lineup_{lineup_num}_{strategy.value}", pulp.LpMaximize)
                flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
                
                # Strategy-specific objective
                if strategy == StrategyType.LEVERAGE:
                    # Emphasize low ownership captains
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 + max(0, 20 - ownership.get(p, 10))/50) +
                        1.5 * adjusted_points[p] * captain[p] * (1 + max(0, 15 - ownership.get(p, 10))/30)
                        for p in players
                    ])
                
                elif strategy == StrategyType.CONTRARIAN:
                    # Fade chalk heavily
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 - ownership.get(p, 5)/150) +
                        1.5 * adjusted_points[p] * captain[p] * (1 - ownership.get(p, 5)/100)
                        for p in players
                    ])
                
                elif strategy == StrategyType.STARS_SCRUBS:
                    # Standard objective but will add salary constraints
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] + 1.5 * adjusted_points[p] * captain[p]
                        for p in players
                    ])
                
                else:
                    # Standard objective
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] + 1.5 * adjusted_points[p] * captain[p]
                        for p in players
                    ])
                
                # Basic constraints
                model += pulp.lpSum(captain.values()) == 1
                model += pulp.lpSum(flex.values()) == 5
                
                for p in players:
                    model += flex[p] + captain[p] <= 1
                
                model += pulp.lpSum([
                    salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                    for p in players
                ]) <= OptimizerConfig.SALARY_CAP
                
                # Team constraint
                for team in self.df['Team'].unique():
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
                
                # Strategy-specific constraints
                if strategy == StrategyType.STARS_SCRUBS:
                    # Force at least 2 min-priced players
                    min_salary_players = [p for p in players if salaries[p] <= 3500]
                    if len(min_salary_players) >= 2:
                        model += pulp.lpSum([flex[p] for p in min_salary_players]) >= 2
                
                elif strategy == StrategyType.GAME_STACK and combined['combined_stacks']:
                    # Try to include a recommended stack
                    for stack in combined['combined_stacks'][:2]:
                        if isinstance(stack, dict):
                            p1, p2 = stack.get('player1'), stack.get('player2')
                            if p1 in players and p2 in players:
                                # Soft constraint for stacking
                                stack_bonus = pulp.LpVariable(f"stack_{p1}_{p2}", cat='Binary')
                                model += stack_bonus <= flex[p1] + captain[p1]
                                model += stack_bonus <= flex[p2] + captain[p2]
                                break
                
                # Optional bucket constraints
                if enforce_buckets and strategy in [StrategyType.LEVERAGE, StrategyType.CONTRARIAN]:
                    strategy_name = 'leverage' if strategy == StrategyType.LEVERAGE else 'contrarian'
                    bucket_rules = OptimizerConfig.BUCKET_RULES.get(strategy_name, {})
                    
                    for bucket_name, (min_count, max_count) in bucket_rules.items():
                        bucket_players = [p for p in players if player_buckets[p] == bucket_name]
                        if bucket_players and min_count > 0:
                            model += pulp.lpSum([flex[p] + captain[p] for p in bucket_players]) >= min_count
                
                # Diversity constraint
                if lineup_num > 1 and all_lineups:
                    for prev_lineup in all_lineups[-min(3, len(all_lineups)):]:
                        prev_players = [prev_lineup['Captain']] + prev_lineup['FLEX']
                        model += pulp.lpSum([flex[p] + captain[p] for p in prev_players]) <= 5
                
                # Solve
                model.solve()
                
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
                        
                        all_lineups.append({
                            'Lineup': lineup_num,
                            'Strategy': strategy.value,
                            'Captain': captain_pick,
                            'FLEX': flex_picks,
                            'Projected': round(total_proj, 2),
                            'Salary': int(total_salary),
                            'Salary_Remaining': int(OptimizerConfig.SALARY_CAP - total_salary),
                            'Total_Ownership': round(total_ownership, 1),
                            'Bucket_Summary': self.bucket_manager.get_bucket_summary(lineup_players, self.df),
                            'Leverage_Score': self.bucket_manager.calculate_lineup_leverage(lineup_players, self.df),
                            'Has_Stack': has_stack,
                            'Stack_Details': ', '.join(stack_details) if stack_details else 'None',
                            'AI1_Captain': captain_pick in rec1.captain_targets,
                            'AI2_Captain': captain_pick in rec2.captain_targets
                        })
        
        return pd.DataFrame(all_lineups)

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate CSV with comprehensive checks"""
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
    
    # Add value column
    df['Value'] = df['Projected_Points'] / (df['Salary'] / 1000)
    
    # Data summary
    st.success(f"✅ Loaded {len(df)} valid players")
    
    with st.expander("📊 Data Quality Report"):
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

# MAIN STREAMLIT UI
with st.sidebar:
    st.header("🤖 AI Strategy Configuration")
    
    st.markdown("### Connection Mode")
    api_mode = st.radio(
        "Select Mode",
        ["Manual (Free)", "API (Automated)"],
        help="API mode requires Claude API key"
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
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        st.markdown("### Optimization Parameters")
        enforce_buckets = st.checkbox("Enforce Ownership Buckets", value=False,
                                     help="Strict ownership distribution rules")
        
        use_correlation = st.checkbox("Use Correlation Model", value=True,
                                     help="Enable player correlation simulations")
        
        st.markdown("### Simulation Settings")
        num_sims = st.slider("Monte Carlo Simulations", 1000, 10000, 5000, 1000,
                           help="More simulations = more accurate but slower")
        
        st.markdown("### Export Options")
        include_pivots = st.checkbox("Generate Captain Pivots", value=True)
        export_detailed = st.checkbox("Detailed Export", value=False,
                                     help="Include all metrics in CSV")

# Main Content Area
st.markdown("## 📁 Data Upload & Configuration")

uploaded_file = st.file_uploader(
    "Upload DraftKings CSV",
    type="csv",
    help="Export player pool from DraftKings"
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
        total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5)
    with col3:
        spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)
    with col4:
        game_time = st.selectbox("Game Time", ["1:00 PM", "4:25 PM", "8:20 PM", "MNF"])
    
    game_info = {
        'teams': teams,
        'total': total,
        'spread': spread,
        'game_time': game_time
    }
    
    # Player Adjustments
    st.markdown("### 📝 Player Adjustments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏥 Injury Report")
        injury_text = st.text_area(
            "Format: Player: Status",
            height=120,
            placeholder="Josh Allen: QUESTIONABLE\nTyreek Hill: OUT",
            help="OUT = 0%, DOUBTFUL = 30%, QUESTIONABLE = 75%"
        )
        
        injuries = {}
        injury_multipliers = {
            'OUT': 0.0,
            'DOUBTFUL': 0.3,
            'QUESTIONABLE': 0.75,
            'PROBABLE': 0.95
        }
        
        for line in injury_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    player = parts[0].strip()
                    status = parts[1].strip().upper()
                    if player in df['Player'].values and status in injury_multipliers:
                        injuries[player] = status
                        mult = injury_multipliers[status]
                        df.loc[df['Player'] == player, 'Projected_Points'] *= mult
                        df.loc[df['Player'] == player, 'Injury_Status'] = status
    
    with col2:
        st.markdown("#### 📊 Ownership Projections")
        ownership_text = st.text_area(
            "Format: Player: %",
            height=120,
            placeholder="Josh Allen: 45\nStefon Diggs: 35\nTyreek Hill: 40",
            help="Enter projected ownership percentages"
        )
        
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
        
        df['Ownership'] = df['Player'].map(ownership_dict).fillna(OptimizerConfig.DEFAULT_OWNERSHIP)
    
    # Add ownership bucket column
    df['Bucket'] = df['Ownership'].apply(OwnershipBucketManager.get_bucket)
    
    # Display enhanced player pool
    st.markdown("### 👥 Player Pool Analysis")
    
    # Bucket distribution visualization
    bucket_counts = df['Bucket'].value_counts()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**Ownership Distribution:**")
        for bucket in ['mega_chalk', 'chalk', 'pivot', 'leverage', 'super_leverage']:
            count = bucket_counts.get(bucket, 0)
            emoji = {'mega_chalk': '🔴', 'chalk': '🟠', 'pivot': '🟢', 
                    'leverage': '🔵', 'super_leverage': '🟣'}.get(bucket, '')
            st.write(f"{emoji} {bucket}: {count}")
    
    with col2:
        # Show player pool
        display_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Value', 'Ownership', 'Bucket']
        
        # Add injury status if any injuries
        if injuries:
            display_cols.append('Injury_Status')
        
        st.dataframe(
            df[display_cols].sort_values('Projected_Points', ascending=False),
            height=300,
            use_container_width=True,
            column_config={
                "Salary": st.column_config.NumberColumn(format="$%d"),
                "Projected_Points": st.column_config.NumberColumn(format="%.1f"),
                "Value": st.column_config.NumberColumn(format="%.2f"),
                "Ownership": st.column_config.NumberColumn(format="%.1f%%")
            }
        )
    
    # Optimization Section
    st.markdown("---")
    st.markdown("## 🚀 Lineup Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lineups = st.number_input("Number of Lineups", 5, 50, 20, 5)
    
    with col2:
        optimization_focus = st.selectbox(
            "Optimization Focus",
            ["Balanced", "Tournament GPP", "Cash Game", "Single Entry"],
            help="Adjusts strategy distribution"
        )
    
    with col3:
        if st.button("🎯 Generate Optimized Lineups", type="primary", use_container_width=True):
            
            # Initialize optimizer
            optimizer = DualAIOptimizer(df, game_info, api_manager)
            
            # Get AI strategies
            with st.spinner("Getting AI strategic analysis..."):
                rec1, rec2 = optimizer.get_ai_strategies(use_api=use_api)
            
            # Display AI insights
            with st.expander("🧠 AI Strategic Insights", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Game Theory AI")
                    st.metric("Confidence", f"{rec1.confidence:.0%}")
                    
                    if rec1.captain_targets:
                        st.markdown("**Leverage Captains:**")
                        for captain in rec1.captain_targets[:3]:
                            own = df[df['Player'] == captain]['Ownership'].values[0] if captain in df['Player'].values else 5
                            st.write(f"• {captain} ({own:.0f}%)")
                    
                    if rec1.fades:
                        st.markdown("**Fade Targets:**")
                        for fade in rec1.fades[:3]:
                            st.write(f"• {fade}")
                    
                    if rec1.key_insights:
                        st.markdown("**Key Insights:**")
                        for insight in rec1.key_insights[:3]:
                            st.info(insight)
                
                with col2:
                    st.markdown("### 🔗 Correlation AI")
                    st.metric("Confidence", f"{rec2.confidence:.0%}")
                    
                    if rec2.stacks:
                        st.markdown("**Recommended Stacks:**")
                        for stack in rec2.stacks[:3]:
                            if isinstance(stack, dict):
                                p1 = stack.get('player1', '')
                                p2 = stack.get('player2', '')
                                if p1 and p2:
                                    st.write(f"• {p1} + {p2}")
                    
                    if rec2.key_insights:
                        st.markdown("**Key Insights:**")
                        for insight in rec2.key_insights[:3]:
                            st.info(insight)
            
            # Generate lineups
            with st.spinner(f"Generating {num_lineups} optimized lineups..."):
                lineups_df = optimizer.generate_optimized_lineups(
                    num_lineups, rec1, rec2, enforce_buckets=enforce_buckets
                )
            
            if lineups_df.empty:
                st.error("❌ No valid lineups could be generated. Try adjusting constraints.")
            else:
                st.success(f"✅ Generated {len(lineups_df)} lineups successfully!")
                
                # Calculate correlations for simulations
                correlations = optimizer.correlation_engine.calculate_dynamic_correlations(df, game_info)
                
                # Run simulations
                if use_correlation:
                    with st.spinner("Running tournament simulations..."):
                        for idx, row in lineups_df.iterrows():
                            sim_results = optimizer.tournament_sim.simulate_with_correlation(
                                row.to_dict(), df, correlations, n_sims=num_sims
                            )
                            for key, value in sim_results.items():
                                lineups_df.loc[idx, key] = value
                else:
                    # Simple projections without correlation
                    for idx, row in lineups_df.iterrows():
                        lineups_df.loc[idx, 'Mean'] = row['Projected']
                        lineups_df.loc[idx, 'Ceiling_90th'] = row['Projected'] * 1.3
                        lineups_df.loc[idx, 'Floor_10th'] = row['Projected'] * 0.7
                
                # Calculate composite scores
                lineups_df['GPP_Score'] = (
                    0.35 * lineups_df.get('Ceiling_90th', lineups_df['Projected']) +
                    0.25 * lineups_df.get('Ceiling_95th', lineups_df['Projected'] * 1.4) +
                    0.20 * (150 - lineups_df['Total_Ownership']) +
                    0.10 * lineups_df['Leverage_Score'] +
                    0.10 * lineups_df['Has_Stack'].astype(int) * 50
                )
                
                lineups_df['Cash_Score'] = (
                    0.50 * lineups_df.get('Median', lineups_df['Projected']) +
                    0.30 * lineups_df.get('Floor_10th', lineups_df['Projected'] * 0.7) +
                    0.20 * lineups_df['Projected']
                )
                
                # Sort based on focus
                if optimization_focus in ["Tournament GPP", "Single Entry"]:
                    lineups_df = lineups_df.sort_values('GPP_Score', ascending=False)
                else:
                    lineups_df = lineups_df.sort_values('Cash_Score', ascending=False)
                
                # Store in session state for access outside button
                st.session_state['lineups_df'] = lineups_df
                st.session_state['df'] = df
                st.session_state['correlations'] = correlations
                
                # Generate captain pivots if enabled
                if include_pivots:
                    with st.spinner("Generating captain pivots..."):
                        pivots_df = optimizer.pivot_generator.generate_pivots(
                            lineups_df.iloc[0].to_dict(), df, max_pivots=3
                        )
                        st.session_state['pivots_df'] = pivots_df
    
    # Display results if lineups exist
    if 'lineups_df' in st.session_state:
        lineups_df = st.session_state['lineups_df']
        df = st.session_state['df']
        
        st.markdown("---")
        st.markdown("## 📊 Optimization Results")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Lineups Generated", len(lineups_df))
        with col2:
            st.metric("Avg Ceiling (90th)", f"{lineups_df.get('Ceiling_90th', lineups_df['Projected']).mean():.1f}")
        with col3:
            st.metric("Avg Ownership", f"{lineups_df['Total_Ownership'].mean():.1f}%")
        with col4:
            st.metric("Avg Leverage", f"{lineups_df['Leverage_Score'].mean():.1f}")
        with col5:
            st.metric("Lineups w/ Stacks", lineups_df['Has_Stack'].sum())
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Lineups", "🔄 Captain Pivots", "📈 Analysis", "🎯 Top Plays", "💾 Export"
        ])
        
        with tab1:
            st.markdown("### Generated Lineups")
            
            # Display options
            col1, col2 = st.columns([3, 1])
            with col1:
                display_format = st.radio(
                    "Display Format",
                    ["Summary", "Detailed", "Compact"],
                    horizontal=True
                )
            with col2:
                show_top_n = st.number_input("Show Top", 5, len(lineups_df), 10, 5)
            
            if display_format == "Summary":
                display_cols = ['Lineup', 'Strategy', 'Captain', 'Projected', 'Ceiling_90th', 
                              'Total_Ownership', 'Leverage_Score', 'GPP_Score']
                
                st.dataframe(
                    lineups_df[display_cols].head(show_top_n),
                    use_container_width=True,
                    column_config={
                        "Projected": st.column_config.NumberColumn(format="%.1f"),
                        "Ceiling_90th": st.column_config.NumberColumn(format="%.1f"),
                        "Total_Ownership": st.column_config.NumberColumn(format="%.1f%%"),
                        "GPP_Score": st.column_config.NumberColumn(format="%.1f")
                    }
                )
            
            elif display_format == "Detailed":
                for i, (idx, lineup) in enumerate(lineups_df.head(show_top_n).iterrows(), 1):
                    with st.expander(f"Lineup #{i} - {lineup['Strategy']} - GPP Score: {lineup.get('GPP_Score', 0):.1f}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Roster:**")
                            st.write(f"🎯 **Captain:** {lineup['Captain']}")
                            st.write("**FLEX:**")
                            for player in lineup['FLEX']:
                                pos = df[df['Player'] == player]['Position'].values[0] if player in df['Player'].values else '??'
                                own = df[df['Player'] == player]['Ownership'].values[0] if player in df['Player'].values else 5
                                st.write(f"• {player} ({pos}) - {own:.0f}%")
                        
                        with col2:
                            st.markdown("**Projections:**")
                            st.metric("Mean", f"{lineup.get('Mean', lineup['Projected']):.1f}")
                            st.metric("90th %ile", f"{lineup.get('Ceiling_90th', lineup['Projected']*1.3):.1f}")
                            st.metric("95th %ile", f"{lineup.get('Ceiling_95th', lineup['Projected']*1.4):.1f}")
                            if 'Boom_Rate' in lineup:
                                st.metric("Boom Rate", f"{lineup['Boom_Rate']:.1f}%")
                        
                        with col3:
                            st.markdown("**Metrics:**")
                            st.write(f"💰 Salary: ${lineup['Salary']:,} (${lineup['Salary_Remaining']} left)")
                            st.write(f"📊 {lineup['Bucket_Summary']}")
                            st.write(f"🎯 Leverage Score: {lineup['Leverage_Score']:.1f}")
                            if lineup['Has_Stack']:
                                st.success(f"✅ Stack: {lineup['Stack_Details']}")
            
            else:  # Compact
                # Simple lineup display
                for i, (idx, lineup) in enumerate(lineups_df.head(show_top_n).iterrows(), 1):
                    st.write(f"**#{i}:** CPT: {lineup['Captain']} | FLEX: {', '.join(lineup['FLEX'][:3])}... | Own: {lineup['Total_Ownership']:.0f}%")
        
        with tab2:
            st.markdown("### Captain Pivot Variations")
            
           if 'pivots_df' in st.session_state and st.session_state['pivots_df']:
    pivots_df = st.session_state['pivots_df']
    
    st.info(f"Generated {len(pivots_df)} captain pivot variations")
    
    for pivot in pivots_df:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Original:** {pivot['Original_Captain']} → **New:** {pivot['Captain']}")
        with col2:
            st.write(f"Ownership Change: {pivot['Ownership_Delta']:+.1f}%")
        with col3:
            st.write(f"Type: {pivot['Pivot_Type']}")
else:
    st.info("Enable captain pivots in settings to generate variations")
        
        with tab3:
            st.markdown("### Lineup Analysis & Visualization")
            
            # Create analysis visualizations
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Strategy Distribution
            ax1 = axes[0, 0]
            strategy_counts = lineups_df['Strategy'].value_counts()
            colors = plt.cm.Set3(range(len(strategy_counts)))
            ax1.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.0f%%',
                   colors=colors, startangle=90)
            ax1.set_title('Strategy Distribution')
            
            # 2. Ownership vs Ceiling
            ax2 = axes[0, 1]
            ceiling_col = 'Ceiling_90th' if 'Ceiling_90th' in lineups_df else 'Projected'
            scatter = ax2.scatter(lineups_df['Total_Ownership'], lineups_df[ceiling_col],
                                c=lineups_df.get('GPP_Score', lineups_df['Projected']), 
                                cmap='viridis', s=100, alpha=0.6)
            ax2.set_xlabel('Total Ownership %')
            ax2.set_ylabel('90th Percentile Points')
            ax2.set_title('Ownership vs Upside')
            plt.colorbar(scatter, ax=ax2, label='GPP Score')
            
            # Annotate top 3
            for i, (idx, row) in enumerate(lineups_df.head(3).iterrows(), 1):
                ax2.annotate(f'#{i}', (row['Total_Ownership'], row.get(ceiling_col, row['Projected'])),
                           fontsize=12, fontweight='bold', color='red')
            
            # 3. Salary Distribution
            ax3 = axes[0, 2]
            ax3.hist(lineups_df['Salary'], bins=15, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(lineups_df['Salary'].mean(), color='red', linestyle='--', 
                       label=f'Avg: ${lineups_df["Salary"].mean():,.0f}')
            ax3.set_xlabel('Salary Used')
            ax3.set_ylabel('Number of Lineups')
            ax3.set_title('Salary Distribution')
            ax3.legend()
            
            # 4. Leverage Score Distribution
            ax4 = axes[1, 0]
            ax4.hist(lineups_df['Leverage_Score'], bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax4.set_xlabel('Leverage Score')
            ax4.set_ylabel('Number of Lineups')
            ax4.set_title('Leverage Distribution')
            
            # 5. Projection vs Ownership by Strategy
            ax5 = axes[1, 1]
            for strategy in lineups_df['Strategy'].unique():
                strategy_data = lineups_df[lineups_df['Strategy'] == strategy]
                ax5.scatter(strategy_data['Total_Ownership'], strategy_data['Projected'],
                          label=strategy, alpha=0.6, s=50)
            ax5.set_xlabel('Total Ownership %')
            ax5.set_ylabel('Projected Points')
            ax5.set_title('Strategy Comparison')
            ax5.legend(fontsize=8)
            
            # 6. Captain Analysis
            ax6 = axes[1, 2]
            captain_counts = lineups_df['Captain'].value_counts().head(10)
            y_pos = np.arange(len(captain_counts))
            ax6.barh(y_pos, captain_counts.values)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(captain_counts.index, fontsize=8)
            ax6.set_xlabel('Times Used as Captain')
            ax6.set_title('Top 10 Captains')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Ownership bucket heatmap
            st.markdown("### Ownership Bucket Distribution")
            
            bucket_data = []
            for idx, row in lineups_df.head(20).iterrows():
                bucket_counts = {'mega_chalk': 0, 'chalk': 0, 'pivot': 0, 'leverage': 0, 'super_leverage': 0}
                lineup_players = [row['Captain']] + row['FLEX']
                
                for player in lineup_players:
                    player_own = df[df['Player'] == player]['Ownership'].values
                    if len(player_own) > 0:
                        bucket = OwnershipBucketManager.get_bucket(player_own[0])
                        bucket_counts[bucket] += 1
                
                bucket_data.append(list(bucket_counts.values()))
            
            if bucket_data:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(bucket_data, cmap='RdYlGn_r', aspect='auto')
                
                ax.set_xticks(range(5))
                ax.set_xticklabels(['Mega\nChalk', 'Chalk', 'Pivot', 'Leverage', 'Super\nLeverage'])
                ax.set_yticks(range(len(bucket_data)))
                ax.set_yticklabels([f'#{i+1}' for i in range(len(bucket_data))])
                ax.set_title('Ownership Bucket Distribution (Top 20 Lineups)')
                ax.set_xlabel('Bucket Type')
                ax.set_ylabel('Lineup Rank')
                
                # Add text annotations
                for i in range(len(bucket_data)):
                    for j in range(5):
                        text = ax.text(j, i, bucket_data[i][j],
                                     ha="center", va="center",
                                     color="white" if bucket_data[i][j] > 3 else "black")
                
                plt.colorbar(im, ax=ax, label='Player Count')
                st.pyplot(fig)
        
        with tab4:
            st.markdown("### Top Plays Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Most Used Captains")
                captain_usage = lineups_df['Captain'].value_counts().head(10)
                for player, count in captain_usage.items():
                    own = df[df['Player'] == player]['Ownership'].values[0] if player in df['Player'].values else 5
                    pct = count / len(lineups_df) * 100
                    st.write(f"• {player} ({own:.0f}% own) - {count} lineups ({pct:.0f}%)")
            
            with col2:
                st.markdown("#### 🔗 Most Common Stacks")
                stack_counts = defaultdict(int)
                
                for idx, row in lineups_df.iterrows():
                    if row['Has_Stack'] and row['Stack_Details'] != 'None':
                        stacks = row['Stack_Details'].split(', ')
                        for stack in stacks:
                            stack_counts[stack] += 1
                
                if stack_counts:
                    sorted_stacks = sorted(stack_counts.items(), key=lambda x: x[1], reverse=True)
                    for stack, count in sorted_stacks[:10]:
                        pct = count / len(lineups_df) * 100
                        st.write(f"• {stack} - {count} lineups ({pct:.0f}%)")
                else:
                    st.info("No stacks found in lineups")
            
            st.markdown("#### 💎 Hidden Gems (Low Owned, High Usage)")
            
            # Find players with low ownership but high usage in lineups
            player_usage = defaultdict(int)
            for idx, row in lineups_df.iterrows():
                for player in [row['Captain']] + row['FLEX']:
                    player_usage[player] += 1
            
            gems = []
            for player, usage in player_usage.items():
                if player in df['Player'].values:
                    own = df[df['Player'] == player]['Ownership'].values[0]
                    if own < 10 and usage >= 3:  # Low owned but used 3+ times
                        gems.append((player, own, usage))
            
            if gems:
                gems.sort(key=lambda x: x[2], reverse=True)
                for player, own, usage in gems[:10]:
                    pct = usage / len(lineups_df) * 100
                    st.write(f"• {player} ({own:.0f}% own) - {usage} lineups ({pct:.0f}%)")
            else:
                st.info("No hidden gems found")
        
        with tab5:
            st.markdown("### 💾 Export Lineups")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### DraftKings Format")
                
                # Create DraftKings export
                dk_lineups = []
                for idx, row in lineups_df.iterrows():
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
                st.write("Preview (first 5):")
                st.dataframe(dk_df.head(), use_container_width=True)
                
                # Download button
                csv = dk_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download DraftKings CSV",
                    data=csv,
                    file_name=f"dk_lineups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("#### Full Analysis Export")
                
                if export_detailed:
                    # Prepare detailed export
                    export_df = lineups_df.copy()
                    export_df['FLEX'] = export_df['FLEX'].apply(lambda x: ', '.join(x))
                    
                    # Select columns for export
                    export_cols = ['Lineup', 'Strategy', 'Captain', 'FLEX', 'Projected', 
                                 'Salary', 'Total_Ownership', 'Leverage_Score']
                    
                    if 'Ceiling_90th' in export_df:
                        export_cols.extend(['Mean', 'Ceiling_90th', 'Ceiling_95th', 'Floor_10th'])
                    
                    export_cols.extend(['Bucket_Summary', 'Has_Stack', 'Stack_Details', 
                                      'GPP_Score', 'Cash_Score'])
                    
                    # Filter to existing columns
                    export_cols = [col for col in export_cols if col in export_df.columns]
                    
                    final_export = export_df[export_cols]
                else:
                    # Simple export
                    export_df = lineups_df.copy()
                    export_df['FLEX'] = export_df['FLEX'].apply(lambda x: ', '.join(x))
                    final_export = export_df[['Lineup', 'Captain', 'FLEX', 'Projected', 
                                            'Salary', 'Total_Ownership']]
                
                # Preview
                st.write("Preview (first 5):")
                st.dataframe(final_export.head(), use_container_width=True)
                
                # Download button
                csv_full = final_export.to_csv(index=False)
                st.download_button(
                    label="📊 Download Full Analysis",
                    data=csv_full,
                    file_name=f"lineup_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
### 📚 NFL Showdown Optimizer - Professional Edition

**Features:**
- 🤖 Dual AI Strategy System (Game Theory + Correlation)
- 📊 Advanced ownership bucketing and leverage scoring
- 🎯 Captain pivoting for lineup uniqueness
- 📈 Monte Carlo simulations with player correlations
- 🏆 Tournament-optimized lineup generation
- 💾 DraftKings-ready export formats

**Quick Tips:**
- Use API mode for faster optimization
- Enable correlation model for accurate simulations
- Check leverage scores for GPP tournaments
- Monitor ownership buckets for proper diversification

**Version:** 4.0 Professional | **Model:** Claude 3 Haiku

*Good luck with your lineups!* 🚀
""")

st.caption(f"Optimizer last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
