
"""
NFL DFS AI-Driven Optimizer - Streamlit Application
Complete Production-Ready UI

Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
import time
from datetime import datetime
import warnings
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Suppress warnings in UI
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT OPTIMIZER COMPONENTS
# ============================================================================

try:
    from nfl_dfs_optimizer import (
        # Enums
        FieldSize,
        AIStrategistType,
        AIEnforcementLevel,
        FitnessMode,
        ConstraintPriority,

        # Data Classes
        AIRecommendation,
        SimulationResults,
        GeneticLineup,
        GeneticConfig,

        # Configuration
        OptimizerConfig,

        # Core Components
        ClaudeAPIManager,
        GPPGameTheoryStrategist,
        GPPCorrelationStrategist,
        GPPContrarianNarrativeStrategist,
        MonteCarloSimulationEngine,
        GeneticAlgorithmOptimizer,
        AIEnforcementEngine,
        AIOwnershipBucketManager,
        AISynthesisEngine,
        AIConfigValidator,
        OptimizedDataProcessor,

        # Utilities
        get_logger,
        get_performance_monitor,
        get_ai_tracker,

        # Version
        __version__
    )
    OPTIMIZER_LOADED = True
except ImportError as e:
    st.error(f"❌ Failed to import optimizer: {e}")
    st.error("Please ensure 'nfl_dfs_optimizer.py' is in the same directory.")
    st.stop()

# Try importing plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL DFS AI Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    h1 {color: #1f77b4; padding-bottom: 10px; border-bottom: 2px solid #1f77b4;}
    h2 {color: #ff7f0e; margin-top: 20px;}
    h3 {color: #2ca02c;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 4px solid #1f77b4;}
    .success-box {padding: 15px; background-color: #d4edda; border-left: 5px solid #28a745; border-radius: 5px; margin: 10px 0;}
    .error-box {padding: 15px; background-color: #f8d7da; border-left: 5px solid #dc3545; border-radius: 5px; margin: 10px 0;}
    .warning-box {padding: 15px; background-color: #fff3cd; border-left: 5px solid #ffc107; border-radius: 5px; margin: 10px 0;}
    .info-box {padding: 15px; background-color: #d1ecf1; border-left: 5px solid #17a2b8; border-radius: 5px; margin: 10px 0;}
    .stButton > button {width: 100%; background-color: #1f77b4; color: white; font-weight: bold; border-radius: 5px; padding: 10px; border: none;}
    .stButton > button:hover {background-color: #155a8a; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

class SessionStateManager:
    """Centralized session state management"""

    @staticmethod
    def initialize():
        """Initialize all session state variables"""

        if 'player_df' not in st.session_state:
            st.session_state.player_df = None

        if 'game_info' not in st.session_state:
            st.session_state.game_info = {
                'teams': 'Team A vs Team B',
                'total': 45.0,
                'spread': 0.0,
                'weather': 'Clear',
                'primetime': False,
                'injury_count': 0
            }

        if 'api_manager' not in st.session_state:
            st.session_state.api_manager = None

        if 'game_theory_ai' not in st.session_state:
            st.session_state.game_theory_ai = None

        if 'correlation_ai' not in st.session_state:
            st.session_state.correlation_ai = None

        if 'contrarian_ai' not in st.session_state:
            st.session_state.contrarian_ai = None

        if 'ai_recommendations' not in st.session_state:
            st.session_state.ai_recommendations = {}

        if 'synthesis' not in st.session_state:
            st.session_state.synthesis = None

        if 'optimized_lineups' not in st.session_state:
            st.session_state.optimized_lineups = None

        if 'config' not in st.session_state:
            st.session_state.config = {
                'num_lineups': 20,
                'field_size': FieldSize.LARGE.value,
                'use_api': True,
                'api_key': '',
                'enforcement_level': AIEnforcementLevel.STRONG.value,
                'run_simulation': True,
                'use_genetic': False,
                'num_simulations': 5000,
                'genetic_generations': 50,
                'min_salary': 49000,
                'max_ownership_total': 150
            }

        if 'optimization_complete' not in st.session_state:
            st.session_state.optimization_complete = False

        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False

        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1

    @staticmethod
    def reset_optimization():
        """Reset optimization-related state"""
        st.session_state.ai_recommendations = {}
        st.session_state.synthesis = None
        st.session_state.optimized_lineups = None
        st.session_state.optimization_complete = False
        st.session_state.analysis_complete = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class UIHelpers:
    """UI helper functions"""

    @staticmethod
    def show_success(message: str):
        st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def show_error(message: str):
        st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def show_warning(message: str):
        st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def show_info(message: str):
        st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def format_currency(value: float) -> str:
        return f"${value:,.0f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        return f"{value:.1f}%"

    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"

# ============================================================================
# FILE HANDLER
# ============================================================================

class FileHandler:
    """File handling with validation"""

    REQUIRED_COLUMNS = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']

    @staticmethod
    def load_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load and validate CSV file"""
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                return None, "CSV file is empty"

            missing_cols = [col for col in FileHandler.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return None, f"Missing columns: {', '.join(missing_cols)}"

            # Clean data
            df = df.drop_duplicates(subset=['Player'], keep='first')
            df['Player'] = df['Player'].str.strip()
            df['Position'] = df['Position'].str.strip()
            df['Team'] = df['Team'].str.strip()
            df['Ownership'] = df['Ownership'].fillna(10.0)

            # Validate ranges
            if df['Salary'].min() < 3000 or df['Salary'].max() > 15000:
                return None, "Invalid salary values detected"

            if df['Ownership'].min() < 0 or df['Ownership'].max() > 100:
                return None, "Invalid ownership values detected"

            return df, f"Successfully loaded {len(df)} players"

        except Exception as e:
            return None, f"Error loading file: {str(e)}"

    @staticmethod
    def create_sample_csv() -> pd.DataFrame:
        """Create sample CSV"""
        return pd.DataFrame({
            'Player': ['Patrick Mahomes', 'Travis Kelce', 'Tyreek Hill', 'Derrick Henry',
                      'Justin Jefferson', 'Mark Andrews', 'Josh Allen', 'Stefon Diggs'],
            'Position': ['QB', 'TE', 'WR', 'RB', 'WR', 'TE', 'QB', 'WR'],
            'Team': ['KC', 'KC', 'MIA', 'TEN', 'MIN', 'BAL', 'BUF', 'BUF'],
            'Salary': [11000, 8500, 9000, 9500, 8800, 7500, 10500, 8000],
            'Projected_Points': [25.5, 18.2, 19.8, 21.3, 19.5, 16.8, 24.8, 17.9],
            'Ownership': [35.2, 28.5, 22.1, 18.7, 25.3, 15.9, 32.8, 20.4]
        })

# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

def render_sidebar():
    """Render complete sidebar"""

    st.sidebar.title("🏈 NFL DFS Optimizer")
    st.sidebar.markdown(f"**Version:** {__version__}")
    st.sidebar.markdown("---")

    # API Configuration
    st.sidebar.header("🔑 API Configuration")

    use_api = st.sidebar.checkbox(
        "Use AI Analysis",
        value=st.session_state.config['use_api'],
        help="Enable Claude AI for strategic analysis"
    )
    st.session_state.config['use_api'] = use_api

    if use_api:
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            value=st.session_state.config['api_key'],
            type="password",
            placeholder="sk-ant-..."
        )
        st.session_state.config['api_key'] = api_key

        if api_key and api_key.startswith('sk-ant-'):
            if st.session_state.api_manager is None:
                try:
                    st.session_state.api_manager = ClaudeAPIManager(api_key)
                    st.session_state.game_theory_ai = GPPGameTheoryStrategist(st.session_state.api_manager)
                    st.session_state.correlation_ai = GPPCorrelationStrategist(st.session_state.api_manager)
                    st.session_state.contrarian_ai = GPPContrarianNarrativeStrategist(st.session_state.api_manager)
                    st.sidebar.success("✅ API Connected")
                except Exception as e:
                    st.sidebar.error(f"API Error: {str(e)}")
            else:
                st.sidebar.success("✅ API Connected")

    st.sidebar.markdown("---")

    # Optimization Settings
    st.sidebar.header("⚙️ Settings")

    st.session_state.config['num_lineups'] = st.sidebar.number_input(
        "Number of Lineups",
        min_value=1,
        max_value=150,
        value=st.session_state.config['num_lineups'],
        step=1
    )

    field_size_options = [
        FieldSize.SMALL.value,
        FieldSize.MEDIUM.value,
        FieldSize.LARGE.value,
        FieldSize.LARGE_AGGRESSIVE.value,
        FieldSize.MILLY_MAKER.value
    ]

    st.session_state.config['field_size'] = st.sidebar.selectbox(
        "Tournament Size",
        options=field_size_options,
        index=2
    )

    if use_api:
        enforcement_options = [
            AIEnforcementLevel.MANDATORY.value,
            AIEnforcementLevel.STRONG.value,
            AIEnforcementLevel.MODERATE.value,
            AIEnforcementLevel.ADVISORY.value
        ]

        st.session_state.config['enforcement_level'] = st.sidebar.select_slider(
            "AI Enforcement",
            options=enforcement_options,
            value=AIEnforcementLevel.STRONG.value
        )

    with st.sidebar.expander("🔧 Advanced"):
        st.session_state.config['run_simulation'] = st.checkbox(
            "Monte Carlo Simulation",
            value=st.session_state.config['run_simulation']
        )

        if st.session_state.config['run_simulation']:
            st.session_state.config['num_simulations'] = st.slider(
                "Simulations",
                1000, 10000, 5000, 1000
            )

        st.session_state.config['use_genetic'] = st.checkbox(
            "Genetic Algorithm",
            value=st.session_state.config['use_genetic']
        )

        if st.session_state.config['use_genetic']:
            st.session_state.config['genetic_generations'] = st.slider(
                "Generations",
                20, 100, 50, 10
            )

        st.session_state.config['min_salary'] = st.slider(
            "Min Salary",
            45000, 50000, 49000, 500
        )

    st.sidebar.markdown("---")

    # Actions
    if st.sidebar.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    sample_df = FileHandler.create_sample_csv()
    csv = sample_df.to_csv(index=False)
    st.sidebar.download_button(
        "📥 Sample CSV",
        data=csv,
        file_name="sample.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# MAIN UI COMPONENTS
# ============================================================================

def render_header():
    """Render header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏈 NFL DFS AI-Driven Optimizer")
        st.markdown("**Advanced DraftKings Showdown Optimizer** powered by triple-AI analysis")
    with col2:
        st.metric("Version", __version__)

def render_data_upload():
    """Render data upload section"""
    st.header("📂 Step 1: Upload Player Data")

    uploaded_file = st.file_uploader(
        "Upload DraftKings CSV",
        type=['csv'],
        help="Upload CSV with player data"
    )

    if uploaded_file:
        with st.spinner("Loading data..."):
            df, message = FileHandler.load_csv(uploaded_file)

            if df is not None:
                st.session_state.player_df = df
                UIHelpers.show_success(message)
                st.session_state.current_step = 2

                # Show preview
                st.markdown("### 📊 Data Preview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Players", len(df))
                with col2:
                    st.metric("Avg Salary", UIHelpers.format_currency(df['Salary'].mean()))
                with col3:
                    st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
                with col4:
                    st.metric("Avg Ownership", UIHelpers.format_percentage(df['Ownership'].mean()))

                st.dataframe(df.head(10), use_container_width=True)
            else:
                UIHelpers.show_error(message)

def render_game_info():
    """Render game information"""
    if st.session_state.player_df is None:
        return

    st.header("⚙️ Step 2: Configure Game")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.game_info['teams'] = st.text_input(
            "Matchup",
            value=st.session_state.game_info['teams']
        )
        st.session_state.game_info['total'] = st.number_input(
            "Game Total",
            30.0, 65.0,
            st.session_state.game_info['total'],
            0.5
        )

    with col2:
        st.session_state.game_info['spread'] = st.number_input(
            "Spread",
            -21.0, 21.0,
            st.session_state.game_info['spread'],
            0.5
        )
        st.session_state.game_info['weather'] = st.selectbox(
            "Weather",
            ['Clear', 'Dome', 'Rain', 'Snow', 'Wind'],
            index=0
        )

    with col3:
        st.session_state.game_info['primetime'] = st.checkbox(
            "Primetime Game",
            value=st.session_state.game_info['primetime']
        )
        st.session_state.game_info['injury_count'] = st.number_input(
            "Notable Injuries",
            0, 10,
            st.session_state.game_info['injury_count'],
            1
        )

    st.session_state.current_step = 3

def render_ai_analysis():
    """Render AI analysis section"""
    if st.session_state.player_df is None:
        return

    st.header("🤖 Step 3: AI Strategic Analysis")

    if not st.session_state.config['use_api']:
        UIHelpers.show_info("AI analysis disabled. Enable in sidebar to use AI strategists.")
        st.session_state.current_step = 4
        return

    if not st.session_state.api_manager:
        UIHelpers.show_warning("Please add API key in sidebar to enable AI analysis.")
        return

    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        run_ai_analysis()

    # Show results if available
    if st.session_state.ai_recommendations:
        display_ai_recommendations()

def run_ai_analysis():
    """Execute AI analysis"""
    try:
        df = st.session_state.player_df
        game_info = st.session_state.game_info
        field_size = st.session_state.config['field_size']

        progress_bar = st.progress(0)
        status = st.empty()

        recommendations = {}

        # Game Theory
        status.text("🎮 Running Game Theory Analysis...")
        progress_bar.progress(0.1)
        try:
            recommendations[AIStrategistType.GAME_THEORY] = st.session_state.game_theory_ai.get_recommendation(
                df, game_info, field_size, use_api=True
            )
            progress_bar.progress(0.4)
        except Exception as e:
            st.error(f"Game Theory error: {str(e)}")

        # Correlation
        status.text("🔗 Running Correlation Analysis...")
        try:
            recommendations[AIStrategistType.CORRELATION] = st.session_state.correlation_ai.get_recommendation(
                df, game_info, field_size, use_api=True
            )
            progress_bar.progress(0.7)
        except Exception as e:
            st.error(f"Correlation error: {str(e)}")

        # Contrarian
        status.text("💡 Running Contrarian Analysis...")
        try:
            recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = st.session_state.contrarian_ai.get_recommendation(
                df, game_info, field_size, use_api=True
            )
            progress_bar.progress(0.9)
        except Exception as e:
            st.error(f"Contrarian error: {str(e)}")

        # Synthesis
        status.text("🔄 Synthesizing Recommendations...")
        if len(recommendations) == 3:
            synthesis_engine = AISynthesisEngine()
            st.session_state.synthesis = synthesis_engine.synthesize_recommendations(
                recommendations[AIStrategistType.GAME_THEORY],
                recommendations[AIStrategistType.CORRELATION],
                recommendations[AIStrategistType.CONTRARIAN_NARRATIVE]
            )

        progress_bar.progress(1.0)
        status.empty()

        st.session_state.ai_recommendations = recommendations
        st.session_state.analysis_complete = True
        st.session_state.current_step = 4

        UIHelpers.show_success("✅ AI Analysis Complete!")
        st.rerun()

    except Exception as e:
        UIHelpers.show_error(f"Analysis failed: {str(e)}")
        st.error(traceback.format_exc())

def display_ai_recommendations():
    """Display AI recommendations"""
    st.markdown("### 📋 AI Recommendations")

    tabs = st.tabs(["🎮 Game Theory", "🔗 Correlation", "💡 Contrarian", "🎯 Synthesis"])

    # Game Theory Tab
    with tabs[0]:
        if AIStrategistType.GAME_THEORY in st.session_state.ai_recommendations:
            rec = st.session_state.ai_recommendations[AIStrategistType.GAME_THEORY]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{rec.confidence:.0%}")
            with col2:
                st.metric("Captain Options", len(rec.captain_targets))

            st.markdown(f"**Strategy:** {rec.narrative}")

            if rec.captain_targets:
                st.markdown("**Recommended Captains:**")
                st.write(", ".join(rec.captain_targets[:5]))

            if rec.must_play:
                st.markdown("**Must Include:**")
                st.write(", ".join(rec.must_play[:3]))

            if rec.key_insights:
                st.markdown("**Key Insights:**")
                for insight in rec.key_insights:
                    st.markdown(f"- {insight}")

    # Correlation Tab
    with tabs[1]:
        if AIStrategistType.CORRELATION in st.session_state.ai_recommendations:
            rec = st.session_state.ai_recommendations[AIStrategistType.CORRELATION]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{rec.confidence:.0%}")
            with col2:
                st.metric("Stacks Identified", len(rec.stacks))

            st.markdown(f"**Strategy:** {rec.narrative}")

            if rec.stacks:
                st.markdown("**Primary Stacks:**")
                for i, stack in enumerate(rec.stacks[:3], 1):
                    if 'player1' in stack and 'player2' in stack:
                        st.markdown(f"{i}. {stack['player1']} + {stack['player2']}")

    # Contrarian Tab
    with tabs[2]:
        if AIStrategistType.CONTRARIAN_NARRATIVE in st.session_state.ai_recommendations:
            rec = st.session_state.ai_recommendations[AIStrategistType.CONTRARIAN_NARRATIVE]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{rec.confidence:.0%}")
            with col2:
                st.metric("Leverage Plays", len(rec.captain_targets))

            st.markdown(f"**Narrative:** {rec.narrative}")

            if rec.captain_targets:
                st.markdown("**Contrarian Captains:**")
                st.write(", ".join(rec.captain_targets[:5]))

            if rec.never_play:
                st.markdown("**Fade List:**")
                st.write(", ".join(rec.never_play[:3]))

    # Synthesis Tab
    with tabs[3]:
        if st.session_state.synthesis:
            synth = st.session_state.synthesis

            st.metric("Combined Confidence", f"{synth.get('confidence', 0):.0%}")

            if synth.get('captain_strategy'):
                st.markdown("**Captain Consensus:**")
                consensus_captains = [
                    k for k, v in synth['captain_strategy'].items()
                    if v in ['consensus', 'majority']
                ]
                if consensus_captains:
                    st.write(", ".join(consensus_captains[:5]))

            if synth.get('patterns'):
                st.markdown("**Strategic Patterns:**")
                for pattern in synth['patterns']:
                    st.markdown(f"- {pattern}")

def render_optimization():
    """Render optimization section"""
    if st.session_state.player_df is None:
        return

    st.header("🎯 Step 4: Generate Lineups")

    if st.button("⚡ Optimize Lineups", type="primary", use_container_width=True):
        run_optimization()

    if st.session_state.optimized_lineups is not None:
        display_optimization_results()

def run_optimization():
    """Execute lineup optimization"""
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus

        df = st.session_state.player_df
        game_info = st.session_state.game_info
        config = st.session_state.config

        progress_bar = st.progress(0)
        status = st.empty()

        status.text("🔧 Setting up optimization...")
        progress_bar.progress(0.1)

        # Create enforcement rules if AI was used
        enforcement_rules = {}
        if st.session_state.ai_recommendations:
            enforcement_engine = AIEnforcementEngine(
                AIEnforcementLevel[config['enforcement_level'].upper()]
            )
            enforcement_rules = enforcement_engine.create_enforcement_rules(
                st.session_state.ai_recommendations
            )

        progress_bar.progress(0.2)

        # Run optimization based on method
        lineups = []

        if config['use_genetic']:
            status.text("🧬 Running Genetic Algorithm...")
            mc_engine = None
            if config['run_simulation']:
                mc_engine = MonteCarloSimulationEngine(
                    df, game_info, config['num_simulations']
                )

            ga_optimizer = GeneticAlgorithmOptimizer(df, game_info, mc_engine)
            ga_config = GeneticConfig(generations=config['genetic_generations'])
            ga_optimizer.config = ga_config

            results = ga_optimizer.optimize(config['num_lineups'], verbose=False)

            for r in results:
                lineups.append({
                    'Captain': r['captain'],
                    'FLEX': ', '.join(r['flex']),
                    'Fitness': r['fitness']
                })

            progress_bar.progress(0.9)

        else:
            status.text("🎯 Running Linear Programming Optimization...")

            # Simple LP optimization
            lineups_generated = 0
            used_lineups = set()

            for attempt in range(config['num_lineups'] * 3):
                try:
                    prob = LpProblem("DFS", LpMaximize)

                    # Variables
                    captain_vars = {p: LpVariable(f"capt_{p}", cat=LpBinary) for p in df['Player']}
                    flex_vars = {p: LpVariable(f"flex_{p}", cat=LpBinary) for p in df['Player']}

                    # Objective
                    prob += lpSum([
                        df.loc[df['Player']==p, 'Projected_Points'].iloc[0] *
                        OptimizerConfig.CAPTAIN_MULTIPLIER * captain_vars[p]
                        for p in df['Player']
                    ]) + lpSum([
                        df.loc[df['Player']==p, 'Projected_Points'].iloc[0] * flex_vars[p]
                        for p in df['Player']
                    ])

                    # Constraints
                    prob += lpSum(captain_vars.values()) == 1
                    prob += lpSum(flex_vars.values()) == 5

                    for p in df['Player']:
                        prob += captain_vars[p] + flex_vars[p] <= 1

                    # Salary
                    prob += lpSum([
                        df.loc[df['Player']==p, 'Salary'].iloc[0] *
                        OptimizerConfig.CAPTAIN_MULTIPLIER * captain_vars[p]
                        for p in df['Player']
                    ]) + lpSum([
                        df.loc[df['Player']==p, 'Salary'].iloc[0] * flex_vars[p]
                        for p in df['Player']
                    ]) <= OptimizerConfig.SALARY_CAP

                    prob.solve()

                    if LpStatus[prob.status] == 'Optimal':
                        captain = [p for p in df['Player'] if captain_vars[p].varValue == 1][0]
                        flex = [p for p in df['Player'] if flex_vars[p].varValue == 1]

                        lineup_key = f"{captain}_{'_'.join(sorted(flex))}"

                        if lineup_key not in used_lineups:
                            used_lineups.add(lineup_key)

                            total_sal = (
                                df.loc[df['Player']==captain, 'Salary'].iloc[0] *
                                OptimizerConfig.CAPTAIN_MULTIPLIER +
                                sum(df.loc[df['Player']==p, 'Salary'].iloc[0] for p in flex)
                            )

                            total_proj = (
                                df.loc[df['Player']==captain, 'Projected_Points'].iloc[0] *
                                OptimizerConfig.CAPTAIN_MULTIPLIER +
                                sum(df.loc[df['Player']==p, 'Projected_Points'].iloc[0] for p in flex)
                            )

                            lineups.append({
                                'Lineup': len(lineups) + 1,
                                'Captain': captain,
                                'FLEX': ', '.join(flex),
                                'Total_Salary': total_sal,
                                'Projected': total_proj
                            })

                            lineups_generated += 1

                            if lineups_generated >= config['num_lineups']:
                                break

                            # Add uniqueness constraint
                            prob += lpSum([
                                captain_vars[captain]
                            ]) + lpSum([
                                flex_vars[p] for p in flex
                            ]) <= 5

                    progress_bar.progress(min(0.9, 0.2 + (lineups_generated / config['num_lineups']) * 0.7))

                except Exception as e:
                    continue

        progress_bar.progress(1.0)
        status.empty()

        if lineups:
            st.session_state.optimized_lineups = pd.DataFrame(lineups)
            st.session_state.optimization_complete = True
            UIHelpers.show_success(f"✅ Generated {len(lineups)} lineups!")
            st.rerun()
        else:
            UIHelpers.show_error("Failed to generate lineups. Try adjusting settings.")

    except Exception as e:
        UIHelpers.show_error(f"Optimization failed: {str(e)}")
        st.error(traceback.format_exc())

def display_optimization_results():
    """Display optimization results"""
    st.markdown("### 📊 Optimized Lineups")

    lineups_df = st.session_state.optimized_lineups

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Lineups", len(lineups_df))
    with col2:
        if 'Projected' in lineups_df.columns:
            st.metric("Avg Projection", f"{lineups_df['Projected'].mean():.1f}")
    with col3:
        if 'Total_Salary' in lineups_df.columns:
            st.metric("Avg Salary", UIHelpers.format_currency(lineups_df['Total_Salary'].mean()))

    st.dataframe(lineups_df, use_container_width=True, height=400)

    # Export
    st.markdown("### 💾 Export Lineups")

    col1, col2 = st.columns(2)

    with col1:
        csv = lineups_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # DraftKings format
        dk_format = []
        for _, row in lineups_df.iterrows():
            flex_players = row['FLEX'].split(', ') if isinstance(row['FLEX'], str) else []
            dk_format.append({
                'CPT': row['Captain'],
                'FLEX1': flex_players[0] if len(flex_players) > 0 else '',
                'FLEX2': flex_players[1] if len(flex_players) > 1 else '',
                'FLEX3': flex_players[2] if len(flex_players) > 2 else '',
                'FLEX4': flex_players[3] if len(flex_players) > 3 else '',
                'FLEX5': flex_players[4] if len(flex_players) > 4 else ''
            })

        dk_df = pd.DataFrame(dk_format)
        dk_csv = dk_df.to_csv(index=False)
        st.download_button(
            "📥 DraftKings Format",
            data=dk_csv,
            file_name=f"dk_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    SessionStateManager.initialize()

    render_sidebar()
    render_header()

    st.markdown("---")

    render_data_upload()

    if st.session_state.player_df is not None:
        st.markdown("---")
        render_game_info()

        st.markdown("---")
        render_ai_analysis()

        st.markdown("---")
        render_optimization()

if __name__ == "__main__":
    main()
