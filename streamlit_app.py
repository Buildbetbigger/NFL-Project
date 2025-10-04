"""
NFL DFS AI-Driven Optimizer - Streamlit Application
Complete Production-Ready UI with Smart CSV Handling & Lineup Diversity

Version: 2.0.0
Supports: DFF CSV format (first_name, last_name) with auto-ownership estimation
Features: Multiple unique lineups with configurable diversity
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
                'min_salary': 45000,
                'max_ownership_total': 150,
                'max_overlap': 4  # NEW: Diversity setting
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
# SMART FILE HANDLER WITH AUTO-OWNERSHIP ESTIMATION
# ============================================================================

class FileHandler:
    """Smart file handler with ownership estimation and flexible salary ranges"""
    
    # Your CSV format (DFF-style)
    REQUIRED_COLUMNS_DFF = ['first_name', 'last_name', 'position', 'team', 'salary', 'point_projection']
    
    # Alternative standard format
    REQUIRED_COLUMNS_STANDARD = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def load_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load and intelligently transform CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                return None, "CSV file is empty"
            
            # Auto-detect and transform format
            df, transform_message = FileHandler._smart_transform(df)
            
            if df is None:
                return None, transform_message
            
            # Validate data ranges
            validation_result = FileHandler._validate_data(df)
            if not validation_result['valid']:
                return None, validation_result['message']
            
            # Clean and prepare
            df = FileHandler._clean_dataframe(df)
            
            return df, f"✅ {len(df)} players loaded. {transform_message}"
            
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    @staticmethod
    def _smart_transform(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Intelligently transform CSV to required format
        Handles: DFF format (first_name, last_name) and standard format
        """
        
        # Check if already in standard format
        if all(col in df.columns for col in FileHandler.REQUIRED_COLUMNS_STANDARD):
            return df, "Standard format detected"
        
        # Check for DFF format (your format)
        if all(col in df.columns for col in FileHandler.REQUIRED_COLUMNS_DFF):
            df_new = pd.DataFrame()
            
            # 1. Combine names
            df_new['Player'] = (df['first_name'].astype(str).str.strip() + ' ' + 
                               df['last_name'].astype(str).str.strip())
            
            # 2. Map basic columns
            df_new['Position'] = df['position'].astype(str).str.strip().str.upper()
            df_new['Team'] = df['team'].astype(str).str.strip().str.upper()
            df_new['Salary'] = pd.to_numeric(df['salary'], errors='coerce')
            df_new['Projected_Points'] = pd.to_numeric(df['point_projection'], errors='coerce')
            
            # 3. SMART OWNERSHIP ESTIMATION
            df_new['Ownership'] = FileHandler._estimate_ownership(
                df_new['Salary'].values,
                df_new['Projected_Points'].values,
                df_new['Position'].values
            )
            
            messages = [
                "DFF format auto-converted",
                "✨ Ownership auto-estimated"
            ]
            
            return df_new, " | ".join(messages)
        
        # If neither format, provide helpful error
        found_cols = list(df.columns)
        return None, (
            f"❌ Unrecognized CSV format.\n\n"
            f"**Found columns:** {', '.join(found_cols)}\n\n"
            f"**Expected either:**\n"
            f"1. DFF Format: {', '.join(FileHandler.REQUIRED_COLUMNS_DFF)}\n"
            f"2. Standard: {', '.join(FileHandler.REQUIRED_COLUMNS_STANDARD)}"
        )
    
    @staticmethod
    def _estimate_ownership(salaries: np.ndarray, projections: np.ndarray, 
                           positions: np.ndarray) -> np.ndarray:
        """
        Smart ownership estimation algorithm
        
        Logic:
        1. Calculate value score (pts per $1000)
        2. Adjust for position popularity (QB/WR higher ownership)
        3. Normalize to realistic 0-100% range
        4. Add slight randomization for variance
        """
        
        # Prevent division by zero
        safe_salaries = np.maximum(salaries, 100)
        
        # Base value calculation (points per $1000 salary)
        value_scores = (projections / safe_salaries) * 1000
        
        # Position multipliers (reflects real DFS behavior)
        position_multipliers = {
            'QB': 1.3,   # QBs are popular
            'WR': 1.2,   # WRs are popular
            'RB': 1.1,   # RBs moderately popular
            'TE': 0.9,   # TEs less popular
            'K': 0.7,    # Kickers least popular
            'DST': 0.8,  # Defense moderate
            'D': 0.8,    # Defense alternate
            'FLEX': 1.0  # Neutral
        }
        
        # Apply position adjustments
        position_adjusted = np.array([
            value_scores[i] * position_multipliers.get(pos, 1.0)
            for i, pos in enumerate(positions)
        ])
        
        # Normalize to 0-1 range
        min_val = position_adjusted.min()
        max_val = position_adjusted.max()
        
        if max_val > min_val:
            normalized = (position_adjusted - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(position_adjusted) * 0.5
        
        # Scale to realistic ownership range (2% to 45%)
        # High value = high ownership
        ownership_pct = 2 + (normalized * 43)
        
        # Add slight randomization (±2%) for realism
        noise = np.random.normal(0, 2, len(ownership_pct))
        ownership_pct = ownership_pct + noise
        
        # Clip to valid range
        ownership_pct = np.clip(ownership_pct, 0.5, 99.0)
        
        return ownership_pct
    
    @staticmethod
    def _validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data with FLEXIBLE salary ranges
        Supports $100-$12,000 range (your format)
        """
        try:
            # Check salary range (flexible for different platforms)
            min_salary = df['Salary'].min()
            max_salary = df['Salary'].max()
            
            if min_salary < 50 or max_salary > 20000:
                return {
                    'valid': False,
                    'message': f"Unusual salary range: ${min_salary:,.0f} - ${max_salary:,.0f}"
                }
            
            # Check projections
            invalid_projections = (df['Projected_Points'] < 0) | (df['Projected_Points'] > 100)
            if invalid_projections.any():
                return {
                    'valid': False,
                    'message': f"{invalid_projections.sum()} players have invalid projections"
                }
            
            # Check for valid positions
            valid_positions = {'QB', 'RB', 'WR', 'TE', 'DST', 'K', 'FLEX', 'D'}
            invalid_positions = ~df['Position'].isin(valid_positions)
            
            if invalid_positions.any():
                unique_invalid = df.loc[invalid_positions, 'Position'].unique()
                return {
                    'valid': False,
                    'message': f"Invalid positions: {', '.join(unique_invalid)}"
                }
            
            # Minimum players check
            if len(df) < 6:
                return {
                    'valid': False,
                    'message': f"Need at least 6 players (found {len(df)})"
                }
            
            return {'valid': True, 'message': 'All validations passed'}
            
        except Exception as e:
            return {'valid': False, 'message': f"Validation error: {str(e)}"}
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Player'], keep='first')
        
        # Clean strings
        df['Player'] = df['Player'].str.strip()
        df['Position'] = df['Position'].str.strip().str.upper()
        df['Team'] = df['Team'].str.strip().str.upper()
        
        # Ensure numeric types
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        df['Projected_Points'] = pd.to_numeric(df['Projected_Points'], errors='coerce')
        df['Ownership'] = pd.to_numeric(df['Ownership'], errors='coerce').fillna(10.0)
        
        # Remove invalid rows
        df = df.dropna(subset=['Player', 'Position', 'Salary', 'Projected_Points'])
        
        # Sort by projection
        df = df.sort_values('Projected_Points', ascending=False).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def create_sample_csv() -> pd.DataFrame:
        """
        Create sample CSV in DFF format (your format)
        """
        return pd.DataFrame({
            'first_name': ['Patrick', 'Travis', 'Tyreek', 'Derrick', 'Justin', 'Mark', 'Josh', 'Stefon'],
            'last_name': ['Mahomes', 'Kelce', 'Hill', 'Henry', 'Jefferson', 'Andrews', 'Allen', 'Diggs'],
            'position': ['QB', 'TE', 'WR', 'RB', 'WR', 'TE', 'QB', 'WR'],
            'team': ['KC', 'KC', 'MIA', 'TEN', 'MIN', 'BAL', 'BUF', 'BUF'],
            'salary': [11000, 8500, 9000, 9500, 8800, 7500, 10500, 8000],
            'point_projection': [25.5, 18.2, 19.8, 21.3, 19.5, 16.8, 24.8, 17.9]
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
            placeholder="sk-ant-...",
            help="Get your API key at console.anthropic.com"
        )
        st.session_state.config['api_key'] = api_key
        
        if api_key and api_key.startswith('sk-ant-'):
            if st.session_state.api_manager is None:
                try:
                    with st.spinner("Validating API key..."):
                        st.session_state.api_manager = ClaudeAPIManager(api_key)
                        st.session_state.game_theory_ai = GPPGameTheoryStrategist(st.session_state.api_manager)
                        st.session_state.correlation_ai = GPPCorrelationStrategist(st.session_state.api_manager)
                        st.session_state.contrarian_ai = GPPContrarianNarrativeStrategist(st.session_state.api_manager)
                        st.sidebar.success("✅ API Connected")
                except Exception as e:
                    st.sidebar.error(f"API Error: {str(e)}")
            else:
                st.sidebar.success("✅ API Connected")
        elif api_key:
            st.sidebar.warning("API key should start with 'sk-ant-'")
    else:
        st.sidebar.info("Using statistical fallback mode")
    
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
            value=AIEnforcementLevel.STRONG.value,
            help="How strictly to enforce AI recommendations"
        )
    
    with st.sidebar.expander("🔧 Advanced"):
        st.session_state.config['run_simulation'] = st.checkbox(
            "Monte Carlo Simulation",
            value=st.session_state.config['run_simulation'],
            help="Run simulations to estimate variance"
        )
        
        if st.session_state.config['run_simulation']:
            st.session_state.config['num_simulations'] = st.slider(
                "Simulations",
                1000, 10000, 5000, 1000
            )
        
        st.session_state.config['use_genetic'] = st.checkbox(
            "Genetic Algorithm",
            value=st.session_state.config['use_genetic'],
            help="Evolutionary optimization (slower but unique lineups)"
        )
        
        if st.session_state.config['use_genetic']:
            st.session_state.config['genetic_generations'] = st.slider(
                "Generations",
                20, 100, 50, 10
            )
        
        st.markdown("---")
        
        # NEW: Lineup Diversity Settings
        st.markdown("**Lineup Diversity**")
        
        diversity_level = st.select_slider(
            "Diversity Level",
            options=['Low', 'Medium', 'High', 'Maximum'],
            value='Medium',
            help=(
                "How different lineups should be:\n"
                "- Low: 5/6 same players allowed\n"
                "- Medium: 4/6 same players allowed\n"
                "- High: 3/6 same players allowed\n"
                "- Maximum: 2/6 same players allowed"
            )
        )
        
        # Map diversity level to max overlap
        diversity_map = {
            'Low': 5,
            'Medium': 4,
            'High': 3,
            'Maximum': 2
        }
        st.session_state.config['max_overlap'] = diversity_map[diversity_level]
        
        st.caption(f"Max shared players: {diversity_map[diversity_level]}/6")
        
        st.markdown("---")
        
        st.session_state.config['min_salary'] = st.slider(
            "Min Total Salary",
            40000, 50000, 45000, 1000
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
        "📥 Download Sample CSV",
        data=csv,
        file_name="sample_dff_format.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download a sample CSV in DFF format"
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
        st.caption("DFF CSV Compatible")

def render_data_upload():
    """Render data upload section"""
    st.header("📂 Step 1: Upload Player Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Your CSV File",
            type=['csv'],
            help="Upload CSV with player data (supports DFF format)"
        )
        
        if uploaded_file:
            with st.spinner("Loading and validating data..."):
                df, message = FileHandler.load_csv(uploaded_file)
                
                if df is not None:
                    st.session_state.player_df = df
                    UIHelpers.show_success(message)
                    st.session_state.current_step = 2
                    
                    # Show ownership estimation details
                    if "auto-estimated" in message.lower():
                        with st.expander("ℹ️ About Auto-Estimated Ownership"):
                            st.markdown("""
                            **Ownership was automatically estimated** based on:
                            - Player value (projection ÷ salary)
                            - Position popularity (QB/WR higher, K/TE lower)
                            - Normalized to realistic 2-45% range
                            
                            This provides realistic tournament simulation even without ownership data.
                            """)
                    
                    # Show preview
                    render_data_preview(df)
                else:
                    UIHelpers.show_error(message)
    
    with col2:
        st.markdown("### 📋 CSV Format")
        st.markdown("""
        **DFF Format (Recommended):**
        - `first_name`
        - `last_name`
        - `position`
        - `team`
        - `salary`
        - `point_projection`
        
        **Or Standard Format:**
        - `Player`
        - `Position`
        - `Team`
        - `Salary`
        - `Projected_Points`
        - `Ownership`
        
        💡 Ownership auto-estimated if not provided
        """)

def render_data_preview(df: pd.DataFrame):
    """Render data preview with statistics"""
    st.markdown("### 📊 Data Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Avg Salary", UIHelpers.format_currency(df['Salary'].mean()))
    with col3:
        st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
    with col4:
        st.metric("Avg Ownership", UIHelpers.format_percentage(df['Ownership'].mean()))
    
    # Position breakdown
    st.markdown("#### Position Breakdown")
    position_counts = df['Position'].value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            position_counts.reset_index().rename(columns={'index': 'Position', 'Position': 'Count'}),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig = px.pie(
                values=position_counts.values,
                names=position_counts.index,
                title="Position Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Top players
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔝 Top 10 by Projection")
        top_proj = df.nlargest(10, 'Projected_Points')[
            ['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']
        ].copy()
        top_proj['Salary'] = top_proj['Salary'].apply(UIHelpers.format_currency)
        top_proj['Ownership'] = top_proj['Ownership'].apply(UIHelpers.format_percentage)
        top_proj.columns = ['Player', 'Pos', 'Salary', 'Proj', 'Own%']
        st.dataframe(top_proj, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### 💎 Best Value Plays")
        df_value = df.copy()
        df_value['Value'] = df_value['Projected_Points'] / (df_value['Salary'] / 1000)
        top_value = df_value.nlargest(10, 'Value')[
            ['Player', 'Position', 'Salary', 'Projected_Points', 'Value', 'Ownership']
        ].copy()
        top_value['Salary'] = top_value['Salary'].apply(UIHelpers.format_currency)
        top_value['Value'] = top_value['Value'].round(2)
        top_value['Ownership'] = top_value['Ownership'].apply(UIHelpers.format_percentage)
        top_value.columns = ['Player', 'Pos', 'Salary', 'Proj', 'Value', 'Own%']
        st.dataframe(top_value, hide_index=True, use_container_width=True)
    
    # Ownership details in expander
    with st.expander("📊 Detailed Ownership Analysis"):
        ownership_df = df[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].copy()
        ownership_df['Value'] = (ownership_df['Projected_Points'] / ownership_df['Salary'] * 1000).round(2)
        ownership_df = ownership_df.sort_values('Ownership', ascending=False)
        
        st.markdown("**All players sorted by ownership (highest to lowest)**")
        st.dataframe(ownership_df, use_container_width=True, height=400)

def render_game_info():
    """Render game information"""
    if st.session_state.player_df is None:
        return
    
    st.header("⚙️ Step 2: Configure Game")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.game_info['teams'] = st.text_input(
            "Matchup",
            value=st.session_state.game_info['teams'],
            placeholder="Chiefs vs Bills"
        )
        st.session_state.game_info['total'] = st.number_input(
            "Game Total (O/U)",
            30.0, 65.0,
            st.session_state.game_info['total'],
            0.5
        )
    
    with col2:
        st.session_state.game_info['spread'] = st.number_input(
            "Spread",
            -21.0, 21.0,
            st.session_state.game_info['spread'],
            0.5,
            help="Negative = favorite"
        )
        st.session_state.game_info['weather'] = st.selectbox(
            "Weather",
            ['Clear', 'Dome', 'Light Rain', 'Heavy Rain', 'Snow', 'Wind'],
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
        UIHelpers.show_info("AI analysis disabled. Using statistical mode. Enable API in sidebar for AI analysis.")
        st.session_state.current_step = 4
        return
    
    if not st.session_state.api_manager:
        UIHelpers.show_warning("Please add your Anthropic API key in the sidebar to enable AI analysis.")
        st.session_state.current_step = 4
        return
    
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        run_ai_analysis()
    
    # Show results if available
    if st.session_state.ai_recommendations:
        display_ai_recommendations()
        st.session_state.current_step = 4

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
            st.warning(f"Game Theory analysis issue: {str(e)[:100]}")
        
        # Correlation
        status.text("🔗 Running Correlation Analysis...")
        try:
            recommendations[AIStrategistType.CORRELATION] = st.session_state.correlation_ai.get_recommendation(
                df, game_info, field_size, use_api=True
            )
            progress_bar.progress(0.7)
        except Exception as e:
            st.warning(f"Correlation analysis issue: {str(e)[:100]}")
        
        # Contrarian
        status.text("💡 Running Contrarian Analysis...")
        try:
            recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = st.session_state.contrarian_ai.get_recommendation(
                df, game_info, field_size, use_api=True
            )
            progress_bar.progress(0.9)
        except Exception as e:
            st.warning(f"Contrarian analysis issue: {str(e)[:100]}")
        
        # Synthesis
        if len(recommendations) >= 2:
            status.text("🔄 Synthesizing Recommendations...")
            try:
                synthesis_engine = AISynthesisEngine()
                
                # Use whatever recommendations we have
                gt = recommendations.get(AIStrategistType.GAME_THEORY)
                corr = recommendations.get(AIStrategistType.CORRELATION)
                cont = recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE)
                
                # Create dummy recommendations if missing
                if not gt:
                    gt = AIRecommendation(captain_targets=[], source_ai=AIStrategistType.GAME_THEORY)
                if not corr:
                    corr = AIRecommendation(captain_targets=[], source_ai=AIStrategistType.CORRELATION)
                if not cont:
                    cont = AIRecommendation(captain_targets=[], source_ai=AIStrategistType.CONTRARIAN_NARRATIVE)
                
                st.session_state.synthesis = synthesis_engine.synthesize_recommendations(gt, corr, cont)
            except Exception as e:
                st.warning(f"Synthesis issue: {str(e)[:100]}")
        
        progress_bar.progress(1.0)
        status.empty()
        
        if recommendations:
            st.session_state.ai_recommendations = recommendations
            st.session_state.analysis_complete = True
            UIHelpers.show_success(f"✅ AI Analysis Complete! ({len(recommendations)} strategists)")
            st.rerun()
        else:
            UIHelpers.show_error("AI analysis failed. Check API key and try again.")
        
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
                st.write(", ".join(rec.captain_targets[:7]))
            
            if rec.must_play:
                st.markdown("**Must Include:**")
                st.write(", ".join(rec.must_play[:5]))
            
            if rec.key_insights:
                st.markdown("**Key Insights:**")
                for insight in rec.key_insights:
                    st.markdown(f"- {insight}")
        else:
            st.info("Game Theory analysis not available")
    
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
                for i, stack in enumerate(rec.stacks[:5], 1):
                    if 'player1' in stack and 'player2' in stack:
                        st.markdown(f"{i}. {stack['player1']} + {stack['player2']}")
            
            if rec.captain_targets:
                st.markdown("**Recommended Captains for Stacking:**")
                st.write(", ".join(rec.captain_targets[:7]))
        else:
            st.info("Correlation analysis not available")
    
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
                st.write(", ".join(rec.captain_targets[:7]))
            
            if rec.never_play:
                st.markdown("**Fade List (Avoid These):**")
                st.write(", ".join(rec.never_play[:5]))
        else:
            st.info("Contrarian analysis not available")
    
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
                    st.write(", ".join(consensus_captains[:7]))
                else:
                    st.info("No strong consensus on captains - diverse strategies recommended")
            
            if synth.get('patterns'):
                st.markdown("**Strategic Patterns:**")
                for pattern in synth['patterns']:
                    st.markdown(f"- {pattern}")
        else:
            st.info("Synthesis not available")

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
    """Execute lineup optimization with proper uniqueness constraints"""
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, value
        
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
            try:
                enforcement_engine = AIEnforcementEngine(
                    AIEnforcementLevel[config['enforcement_level'].upper().replace(' ', '_')]
                )
                enforcement_rules = enforcement_engine.create_enforcement_rules(
                    st.session_state.ai_recommendations
                )
            except:
                pass
        
        progress_bar.progress(0.2)
        
        # Run optimization
        lineups = []
        
        if config['use_genetic']:
            status.text("🧬 Running Genetic Algorithm...")
            mc_engine = None
            if config['run_simulation']:
                mc_engine = MonteCarloSimulationEngine(df, game_info, config['num_simulations'])
            
            ga_optimizer = GeneticAlgorithmOptimizer(df, game_info, mc_engine)
            ga_config = GeneticConfig(generations=config['genetic_generations'])
            ga_optimizer.config = ga_config
            
            results = ga_optimizer.optimize(config['num_lineups'], verbose=False)
            
            for r in results:
                captain_sal = df.loc[df['Player']==r['captain'], 'Salary'].iloc[0] if not df[df['Player']==r['captain']].empty else 0
                flex_sal = sum(df.loc[df['Player']==p, 'Salary'].iloc[0] for p in r['flex'] if not df[df['Player']==p].empty)
                total_sal = captain_sal * OptimizerConfig.CAPTAIN_MULTIPLIER + flex_sal
                
                captain_proj = df.loc[df['Player']==r['captain'], 'Projected_Points'].iloc[0] if not df[df['Player']==r['captain']].empty else 0
                flex_proj = sum(df.loc[df['Player']==p, 'Projected_Points'].iloc[0] for p in r['flex'] if not df[df['Player']==p].empty)
                total_proj = captain_proj * OptimizerConfig.CAPTAIN_MULTIPLIER + flex_proj
                
                lineups.append({
                    'Lineup': len(lineups) + 1,
                    'Captain': r['captain'],
                    'FLEX': ', '.join(r['flex']),
                    'Total_Salary': total_sal,
                    'Projected': total_proj,
                    'Fitness': r['fitness']
                })
            
            progress_bar.progress(0.9)
            
        else:
            status.text("🎯 Running Linear Programming Optimization...")
            
            # FIXED: Proper multi-lineup generation with uniqueness
            player_list = df['Player'].tolist()
            player_indices = {p: i for i, p in enumerate(player_list)}
            
            # Store used lineups to ensure uniqueness
            used_lineups = []
            
            # Calculate salary cap dynamically
            max_possible_salary = df['Salary'].max() * OptimizerConfig.CAPTAIN_MULTIPLIER + df['Salary'].nlargest(5).sum()
            salary_cap = min(max_possible_salary * 0.98, OptimizerConfig.SALARY_CAP)
            
            target_lineups = config['num_lineups']
            max_attempts = target_lineups * 3
            
            # Get diversity setting
            max_overlap = config.get('max_overlap', 4)
            
            for attempt in range(max_attempts):
                try:
                    # Create NEW problem for each lineup
                    prob = LpProblem(f"DFS_Lineup_{attempt}", LpMaximize)
                    
                    # Create variables
                    captain_vars = LpVariable.dicts("captain", player_indices.values(), cat=LpBinary)
                    flex_vars = LpVariable.dicts("flex", player_indices.values(), cat=LpBinary)
                    
                    # Objective: Maximize projected points
                    prob += lpSum([
                        df.iloc[i]['Projected_Points'] * OptimizerConfig.CAPTAIN_MULTIPLIER * captain_vars[i]
                        for i in range(len(player_list))
                    ]) + lpSum([
                        df.iloc[i]['Projected_Points'] * flex_vars[i]
                        for i in range(len(player_list))
                    ])
                    
                    # CONSTRAINT 1: Exactly 1 captain
                    prob += lpSum([captain_vars[i] for i in range(len(player_list))]) == 1
                    
                    # CONSTRAINT 2: Exactly 5 FLEX
                    prob += lpSum([flex_vars[i] for i in range(len(player_list))]) == 5
                    
                    # CONSTRAINT 3: Player can't be both captain and flex
                    for i in range(len(player_list)):
                        prob += captain_vars[i] + flex_vars[i] <= 1
                    
                    # CONSTRAINT 4: Salary cap
                    prob += lpSum([
                        df.iloc[i]['Salary'] * OptimizerConfig.CAPTAIN_MULTIPLIER * captain_vars[i]
                        for i in range(len(player_list))
                    ]) + lpSum([
                        df.iloc[i]['Salary'] * flex_vars[i]
                        for i in range(len(player_list))
                    ]) <= salary_cap
                    
                    # CONSTRAINT 5: Minimum salary
                    prob += lpSum([
                        df.iloc[i]['Salary'] * OptimizerConfig.CAPTAIN_MULTIPLIER * captain_vars[i]
                        for i in range(len(player_list))
                    ]) + lpSum([
                        df.iloc[i]['Salary'] * flex_vars[i]
                        for i in range(len(player_list))
                    ]) >= config['min_salary']
                    
                    # CONSTRAINT 6: Uniqueness with configurable diversity
                    for used_lineup in used_lineups:
                        used_captain_idx = player_indices[used_lineup['captain']]
                        used_flex_indices = [player_indices[p] for p in used_lineup['flex']]
                        
                        # This lineup can have at most max_overlap players from any previous lineup
                        prob += (
                            captain_vars[used_captain_idx] +
                            lpSum([flex_vars[i] for i in used_flex_indices])
                        ) <= max_overlap
                    
                    # CONSTRAINT 7: AI Recommendations (if available)
                    if enforcement_rules and enforcement_rules.get('hard_constraints'):
                        for rule in enforcement_rules['hard_constraints'][:3]:  # Apply top 3 rules
                            rule_type = rule.get('rule')
                            
                            if rule_type == 'captain_from_list':
                                captain_players = rule.get('players', [])
                                captain_player_indices = [
                                    player_indices[p] for p in captain_players 
                                    if p in player_indices
                                ]
                                if captain_player_indices:
                                    # Captain must be from this list
                                    prob += lpSum([captain_vars[i] for i in captain_player_indices]) >= 1
                            
                            elif rule_type == 'must_include':
                                player = rule.get('player')
                                if player and player in player_indices:
                                    p_idx = player_indices[player]
                                    # Player must be in lineup (either captain or flex)
                                    prob += captain_vars[p_idx] + flex_vars[p_idx] >= 1
                    
                    # Solve
                    prob.solve()
                    
                    # Check if we found a solution
                    if LpStatus[prob.status] != 'Optimal':
                        # If we can't find more lineups, stop
                        if len(lineups) > 0:
                            break
                        else:
                            continue
                    
                    # Extract solution
                    captain_idx = [i for i in range(len(player_list)) if value(captain_vars[i]) == 1]
                    flex_indices = [i for i in range(len(player_list)) if value(flex_vars[i]) == 1]
                    
                    if not captain_idx or len(flex_indices) != 5:
                        continue
                    
                    captain = player_list[captain_idx[0]]
                    flex = [player_list[i] for i in flex_indices]
                    
                    # Calculate metrics
                    captain_data = df[df['Player'] == captain].iloc[0]
                    flex_data = df[df['Player'].isin(flex)]
                    
                    total_salary = (
                        captain_data['Salary'] * OptimizerConfig.CAPTAIN_MULTIPLIER +
                        flex_data['Salary'].sum()
                    )
                    
                    total_proj = (
                        captain_data['Projected_Points'] * OptimizerConfig.CAPTAIN_MULTIPLIER +
                        flex_data['Projected_Points'].sum()
                    )
                    
                    total_own = (
                        captain_data['Ownership'] * OptimizerConfig.CAPTAIN_MULTIPLIER +
                        flex_data['Ownership'].sum()
                    )
                    
                    # Add to results
                    lineup_dict = {
                        'captain': captain,
                        'flex': flex
                    }
                    
                    lineups.append({
                        'Lineup': len(lineups) + 1,
                        'Captain': captain,
                        'FLEX': ', '.join(flex),
                        'Total_Salary': total_salary,
                        'Projected': total_proj,
                        'Total_Ownership': total_own,
                        'Avg_Ownership': total_own / 6
                    })
                    
                    used_lineups.append(lineup_dict)
                    
                    # Update progress
                    progress = 0.2 + (len(lineups) / target_lineups) * 0.7
                    progress_bar.progress(min(0.95, progress))
                    status.text(f"🎯 Generated {len(lineups)}/{target_lineups} lineups...")
                    
                    # Stop if we have enough
                    if len(lineups) >= target_lineups:
                        break
                    
                except Exception as e:
                    # If this iteration fails, continue to next
                    continue
            
            # If we didn't get enough lineups, show warning
            if len(lineups) < target_lineups:
                st.warning(f"⚠️ Generated {len(lineups)} lineups (requested {target_lineups}). Try relaxing diversity or increasing player pool.")
        
        progress_bar.progress(1.0)
        status.empty()
        
        if lineups:
            st.session_state.optimized_lineups = pd.DataFrame(lineups)
            st.session_state.optimization_complete = True
            UIHelpers.show_success(f"✅ Generated {len(lineups)} unique lineups!")
            st.rerun()
        else:
            UIHelpers.show_error("Failed to generate lineups. Try adjusting settings or check your data.")
        
    except Exception as e:
        UIHelpers.show_error(f"Optimization failed: {str(e)}")
        st.error(traceback.format_exc())

def display_optimization_results():
    """Display optimization results"""
    st.markdown("### 📊 Optimized Lineups")
    
    lineups_df = st.session_state.optimized_lineups
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Lineups", len(lineups_df))
    with col2:
        if 'Projected' in lineups_df.columns:
            st.metric("Avg Projection", f"{lineups_df['Projected'].mean():.1f}")
    with col3:
        if 'Total_Salary' in lineups_df.columns:
            st.metric("Avg Salary", UIHelpers.format_currency(lineups_df['Total_Salary'].mean()))
    with col4:
        if 'Projected' in lineups_df.columns:
            st.metric("Best Lineup", f"{lineups_df['Projected'].max():.1f} pts")
    
    # Display table
    display_df = lineups_df.copy()
    if 'Total_Salary' in display_df.columns:
        display_df['Total_Salary'] = display_df['Total_Salary'].apply(lambda x: f"${x:,.0f}")
    if 'Projected' in display_df.columns:
        display_df['Projected'] = display_df['Projected'].round(1)
    if 'Total_Ownership' in display_df.columns:
        display_df['Total_Ownership'] = display_df['Total_Ownership'].round(1)
    if 'Avg_Ownership' in display_df.columns:
        display_df['Avg_Ownership'] = display_df['Avg_Ownership'].round(1)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export
    st.markdown("### 💾 Export Lineups")
    
    # Check if xlsxwriter is available
    try:
        import xlsxwriter
        EXCEL_AVAILABLE = True
    except ImportError:
        EXCEL_AVAILABLE = False
    
    # Dynamic column layout based on Excel availability
    if EXCEL_AVAILABLE:
        col1, col2, col3 = st.columns(3)
    else:
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
    
    # Excel export (only if available)
    if EXCEL_AVAILABLE:
        with col3:
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    lineups_df.to_excel(writer, sheet_name='Lineups', index=False)
                excel_buffer.seek(0)
                
                st.download_button(
                    "📥 Excel Format",
                    data=excel_buffer,
                    file_name=f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Excel export error: {str(e)}")
    else:
        st.info("💡 Excel export not available. Install xlsxwriter to enable.")

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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>NFL DFS AI-Driven Optimizer v{} | Built with ❤️ and AI</p>
        <p>⚠️ For entertainment purposes only. Always gamble responsibly.</p>
    </div>
    """.format(__version__), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
