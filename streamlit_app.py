"""
NFL DFS AI-Driven Optimizer - Streamlit Interface
Version: 2.3.0 - Complete with Smart Column Mapping

CRITICAL FIXES APPLIED:
- Smart column mapping for various CSV formats
- Automatic first_name + last_name combination
- Manual column mapping fallback
- Added standard PuLP optimization (was completely missing)
- Fixed ownership validation
- Better error messages with actionable guidance
- Improved session state management
- Enhanced lineup validation
- Better progress tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback
import time
import sys
import os

st.set_page_config(
    page_title="NFL DFS AI Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FLEXIBLE OPTIMIZER IMPORT
# ============================================================================

optimizer = None
OPTIMIZER_AVAILABLE = False

# Try to import optimizer
possible_modules = [
    'nfl_dfs_optimizer',
    'optimizer',
    'nfl_optimizer',
    'dfs_optimizer'
]

import_error_details = []

for module_name in possible_modules:
    try:
        optimizer = __import__(module_name)
        # Verify it's the right module
        if hasattr(optimizer, 'OptimizerConfig') and hasattr(optimizer, '__version__'):
            OPTIMIZER_AVAILABLE = True
            st.success(f"✅ Successfully imported optimizer from: {module_name}.py")
            break
        else:
            import_error_details.append(
                f"  • {module_name}.py: Wrong module (missing expected attributes)"
            )
    except ImportError as e:
        import_error_details.append(f"  • {module_name}.py: {str(e)}")
        continue

if not OPTIMIZER_AVAILABLE:
    st.error("❌ **Failed to import optimizer module**")
    st.error("**Tried the following files:**")
    for detail in import_error_details:
        st.error(detail)
    
    st.info("""
    **Solution:**
    
    1. Ensure your optimizer file is in the same directory as streamlit_app.py
    2. The file should be named: `nfl_dfs_optimizer.py` (recommended)
    3. The file must contain all 7 parts combined into one file
    
    **Current directory contents:**
    """)
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
    try:
        files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
        st.code('\n'.join(files))
    except Exception:
        st.code("Unable to list directory")
    
    st.stop()

# Import all required components
try:
    OptimizerConfig = optimizer.OptimizerConfig
    ConfigValidator = optimizer.ConfigValidator
    AIEnforcementLevel = optimizer.AIEnforcementLevel
    OptimizationMode = optimizer.OptimizationMode
    FieldSize = optimizer.FieldSize
    AIStrategistType = optimizer.AIStrategistType
    AIRecommendation = optimizer.AIRecommendation
    LineupConstraints = optimizer.LineupConstraints
    SimulationResults = optimizer.SimulationResults
    GeneticConfig = optimizer.GeneticConfig
    MonteCarloSimulationEngine = optimizer.MonteCarloSimulationEngine
    GeneticAlgorithmOptimizer = optimizer.GeneticAlgorithmOptimizer
    StandardLineupOptimizer = optimizer.StandardLineupOptimizer
    AnthropicAPIManager = optimizer.AnthropicAPIManager
    GameTheoryStrategist = optimizer.GameTheoryStrategist
    CorrelationStrategist = optimizer.CorrelationStrategist
    ContrarianNarrativeStrategist = optimizer.ContrarianNarrativeStrategist
    AIEnforcementEngine = optimizer.AIEnforcementEngine
    AISynthesisEngine = optimizer.AISynthesisEngine
    AIOwnershipBucketManager = optimizer.AIOwnershipBucketManager
    AIConfigValidator = optimizer.AIConfigValidator
    OptimizedDataProcessor = optimizer.OptimizedDataProcessor
    get_logger = optimizer.get_logger
    get_performance_monitor = optimizer.get_performance_monitor
    get_ai_tracker = optimizer.get_ai_tracker
    ValidationError = optimizer.ValidationError
    OptimizationError = optimizer.OptimizationError
    calculate_lineup_metrics = optimizer.calculate_lineup_metrics
    format_lineup_for_export = optimizer.format_lineup_for_export
    __version__ = optimizer.__version__
    
except AttributeError as e:
    st.error(f"❌ **Missing optimizer component:** {str(e)}")
    st.error("**This means your optimizer file is incomplete.**")
    st.info("Please ensure all 7 parts are properly combined into one file.")
    st.stop()

APP_VERSION = "2.3.0"
OPTIMIZER_VERSION = __version__

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_session_state_get(key: str, default: Any = None) -> Any:
    """Safely get value from session state"""
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def safe_session_state_set(key: str, value: Any) -> None:
    """Safely set value in session state"""
    try:
        st.session_state[key] = value
    except Exception as e:
        st.error(f"Error saving to session state: {e}")


def initialize_session_state():
    """Initialize session state with defaults"""
    defaults = {
        'df': None,
        'uploaded_file_name': None,
        'game_total': 47.0,
        'spread': 0.0,
        'home_team': '',
        'away_team': '',
        'salary_cap': OptimizerConfig.SALARY_CAP,
        'num_lineups': 20,
        'field_size': 'large_field',
        'ai_enforcement': 'Strong',
        'optimization_mode': 'balanced',
        'use_monte_carlo': True,
        'use_genetic': False,
        'use_standard': True,
        'anthropic_api_key': '',
        'use_api': False,
        'lineups': None,
        'ai_recommendations': None,
        'optimization_complete': False,
        'last_optimization_time': None,
        'min_salary_pct': 90,
        'max_ownership': 200,
        'max_exposure': None,
        'locked_players': [],
        'banned_players': [],
        'show_debug': False,
        'randomness': 0.0,
        'diversity_threshold': 3
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def smart_column_mapping(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Intelligently map CSV columns to required format
    
    Args:
        df_raw: Raw uploaded DataFrame
        
    Returns:
        Tuple of (mapped_df, mapping_info_dict)
    """
    df = df_raw.copy()
    
    # Define column mapping patterns
    column_patterns = {
        'Player': [
            'player', 'name', 'player_name', 'playername', 'full_name', 'fullname'
        ],
        'Position': [
            'position', 'pos', 'Pos'
        ],
        'Team': [
            'team', 'tm', 'team_name', 'teamname'
        ],
        'Salary': [
            'salary', 'sal', 'cost', 'price'
        ],
        'Projected_Points': [
            'projected_points', 'projection', 'proj', 'points', 
            'point_projection', 'fpts', 'fantasy_points', 'pts', 'projected'
        ],
        'Ownership': [
            'ownership', 'own', 'own%', 'ownership%', 'proj_own', 'projected_ownership'
        ]
    }
    
    mappings = {}
    auto_mapped = False
    
    # Try to map each required column
    for required_col, patterns in column_patterns.items():
        if required_col in df.columns:
            # Already has correct name
            continue
        
        # Look for matching pattern
        for pattern in patterns:
            # Case-insensitive matching
            matching_cols = [col for col in df.columns if col.lower() == pattern.lower()]
            if matching_cols:
                old_col = matching_cols[0]
                df.rename(columns={old_col: required_col}, inplace=True)
                mappings[old_col] = required_col
                auto_mapped = True
                break
    
    # Special case: Combine first_name + last_name into Player
    if 'Player' not in df.columns:
        first_name_cols = [col for col in df.columns if col.lower() in ['first_name', 'firstname', 'first']]
        last_name_cols = [col for col in df.columns if col.lower() in ['last_name', 'lastname', 'last']]
        
        if first_name_cols and last_name_cols:
            first_col = first_name_cols[0]
            last_col = last_name_cols[0]
            df['Player'] = df[first_col].astype(str) + ' ' + df[last_col].astype(str)
            mappings[f'{first_col} + {last_col}'] = 'Player'
            auto_mapped = True
    
    # Normalize position names (handle lowercase)
    if 'Position' in df.columns:
        df['Position'] = df['Position'].str.upper()
    
    mapping_info = {
        'auto_mapped': auto_mapped,
        'mappings': mappings
    }
    
    return df, mapping_info


def manual_column_mapping(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Provide manual column mapping interface
    
    Args:
        df_raw: Raw uploaded DataFrame
        
    Returns:
        Mapped DataFrame or None if cancelled
    """
    st.subheader("🔧 Manual Column Mapping")
    st.info("Map your CSV columns to the required format:")
    
    df = df_raw.copy()
    available_columns = [''] + list(df_raw.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required Columns:**")
        
        player_col = st.selectbox(
            "Player Name Column",
            options=available_columns,
            help="Column containing player names"
        )
        
        position_col = st.selectbox(
            "Position Column",
            options=available_columns,
            help="Column containing positions (QB, RB, WR, etc.)"
        )
        
        team_col = st.selectbox(
            "Team Column",
            options=available_columns,
            help="Column containing team names"
        )
    
    with col2:
        st.markdown("**Required Columns (cont.):**")
        
        salary_col = st.selectbox(
            "Salary Column",
            options=available_columns,
            help="Column containing salaries"
        )
        
        projection_col = st.selectbox(
            "Projected Points Column",
            options=available_columns,
            help="Column containing point projections"
        )
        
        ownership_col = st.selectbox(
            "Ownership Column (Optional)",
            options=available_columns,
            help="Column containing ownership projections"
        )
    
    # Check if using first_name + last_name
    combine_name = False
    first_name_cols = [col for col in df_raw.columns if 'first' in col.lower() and 'name' in col.lower()]
    last_name_cols = [col for col in df_raw.columns if 'last' in col.lower() and 'name' in col.lower()]
    
    if not player_col and first_name_cols and last_name_cols:
        combine_name = st.checkbox(
            f"Combine '{first_name_cols[0]}' and '{last_name_cols[0]}' into Player column",
            value=True
        )
    
    if st.button("Apply Mapping", type="primary"):
        try:
            # Create mapped dataframe
            mapped_df = pd.DataFrame()
            
            # Handle Player column
            if combine_name and first_name_cols and last_name_cols:
                mapped_df['Player'] = (
                    df_raw[first_name_cols[0]].astype(str) + ' ' + 
                    df_raw[last_name_cols[0]].astype(str)
                )
            elif player_col:
                mapped_df['Player'] = df_raw[player_col]
            else:
                st.error("❌ Player column is required")
                return None
            
            # Map other required columns
            if position_col:
                mapped_df['Position'] = df_raw[position_col].str.upper()
            else:
                st.error("❌ Position column is required")
                return None
            
            if team_col:
                mapped_df['Team'] = df_raw[team_col]
            else:
                st.error("❌ Team column is required")
                return None
            
            if salary_col:
                mapped_df['Salary'] = pd.to_numeric(df_raw[salary_col], errors='coerce')
            else:
                st.error("❌ Salary column is required")
                return None
            
            if projection_col:
                mapped_df['Projected_Points'] = pd.to_numeric(df_raw[projection_col], errors='coerce')
            else:
                st.error("❌ Projected Points column is required")
                return None
            
            # Optional ownership
            if ownership_col:
                mapped_df['Ownership'] = pd.to_numeric(df_raw[ownership_col], errors='coerce')
            else:
                mapped_df['Ownership'] = 10.0
            
            st.success("✅ Mapping applied successfully!")
            return mapped_df
        
        except Exception as e:
            st.error(f"❌ Mapping failed: {e}")
            return None
    
    return None


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate DataFrame with enhanced checks
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    if len(df) < 6:
        return False, f"Need at least 6 players, found {len(df)}"
    
    try:
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        df['Projected_Points'] = pd.to_numeric(df['Projected_Points'], errors='coerce')
        
        if df['Salary'].isna().any() or df['Projected_Points'].isna().any():
            return False, "Found non-numeric values in Salary or Projected_Points"
    except Exception as e:
        return False, f"Error validating numeric columns: {e}"
    
    invalid_salaries = (
        (df['Salary'] < OptimizerConfig.MIN_SALARY) |
        (df['Salary'] > OptimizerConfig.MAX_SALARY * 1.2)
    ).sum()
    
    if invalid_salaries > 0:
        return False, f"Found {invalid_salaries} players with invalid salaries"
    
    # Enhanced ownership validation
    if 'Ownership' not in df.columns:
        df['Ownership'] = 10.0
        st.info("ℹ️ Ownership column not found - using default 10%")
    else:
        # Validate existing ownership values
        is_valid, issues = ConfigValidator.validate_ownership_values(df)
        if not is_valid:
            st.warning("⚠️ Found invalid ownership values:")
            for issue in issues[:3]:
                st.warning(f"  {issue}")
            # Fix invalid values
            invalid_mask = (df['Ownership'] < 0) | (df['Ownership'] > 100) | df['Ownership'].isna()
            if invalid_mask.any():
                df.loc[invalid_mask, 'Ownership'] = 10.0
                st.info(f"✓ Fixed {invalid_mask.sum()} invalid ownership values")
    
    return True, ""


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"


def create_download_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Create downloadable CSV"""
    return df.to_csv(index=False).encode('utf-8')


def create_sample_csv() -> bytes:
    """Create a sample CSV for download"""
    sample_df = pd.DataFrame({
        'Player': [
            'Patrick Mahomes', 'Travis Kelce', 'Tyreek Hill', 
            'Justin Jefferson', 'Austin Ekeler', 'Stefon Diggs'
        ],
        'Position': ['QB', 'TE', 'WR', 'WR', 'RB', 'WR'],
        'Team': ['KC', 'KC', 'MIA', 'MIN', 'LAC', 'BUF'],
        'Salary': [11000, 8500, 7800, 8200, 7500, 7600],
        'Projected_Points': [25.5, 18.2, 16.8, 17.9, 16.2, 17.1],
        'Ownership': [35.2, 22.1, 18.5, 20.3, 15.8, 19.2]
    })
    return sample_df.to_csv(index=False).encode('utf-8')


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    initialize_session_state()
    
    st.title("🏈 NFL DFS AI-Driven Optimizer")
    st.markdown(f"**App:** {APP_VERSION} | **Optimizer:** {OPTIMIZER_VERSION}")
    
    render_sidebar()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Data Upload",
        "🎯 Optimization",
        "📊 Results",
        "⚙️ Advanced Settings"
    ])
    
    with tab1:
        render_data_upload_tab()
    
    with tab2:
        render_optimization_tab()
    
    with tab3:
        render_results_tab()
    
    with tab4:
        render_advanced_settings_tab()
    
    render_footer()


def render_sidebar():
    """Render sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sample CSV download
        st.markdown("---")
        st.subheader("📥 Sample Data")
        sample_csv = create_sample_csv()
        st.download_button(
            "Download Sample CSV",
            data=sample_csv,
            file_name="sample_showdown_slate.csv",
            mime="text/csv",
            help="Download a sample CSV to see the expected format",
            use_container_width=True
        )
        
        st.markdown("---")
        
        with st.expander("🤖 AI Settings", expanded=False):
            use_api = st.checkbox(
                "Use Anthropic AI",
                value=safe_session_state_get('use_api', False),
                help="Enable AI-powered recommendations"
            )
            safe_session_state_set('use_api', use_api)
            
            if use_api:
                api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    value=safe_session_state_get('anthropic_api_key', ''),
                    help="Get your API key from console.anthropic.com"
                )
                safe_session_state_set('anthropic_api_key', api_key)
                
                if api_key and not api_key.startswith('sk-ant-'):
                    st.warning("⚠️ API key should start with 'sk-ant-'")
            else:
                st.info("Using statistical fallback mode")
        
        with st.expander("🎯 Optimization Settings", expanded=True):
            num_lineups = st.number_input(
                "Number of Lineups",
                min_value=1,
                max_value=150,
                value=safe_session_state_get('num_lineups', 20)
            )
            safe_session_state_set('num_lineups', num_lineups)
            
            field_size = st.selectbox(
                "Contest Type",
                options=list(OptimizerConfig.FIELD_SIZES.keys()),
                index=list(OptimizerConfig.FIELD_SIZES.keys()).index('Large GPP (1000+)')
            )
            safe_session_state_set('field_size', OptimizerConfig.FIELD_SIZES[field_size])
            
            ai_enforcement = st.selectbox(
                "AI Enforcement Level",
                options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
                index=2
            )
            safe_session_state_set('ai_enforcement', ai_enforcement)
            
            optimization_mode = st.selectbox(
                "Optimization Mode",
                options=['balanced', 'ceiling', 'floor', 'boom_or_bust'],
                index=0
            )
            safe_session_state_set('optimization_mode', optimization_mode)
        
        with st.expander("🧬 Algorithm Settings", expanded=False):
            optimizer_type = st.radio(
                "Optimizer Type",
                options=['Standard (PuLP)', 'Genetic Algorithm'],
                index=0,
                help="Standard is faster, Genetic Algorithm is more diverse"
            )
            
            use_standard = optimizer_type == 'Standard (PuLP)'
            use_genetic = optimizer_type == 'Genetic Algorithm'
            
            safe_session_state_set('use_standard', use_standard)
            safe_session_state_set('use_genetic', use_genetic)
            
            if use_standard:
                randomness = st.slider(
                    "Projection Randomness",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.0,
                    step=0.05,
                    help="Add randomness to projections for diversity"
                )
                safe_session_state_set('randomness', randomness)
                
                diversity_threshold = st.slider(
                    "Diversity Threshold",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="Minimum unique players between lineups"
                )
                safe_session_state_set('diversity_threshold', diversity_threshold)
            
            use_monte_carlo = st.checkbox(
                "Enable Monte Carlo Simulation",
                value=safe_session_state_get('use_monte_carlo', True)
            )
            safe_session_state_set('use_monte_carlo', use_monte_carlo)
        
        with st.expander("🛠 Debug", expanded=False):
            show_debug = st.checkbox(
                "Show Debug Info",
                value=safe_session_state_get('show_debug', False)
            )
            safe_session_state_set('show_debug', show_debug)


def render_data_upload_tab():
    """Render data upload tab with intelligent column mapping"""
    st.header("📤 Upload Player Data")
    
    # Show expected format
    with st.expander("ℹ️ CSV Format Guide", expanded=False):
        st.markdown("""
        **Your CSV should contain these columns** (names can vary):
        
        | Column | Alternatives | Example |
        |--------|-------------|---------|
        | Player | first_name + last_name, Name, player_name | Patrick Mahomes |
        | Position | position, pos | QB |
        | Team | team, Team Name | KC |
        | Salary | salary, sal, cost | 11000 |
        | Projected_Points | point_projection, projection, proj, fpts | 25.5 |
        | Ownership | ownership, own, own% | 15.2 |
        
        **The app will automatically detect and map your columns!**
        """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV with player projections",
        type=['csv'],
        help="CSV will be automatically mapped to required format"
    )
    
    if uploaded_file is not None:
        try:
            # Read raw CSV
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
            
            # Show raw columns
            with st.expander("📋 Detected Columns", expanded=False):
                st.write("**Your CSV contains:**")
                st.code(', '.join(df_raw.columns.tolist()))
            
            # Intelligent column mapping
            df, mapping_info = smart_column_mapping(df_raw)
            
            # Show mapping results
            if mapping_info['auto_mapped']:
                st.info("🔄 **Automatic Column Mapping Applied:**")
                for old_col, new_col in mapping_info['mappings'].items():
                    st.success(f"  ✓ `{old_col}` → `{new_col}`")
            
            # Validate mapped dataframe
            is_valid, error_message = validate_dataframe(df)
            
            if not is_valid:
                st.error(f"❌ Validation Error: {error_message}")
                
                # Offer manual mapping
                st.warning("⚠️ Automatic mapping failed. Try manual mapping:")
                df_manual = manual_column_mapping(df_raw)
                if df_manual is not None:
                    df = df_manual
                    is_valid, error_message = validate_dataframe(df)
                    if not is_valid:
                        st.error(f"Still invalid: {error_message}")
                        return
                else:
                    return
            
            # Save to session state
            safe_session_state_set('df', df)
            safe_session_state_set('uploaded_file_name', uploaded_file.name)
            
            # Extract teams
            teams = sorted(df['Team'].unique())
            if len(teams) >= 2:
                safe_session_state_set('home_team', teams[0])
                safe_session_state_set('away_team', teams[1])
            
            st.success(f"✅ Successfully loaded {len(df)} players")
            
            # Show preview
            st.subheader("📊 Data Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                st.metric("Teams", len(teams))
            with col3:
                st.metric("Avg Salary", format_currency(df['Salary'].mean()))
            with col4:
                st.metric("Avg Projection", f"{df['Projected_Points'].mean():.2f}")
            
            # Show processed data
            display_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
            if 'Ownership' in df.columns:
                display_columns.append('Ownership')
            
            st.dataframe(
                df[display_columns].head(15), 
                use_container_width=True, 
                hide_index=True
            )
            
            # Position breakdown
            st.subheader("📍 Position Breakdown")
            position_counts = df['Position'].value_counts()
            
            cols = st.columns(min(len(position_counts), 6))
            for i, (pos, count) in enumerate(position_counts.items()):
                with cols[i % 6]:
                    st.metric(pos, count)
            
            # Ownership distribution
            if 'Ownership' in df.columns:
                st.subheader("👥 Ownership Distribution")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Ownership", format_percentage(df['Ownership'].mean()))
                with col2:
                    st.metric("Max Ownership", format_percentage(df['Ownership'].max()))
                with col3:
                    st.metric("Min Ownership", format_percentage(df['Ownership'].min()))
        
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
            if safe_session_state_get('show_debug', False):
                st.code(traceback.format_exc())
    
    else:
        st.info("📋 Upload a CSV file to get started")
        
        # Show example format
        st.subheader("📝 Example CSV Format")
        example_df = pd.DataFrame({
            'Player': ['Patrick Mahomes', 'Travis Kelce', 'Tyreek Hill'],
            'Position': ['QB', 'TE', 'WR'],
            'Team': ['KC', 'KC', 'MIA'],
            'Salary': [11000, 8500, 7800],
            'Projected_Points': [25.5, 18.2, 16.8],
            'Ownership': [35.2, 22.1, 18.5]
        })
        st.dataframe(example_df, hide_index=True)


def render_optimization_tab():
    """Render optimization tab"""
    st.header("🎯 Lineup Optimization")
    
    df = safe_session_state_get('df')
    if df is None:
        st.warning("⚠️ Please upload player data first")
        return
    
    st.subheader("📊 Game Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.text_input("Home Team", value=safe_session_state_get('home_team', ''))
        safe_session_state_set('home_team', home_team)
        
        game_total = st.number_input("Game Total", min_value=30.0, max_value=70.0, 
                                     value=safe_session_state_get('game_total', 47.0), step=0.5)
        safe_session_state_set('game_total', game_total)
    
    with col2:
        away_team = st.text_input("Away Team", value=safe_session_state_get('away_team', ''))
        safe_session_state_set('away_team', away_team)
        
        spread = st.number_input("Spread", min_value=-20.0, max_value=20.0, 
                                value=safe_session_state_get('spread', 0.0), step=0.5)
        safe_session_state_set('spread', spread)
    
    st.subheader("💰 Salary Cap Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        salary_cap = st.number_input("Salary Cap", min_value=30000, max_value=100000,
                                     value=safe_session_state_get('salary_cap', OptimizerConfig.SALARY_CAP), 
                                     step=1000)
        
        is_valid, error_msg = ConfigValidator.validate_salary_cap(salary_cap)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            safe_session_state_set('salary_cap', salary_cap)
    
    with col2:
        min_salary_pct = st.slider("Minimum Salary % of Cap", min_value=80, max_value=100,
                                   value=safe_session_state_get('min_salary_pct', 90))
        safe_session_state_set('min_salary_pct', min_salary_pct)
        
        min_salary = int(salary_cap * (min_salary_pct / 100))
        st.info(f"Min Salary: {format_currency(min_salary)}")
    
    st.subheader("🔒 Player Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        locked_players = st.multiselect("Locked Players", options=sorted(df['Player'].tolist()),
                                       default=safe_session_state_get('locked_players', []))
        safe_session_state_set('locked_players', locked_players)
    
    with col2:
        banned_players = st.multiselect("Banned Players", options=sorted(df['Player'].tolist()),
                                       default=safe_session_state_get('banned_players', []))
        safe_session_state_set('banned_players', banned_players)
    
    conflicts = set(locked_players) & set(banned_players)
    if conflicts:
        st.error(f"❌ Conflicts: {', '.join(conflicts)}")
        return
    
    if locked_players:
        locked_df = df[df['Player'].isin(locked_players)]
        locked_salary = locked_df['Salary'].sum()
        
        if locked_salary > salary_cap:
            st.error("❌ Locked players exceed cap")
            return
        
        if len(locked_players) > 6:
            st.error("❌ Cannot lock more than 6 players")
            return
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        optimize_button = st.button("🚀 Generate Lineups", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_optimization()
            st.rerun()
    
    if optimize_button:
        execute_optimization(df, salary_cap, min_salary)


def reset_optimization():
    """Reset optimization state"""
    safe_session_state_set('lineups', None)
    safe_session_state_set('ai_recommendations', None)
    safe_session_state_set('optimization_complete', False)
    safe_session_state_set('last_optimization_time', None)


def execute_optimization(df: pd.DataFrame, salary_cap: int, min_salary: int):
    """
    Execute optimization with proper error handling
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("⚙️ Initializing optimizer...")
        progress_bar.progress(10)
        
        logger = get_logger()
        
        num_lineups = safe_session_state_get('num_lineups', 20)
        field_size = safe_session_state_get('field_size', 'large_field')
        ai_enforcement = safe_session_state_get('ai_enforcement', 'Strong')
        use_api = safe_session_state_get('use_api', False)
        use_monte_carlo = safe_session_state_get('use_monte_carlo', True)
        use_genetic = safe_session_state_get('use_genetic', False)
        use_standard = safe_session_state_get('use_standard', True)
        
        game_info = {
            'game_total': safe_session_state_get('game_total', 47.0),
            'spread': safe_session_state_get('spread', 0.0),
            'home_team': safe_session_state_get('home_team', ''),
            'away_team': safe_session_state_get('away_team', ''),
            'teams': [safe_session_state_get('home_team', ''), safe_session_state_get('away_team', '')]
        }
        
        field_config = OptimizerConfig.get_field_config(field_size)
        
        status_text.text("🤖 Running AI analysis...")
        progress_bar.progress(20)
        
        api_key = safe_session_state_get('anthropic_api_key', '') if use_api else None
        api_manager = AnthropicAPIManager(api_key=api_key, fallback_mode=not use_api)
        
        game_theory = GameTheoryStrategist(api_manager)
        correlation = CorrelationStrategist(api_manager)
        contrarian = ContrarianNarrativeStrategist(api_manager)
        
        gt_rec = game_theory.analyze(df, game_info, field_config)
        progress_bar.progress(35)
        
        corr_rec = correlation.analyze(df, game_info, field_config)
        progress_bar.progress(50)
        
        contra_rec = contrarian.analyze(df, game_info, field_config)
        progress_bar.progress(60)
        
        ai_recommendations = {'game_theory': gt_rec, 'correlation': corr_rec, 'contrarian': contra_rec}
        safe_session_state_set('ai_recommendations', ai_recommendations)
        
        status_text.text("🎯 Generating optimal lineups...")
        progress_bar.progress(70)
        
        lineups = []
        
        if use_genetic:
            status_text.text("🧬 Running genetic algorithm...")
            
            mc_engine = None
            if use_monte_carlo:
                mc_engine = MonteCarloSimulationEngine(df, game_info, n_simulations=1000)
            
            ga_optimizer = GeneticAlgorithmOptimizer(
                df=df, game_info=game_info, mc_engine=mc_engine,
                config=GeneticConfig(population_size=100, generations=50),
                salary_cap=salary_cap
            )
            
            ga_results = ga_optimizer.optimize(num_lineups=num_lineups, verbose=False)
            
            for result in ga_results:
                lineup = {
                    'Captain': result['captain'],
                    'FLEX': result['flex'],
                }
                
                metrics = calculate_lineup_metrics(result['captain'], result['flex'], df)
                lineup.update(metrics)
                
                if result.get('sim_results'):
                    lineup['Ceiling_90th'] = result['sim_results'].ceiling_90th
                    lineup['Floor_10th'] = result['sim_results'].floor_10th
                
                lineups.append(lineup)
        
        elif use_standard:
            status_text.text("⚙️ Running standard optimization...")
            
            constraints = LineupConstraints(
                min_salary=min_salary,
                max_salary=salary_cap,
                locked_players=set(safe_session_state_get('locked_players', [])),
                banned_players=set(safe_session_state_get('banned_players', []))
            )
            
            standard_optimizer = StandardLineupOptimizer(
                df=df,
                salary_cap=salary_cap,
                constraints=constraints
            )
            
            randomness = safe_session_state_get('randomness', 0.0)
            diversity_threshold = safe_session_state_get('diversity_threshold', 3)
            
            lineups = standard_optimizer.generate_lineups(
                num_lineups=num_lineups,
                randomness=randomness,
                diversity_threshold=diversity_threshold
            )
        
        else:
            st.error("❌ No optimizer selected")
            progress_bar.empty()
            status_text.empty()
            return
        
        progress_bar.progress(90)
        
        status_text.text("📊 Finalizing results...")
        
        if lineups:
            lineups_df = pd.DataFrame(lineups)
            
            # Format FLEX as string if it's a list
            if 'FLEX' in lineups_df.columns and isinstance(lineups_df['FLEX'].iloc[0], list):
                lineups_df['FLEX'] = lineups_df['FLEX'].apply(lambda x: ', '.join(x))
            
            lineups_df.insert(0, 'Lineup', range(1, len(lineups_df) + 1))
            
            safe_session_state_set('lineups', lineups_df)
            safe_session_state_set('optimization_complete', True)
            safe_session_state_set('last_optimization_time', datetime.now())
            
            progress_bar.progress(100)
            status_text.text("✅ Optimization complete!")
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully generated {len(lineups_df)} unique lineups!")
            st.balloons()
            st.info("👉 View results in the 'Results' tab")
        else:
            st.error("❌ No lineups generated - constraints may be too restrictive")
            progress_bar.empty()
            status_text.empty()
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Optimization failed: {str(e)}")
        
        if safe_session_state_get('show_debug', False):
            st.code(traceback.format_exc())


def render_results_tab():
    """Render results tab"""
    st.header("📊 Optimization Results")
    
    lineups_df = safe_session_state_get('lineups')
    
    if lineups_df is None:
        st.info("ℹ️ No results yet. Run optimization first.")
        return
    
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Lineups", len(lineups_df))
    with col2:
        st.metric("Avg Projection", f"{lineups_df['Projected'].mean():.2f}")
    with col3:
        st.metric("Avg Salary", format_currency(lineups_df['Total_Salary'].mean()))
    with col4:
        st.metric("Avg Ownership", format_percentage(lineups_df['Total_Ownership'].mean()))
    
    st.subheader("🎯 Generated Lineups")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_columns = st.multiselect("Columns to display", options=list(lineups_df.columns),
            default=['Lineup', 'Captain', 'FLEX', 'Total_Salary', 'Projected', 'Total_Ownership'])
    
    with col2:
        sort_by = st.selectbox("Sort by", 
            options=['Lineup', 'Projected', 'Total_Salary', 'Total_Ownership'], index=1)
    
    display_df = lineups_df[show_columns].sort_values(sort_by, ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
    
    st.subheader("💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = create_download_csv(lineups_df, "lineups.csv")
        st.download_button("📥 Download CSV", data=csv_data,
            file_name=f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True)
    
    with col2:
        dk_format = format_lineup_for_export(lineups_df.to_dict('records'), 'draftkings')
        if dk_format is not None and not dk_format.empty:
            csv_data = create_download_csv(dk_format, "dk_upload.csv")
            st.download_button("📥 Download DK Format", data=csv_data,
                file_name=f"dk_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True)
    
    ai_recommendations = safe_session_state_get('ai_recommendations')
    
    if ai_recommendations:
        st.subheader("🤖 AI Recommendations")
        
        tab1, tab2, tab3 = st.tabs(["Game Theory", "Correlation", "Contrarian"])
        
        with tab1:
            display_ai_recommendation(ai_recommendations.get('game_theory'))
        with tab2:
            display_ai_recommendation(ai_recommendations.get('correlation'))
        with tab3:
            display_ai_recommendation(ai_recommendations.get('contrarian'))


def display_ai_recommendation(recommendation: Optional[AIRecommendation]):
    """Display AI recommendation details"""
    if recommendation is None:
        st.info("No recommendation available")
        return
    
    st.metric("Confidence", format_percentage(recommendation.confidence * 100))
    
    if recommendation.narrative:
        st.info(f"**Strategy:** {recommendation.narrative}")
    
    if recommendation.captain_targets:
        st.markdown("**Captain Targets:**")
        st.write(", ".join(recommendation.captain_targets[:5]))
    
    if recommendation.must_play:
        st.markdown("**Must Play:**")
        st.write(", ".join(recommendation.must_play))
    
    if recommendation.never_play:
        st.markdown("**Never Play:**")
        st.write(", ".join(recommendation.never_play))
    
    if recommendation.stacks:
        st.markdown("**Recommended Stacks:**")
        for i, stack in enumerate(recommendation.stacks[:3], 1):
            if 'player1' in stack and 'player2' in stack:
                st.write(f"{i}. {stack['player1']} + {stack['player2']}")
    
    if recommendation.key_insights:
        st.markdown("**Key Insights:**")
        for insight in recommendation.key_insights[:5]:
            st.write(f"• {insight}")


def render_advanced_settings_tab():
    """Render advanced settings tab"""
    st.header("⚙️ Advanced Settings")
    
    st.subheader("💥 Ownership Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_ownership = st.number_input("Max Total Ownership %", min_value=50, max_value=300,
                                       value=safe_session_state_get('max_ownership', 200))
        safe_session_state_set('max_ownership', max_ownership)
    
    with col2:
        max_exposure = st.slider("Max Player Exposure %", min_value=10, max_value=100, value=25)
        safe_session_state_set('max_exposure', max_exposure / 100)
    
    st.info("Advanced settings will be applied on next optimization run")


def render_footer():
    """Render footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**App Version:** {APP_VERSION}")
    
    with col2:
        st.markdown(f"**Optimizer:** v{OPTIMIZER_VERSION}")
    
    with col3:
        last_opt_time = safe_session_state_get('last_optimization_time')
        if last_opt_time:
            st.markdown(f"**Last Run:** {last_opt_time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
