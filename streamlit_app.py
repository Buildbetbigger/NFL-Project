
"""
NFL DFS AI-Driven Optimizer - Streamlit Application
Production-Ready UI with Comprehensive Error Handling

IMPROVEMENTS:
- Robust session state management
- Comprehensive input validation
- Better error messaging
- Progress tracking
- Safe file handling
- Memory management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import io
import traceback
from datetime import datetime
import warnings

# Suppress warnings in UI
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL DFS AI Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/nfl-dfs-optimizer',
        'Report a bug': 'https://github.com/yourusername/nfl-dfs-optimizer/issues',
        'About': """
        # NFL DFS AI-Driven Optimizer

        Advanced DFS lineup optimization powered by triple-AI strategic analysis.

        **Version:** 2.0.0
        **Features:**
        - Triple AI strategist consensus
        - Monte Carlo simulation
        - Genetic algorithm optimization
        - Game theory analysis
        - Correlation stacking
        - Contrarian angle detection
        """
    }
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }

    /* Headers */
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 2px solid #1f77b4;
    }

    h2 {
        color: #ff7f0e;
        margin-top: 20px;
    }

    h3 {
        color: #2ca02c;
    }

    /* Metrics */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }

    /* Success/Error boxes */
    .success-box {
        padding: 15px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }

    .error-box {
        padding: 15px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        margin: 10px 0;
    }

    .warning-box {
        padding: 15px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }

    .info-box {
        padding: 15px;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 10px 0;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
        border: none;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* DataFrames */
    .dataframe {
        font-size: 12px;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: bold;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

class SessionStateManager:
    """
    OPTIMIZED: Centralized session state management

    IMPROVEMENTS:
    - Safe initialization
    - Type checking
    - Default values
    - Reset functionality
    """

    @staticmethod
    def initialize():
        """Initialize all session state variables with safe defaults"""

        # Core data
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

        # AI components
        if 'api_manager' not in st.session_state:
            st.session_state.api_manager = None

        if 'game_theory_ai' not in st.session_state:
            st.session_state.game_theory_ai = None

        if 'correlation_ai' not in st.session_state:
            st.session_state.correlation_ai = None

        if 'contrarian_ai' not in st.session_state:
            st.session_state.contrarian_ai = None

        # AI recommendations
        if 'ai_recommendations' not in st.session_state:
            st.session_state.ai_recommendations = {}

        if 'synthesis' not in st.session_state:
            st.session_state.synthesis = None

        # Optimization results
        if 'optimized_lineups' not in st.session_state:
            st.session_state.optimized_lineups = None

        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = {}

        if 'genetic_results' not in st.session_state:
            st.session_state.genetic_results = None

        # Configuration
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
                'ownership_focus': 'balanced',
                'stack_preference': 'moderate',
                'min_salary': 49000,
                'max_ownership_total': 150
            }

        # UI state
        if 'optimization_complete' not in st.session_state:
            st.session_state.optimization_complete = False

        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False

        if 'show_advanced' not in st.session_state:
            st.session_state.show_advanced = False

        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1

        # Performance tracking
        if 'execution_times' not in st.session_state:
            st.session_state.execution_times = {}

        if 'error_log' not in st.session_state:
            st.session_state.error_log = []

        # Export settings
        if 'export_format' not in st.session_state:
            st.session_state.export_format = 'csv'

    @staticmethod
    def reset_optimization():
        """Reset optimization-related state"""
        st.session_state.ai_recommendations = {}
        st.session_state.synthesis = None
        st.session_state.optimized_lineups = None
        st.session_state.simulation_results = {}
        st.session_state.genetic_results = None
        st.session_state.optimization_complete = False
        st.session_state.analysis_complete = False
        st.session_state.execution_times = {}

    @staticmethod
    def reset_all():
        """Complete reset of session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        SessionStateManager.initialize()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class UIHelpers:
    """
    OPTIMIZED: UI helper functions with error handling
    """

    @staticmethod
    def show_success(message: str):
        """Display success message"""
        st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def show_error(message: str):
        """Display error message"""
        st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def show_warning(message: str):
        """Display warning message"""
        st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def show_info(message: str):
        """Display info message"""
        st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)

    @staticmethod
    def create_metric_card(label: str, value: Any, delta: Optional[Any] = None):
        """Create a metric display card"""
        st.metric(label=label, value=value, delta=delta)

    @staticmethod
    def format_currency(value: float) -> str:
        """Format value as currency"""
        return f"${value:,.0f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format value as percentage"""
        return f"{value:.1f}%"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"

    @staticmethod
    def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except:
            return default

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame has required columns

        Returns:
            Tuple of (is_valid, missing_columns)
        """
        if df is None or df.empty:
            return False, required_columns

        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing


# ============================================================================
# FILE UPLOAD AND VALIDATION
# ============================================================================

class FileHandler:
    """
    OPTIMIZED: Secure file handling with validation

    IMPROVEMENTS:
    - Size limits
    - Format validation
    - Column validation
    - Safe parsing
    """

    REQUIRED_COLUMNS = [
        'Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership'
    ]

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    @staticmethod
    def validate_csv_file(uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded CSV file

        Returns:
            Tuple of (is_valid, message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"

        # Check file size
        file_size = uploaded_file.size
        if file_size > FileHandler.MAX_FILE_SIZE:
            return False, f"File too large: {file_size/1024/1024:.1f}MB (max 10MB)"

        # Check file extension
        if not uploaded_file.name.endswith('.csv'):
            return False, "File must be a CSV file"

        return True, "File validation passed"

    @staticmethod
    def load_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load and validate CSV file

        Returns:
            Tuple of (dataframe, message)
        """
        try:
            # Validate file first
            is_valid, message = FileHandler.validate_csv_file(uploaded_file)
            if not is_valid:
                return None, message

            # Try to load CSV
            df = pd.read_csv(uploaded_file)

            if df.empty:
                return None, "CSV file is empty"

            # Validate columns
            is_valid, missing_cols = UIHelpers.validate_dataframe(
                df, FileHandler.REQUIRED_COLUMNS
            )

            if not is_valid:
                return None, f"Missing required columns: {', '.join(missing_cols)}"

            # Validate data types and ranges
            validation_result = FileHandler.validate_data(df)
            if not validation_result['valid']:
                return None, f"Data validation failed: {validation_result['message']}"

            # Clean and prepare data
            df = FileHandler.clean_dataframe(df)

            return df, f"Successfully loaded {len(df)} players"

        except pd.errors.EmptyDataError:
            return None, "CSV file is empty or corrupted"
        except pd.errors.ParserError as e:
            return None, f"CSV parsing error: {str(e)}"
        except Exception as e:
            return None, f"Error loading file: {str(e)}"

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data types and ranges

        Returns:
            Dictionary with validation results
        """
        try:
            # Check salary range
            invalid_salaries = (
                (df['Salary'] < OptimizerConfig.MIN_SALARY) |
                (df['Salary'] > OptimizerConfig.MAX_SALARY * 1.5)
            )

            if invalid_salaries.any():
                return {
                    'valid': False,
                    'message': f"{invalid_salaries.sum()} players have invalid salaries"
                }

            # Check ownership range
            invalid_ownership = (df['Ownership'] < 0) | (df['Ownership'] > 100)
            if invalid_ownership.any():
                return {
                    'valid': False,
                    'message': f"{invalid_ownership.sum()} players have invalid ownership %"
                }

            # Check projected points
            invalid_projections = (df['Projected_Points'] < 0) | (df['Projected_Points'] > 100)
            if invalid_projections.any():
                return {
                    'valid': False,
                    'message': f"{invalid_projections.sum()} players have invalid projections"
                }

            # Check for required positions
            valid_positions = {'QB', 'RB', 'WR', 'TE', 'DST', 'K', 'FLEX'}
            invalid_positions = ~df['Position'].isin(valid_positions)

            if invalid_positions.any():
                unique_invalid = df.loc[invalid_positions, 'Position'].unique()
                return {
                    'valid': False,
                    'message': f"Invalid positions found: {', '.join(unique_invalid)}"
                }

            # Check for minimum players per position
            position_counts = df['Position'].value_counts()

            min_requirements = {'QB': 2, 'RB': 3, 'WR': 3, 'TE': 2}

            for pos, min_count in min_requirements.items():
                if position_counts.get(pos, 0) < min_count:
                    return {
                        'valid': False,
                        'message': f"Need at least {min_count} {pos} players (found {position_counts.get(pos, 0)})"
                    }

            return {'valid': True, 'message': 'All validations passed'}

        except Exception as e:
            return {'valid': False, 'message': f"Validation error: {str(e)}"}

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=['Player'], keep='first')

        # Strip whitespace from string columns
        string_cols = ['Player', 'Position', 'Team']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Ensure numeric columns
        numeric_cols = ['Salary', 'Projected_Points', 'Ownership']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill NaN ownership with default
        df['Ownership'] = df['Ownership'].fillna(10.0)

        # Remove rows with invalid data
        df = df.dropna(subset=['Player', 'Position', 'Salary', 'Projected_Points'])

        # Sort by projected points
        df = df.sort_values('Projected_Points', ascending=False).reset_index(drop=True)

        return df

    @staticmethod
    def create_sample_csv() -> pd.DataFrame:
        """
        Create sample CSV template

        Returns:
            Sample DataFrame
        """
        sample_data = {
            'Player': [
                'Patrick Mahomes', 'Travis Kelce', 'Tyreek Hill',
                'Derrick Henry', 'Justin Jefferson', 'Mark Andrews',
                'Josh Allen', 'Stefon Diggs', 'Christian McCaffrey',
                'Cooper Kupp', 'Austin Ekeler', 'Davante Adams'
            ],
            'Position': [
                'QB', 'TE', 'WR', 'RB', 'WR', 'TE',
                'QB', 'WR', 'RB', 'WR', 'RB', 'WR'
            ],
            'Team': [
                'KC', 'KC', 'MIA', 'TEN', 'MIN', 'BAL',
                'BUF', 'BUF', 'SF', 'LAR', 'LAC', 'LV'
            ],
            'Salary': [
                11000, 8500, 9000, 9500, 8800, 7500,
                10500, 8000, 10000, 8500, 8800, 8200
            ],
            'Projected_Points': [
                25.5, 18.2, 19.8, 21.3, 19.5, 16.8,
                24.8, 17.9, 22.7, 18.5, 20.1, 17.6
            ],
            'Ownership': [
                35.2, 28.5, 22.1, 18.7, 25.3, 15.9,
                32.8, 20.4, 30.1, 24.7, 19.3, 16.2
            ]
        }

        return pd.DataFrame(sample_data)


# ============================================================================
# DATA PREVIEW COMPONENT
# ============================================================================

def render_data_preview(df: pd.DataFrame):
    """
    Render data preview with statistics

    Args:
        df: Player DataFrame
    """
    st.subheader("📊 Data Preview & Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        UIHelpers.create_metric_card(
            "Total Players",
            len(df)
        )

    with col2:
        UIHelpers.create_metric_card(
            "Avg Salary",
            UIHelpers.format_currency(df['Salary'].mean())
        )

    with col3:
        UIHelpers.create_metric_card(
            "Avg Projection",
            f"{df['Projected_Points'].mean():.1f}"
        )

    with col4:
        UIHelpers.create_metric_card(
            "Avg Ownership",
            UIHelpers.format_percentage(df['Ownership'].mean())
        )

    # Position breakdown
    st.markdown("#### Position Breakdown")

    position_stats = df.groupby('Position').agg({
        'Player': 'count',
        'Salary': 'mean',
        'Projected_Points': 'mean',
        'Ownership': 'mean'
    }).round(1)

    position_stats.columns = ['Count', 'Avg Salary', 'Avg Proj', 'Avg Own%']
    position_stats['Avg Salary'] = position_stats['Avg Salary'].apply(
        lambda x: UIHelpers.format_currency(x)
    )

    st.dataframe(position_stats, use_container_width=True)

    # Top players by projection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 Projected")
        top_proj = df.nlargest(10, 'Projected_Points')[
            ['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']
        ].copy()
        top_proj['Salary'] = top_proj['Salary'].apply(UIHelpers.format_currency)
        top_proj.columns = ['Player', 'Pos', 'Salary', 'Proj', 'Own%']
        st.dataframe(top_proj, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Highest Value (Proj/Salary)")
        df_value = df.copy()
        df_value['Value'] = df_value['Projected_Points'] / (df_value['Salary'] / 1000)
        top_value = df_value.nlargest(10, 'Value')[
            ['Player', 'Position', 'Salary', 'Projected_Points', 'Value']
        ].copy()
        top_value['Salary'] = top_value['Salary'].apply(UIHelpers.format_currency)
        top_value['Value'] = top_value['Value'].round(2)
        top_value.columns = ['Player', 'Pos', 'Salary', 'Proj', 'Value']
        st.dataframe(top_value, hide_index=True, use_container_width=True)


# Initialize session state
SessionStateManager.initialize()

# ============================================================================
# PART 2: SIDEBAR CONFIGURATION & DATA UPLOAD
# ============================================================================

# ============================================================================
# SIDEBAR: API CONFIGURATION
# ============================================================================

def render_api_configuration():
    """
    Render API configuration section in sidebar
    """
    st.sidebar.header("🔑 API Configuration")

    use_api = st.sidebar.checkbox(
        "Use AI Analysis",
        value=st.session_state.config.get('use_api', True),
        help="Enable Claude AI for strategic analysis. Requires API key."
    )

    st.session_state.config['use_api'] = use_api

    if use_api:
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            value=st.session_state.config.get('api_key', ''),
            type="password",
            help="Enter your Anthropic API key (starts with 'sk-ant-')",
            placeholder="sk-ant-..."
        )

        st.session_state.config['api_key'] = api_key

        if api_key:
            if api_key.startswith('sk-ant-'):
                # Try to initialize API manager
                if st.session_state.api_manager is None:
                    try:
                        with st.spinner("Validating API key..."):
                            st.session_state.api_manager = ClaudeAPIManager(api_key)

                            # Initialize AI strategists
                            st.session_state.game_theory_ai = GPPGameTheoryStrategist(
                                st.session_state.api_manager
                            )
                            st.session_state.correlation_ai = GPPCorrelationStrategist(
                                st.session_state.api_manager
                            )
                            st.session_state.contrarian_ai = GPPContrarianNarrativeStrategist(
                                st.session_state.api_manager
                            )

                            UIHelpers.show_success("API key validated successfully!")
                    except Exception as e:
                        UIHelpers.show_error(f"API validation failed: {str(e)}")
                        st.session_state.api_manager = None
                else:
                    st.sidebar.success("✅ API connected")
            else:
                UIHelpers.show_warning("API key should start with 'sk-ant-'")

        if not api_key:
            UIHelpers.show_info(
                "AI analysis requires an Anthropic API key. "
                "Get one at: https://console.anthropic.com"
            )
    else:
        st.sidebar.info("Using statistical fallback mode (no API)")
        st.session_state.api_manager = None


# ============================================================================
# SIDEBAR: OPTIMIZATION SETTINGS
# ============================================================================

def render_optimization_settings():
    """
    Render optimization settings in sidebar
    """
    st.sidebar.header("⚙️ Optimization Settings")

    # Number of lineups
    num_lineups = st.sidebar.number_input(
        "Number of Lineups",
        min_value=1,
        max_value=150,
        value=st.session_state.config.get('num_lineups', 20),
        step=1,
        help="Number of unique lineups to generate"
    )
    st.session_state.config['num_lineups'] = num_lineups

    # Field size
    field_size_options = [
        FieldSize.SMALL.value,
        FieldSize.MEDIUM.value,
        FieldSize.LARGE.value,
        FieldSize.LARGE_AGGRESSIVE.value,
        FieldSize.MILLY_MAKER.value
    ]

    field_size = st.sidebar.selectbox(
        "Tournament Size",
        options=field_size_options,
        index=field_size_options.index(
            st.session_state.config.get('field_size', FieldSize.LARGE.value)
        ),
        help="Adjust strategy based on tournament size"
    )
    st.session_state.config['field_size'] = field_size

    # AI Enforcement Level
    if st.session_state.config.get('use_api', True):
        enforcement_options = [
            AIEnforcementLevel.MANDATORY.value,
            AIEnforcementLevel.STRONG.value,
            AIEnforcementLevel.MODERATE.value,
            AIEnforcementLevel.ADVISORY.value
        ]

        enforcement_level = st.sidebar.select_slider(
            "AI Enforcement Level",
            options=enforcement_options,
            value=st.session_state.config.get(
                'enforcement_level',
                AIEnforcementLevel.STRONG.value
            ),
            help=(
                "MANDATORY: All AI decisions enforced\n"
                "STRONG: High-confidence as hard constraints\n"
                "MODERATE: Consensus enforced, rest as soft\n"
                "ADVISORY: All as soft preferences"
            )
        )
        st.session_state.config['enforcement_level'] = enforcement_level

    # Advanced settings expander
    with st.sidebar.expander("🔧 Advanced Settings"):

        # Monte Carlo simulation
        run_simulation = st.checkbox(
            "Enable Monte Carlo Simulation",
            value=st.session_state.config.get('run_simulation', True),
            help="Run simulations to estimate lineup variance and ceiling"
        )
        st.session_state.config['run_simulation'] = run_simulation

        if run_simulation:
            num_simulations = st.slider(
                "Simulations per Lineup",
                min_value=1000,
                max_value=10000,
                value=st.session_state.config.get('num_simulations', 5000),
                step=1000,
                help="More simulations = more accurate but slower"
            )
            st.session_state.config['num_simulations'] = num_simulations

        # Genetic algorithm
        use_genetic = st.checkbox(
            "Use Genetic Algorithm",
            value=st.session_state.config.get('use_genetic', False),
            help="Evolutionary optimization (slower but can find unique lineups)"
        )
        st.session_state.config['use_genetic'] = use_genetic

        if use_genetic:
            genetic_generations = st.slider(
                "GA Generations",
                min_value=20,
                max_value=100,
                value=st.session_state.config.get('genetic_generations', 50),
                step=10,
                help="More generations = better optimization but slower"
            )
            st.session_state.config['genetic_generations'] = genetic_generations

        st.markdown("---")

        # Ownership focus
        ownership_focus = st.selectbox(
            "Ownership Focus",
            options=['chalk', 'balanced', 'leverage', 'contrarian'],
            index=['chalk', 'balanced', 'leverage', 'contrarian'].index(
                st.session_state.config.get('ownership_focus', 'balanced')
            ),
            help=(
                "chalk: Higher ownership, safer\n"
                "balanced: Mix of ownership levels\n"
                "leverage: Lower ownership plays\n"
                "contrarian: Maximum differentiation"
            )
        )
        st.session_state.config['ownership_focus'] = ownership_focus

        # Stack preference
        stack_preference = st.selectbox(
            "Stacking Preference",
            options=['minimal', 'moderate', 'aggressive'],
            index=['minimal', 'moderate', 'aggressive'].index(
                st.session_state.config.get('stack_preference', 'moderate')
            ),
            help="How aggressively to stack correlated players"
        )
        st.session_state.config['stack_preference'] = stack_preference

        st.markdown("---")

        # Salary constraints
        min_salary = st.slider(
            "Minimum Total Salary",
            min_value=45000,
            max_value=50000,
            value=st.session_state.config.get('min_salary', 49000),
            step=500,
            help="Minimum salary to use (higher = less punt plays)"
        )
        st.session_state.config['min_salary'] = min_salary

        # Ownership constraint
        max_ownership = st.slider(
            "Max Total Ownership %",
            min_value=100,
            max_value=250,
            value=st.session_state.config.get('max_ownership_total', 150),
            step=10,
            help="Maximum combined ownership percentage"
        )
        st.session_state.config['max_ownership_total'] = max_ownership


# ============================================================================
# SIDEBAR: ACTIONS
# ============================================================================

def render_sidebar_actions():
    """
    Render action buttons in sidebar
    """
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Actions")

    # Reset button
    if st.sidebar.button("🔄 Reset All", use_container_width=True):
        SessionStateManager.reset_all()
        st.rerun()

    # Download sample CSV
    if st.sidebar.button("📥 Download Sample CSV", use_container_width=True):
        sample_df = FileHandler.create_sample_csv()
        csv = sample_df.to_csv(index=False)

        st.sidebar.download_button(
            label="💾 Download Template",
            data=csv,
            file_name="nfl_dfs_template.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================================
# SIDEBAR: HELP & INFO
# ============================================================================

def render_sidebar_help():
    """
    Render help section in sidebar
    """
    st.sidebar.markdown("---")

    with st.sidebar.expander("❓ Help & Information"):
        st.markdown("""
        ### Quick Start Guide

        1. **Upload CSV**: Upload your player pool CSV
        2. **Configure Game**: Set game info (total, spread)
        3. **API Key**: Add Anthropic API key for AI analysis
        4. **Settings**: Adjust optimization parameters
        5. **Analyze**: Click "Run AI Analysis"
        6. **Optimize**: Click "Generate Lineups"
        7. **Export**: Download your lineups

        ### Required CSV Columns

        - `Player`: Player name
        - `Position`: QB, RB, WR, TE, etc.
        - `Team`: Team abbreviation
        - `Salary`: DraftKings salary
        - `Projected_Points`: Fantasy points projection
        - `Ownership`: Projected ownership %

        ### AI Strategists

        - **Game Theory**: Finds leverage plays and optimal captain selection
        - **Correlation**: Identifies profitable stacking combinations
        - **Contrarian**: Discovers unique winning angles

        ### Support

        - GitHub: [Report Issues](https://github.com/yourusername/nfl-dfs-optimizer)
        - Docs: [Full Documentation](#)
        """)


# ============================================================================
# MAIN: HEADER
# ============================================================================

def render_header():
    """
    Render main application header
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🏈 NFL DFS AI-Driven Optimizer")
        st.markdown(
            "**Advanced DraftKings Showdown Optimizer** powered by triple-AI strategic analysis"
        )

    with col2:
        st.markdown(f"**Version:** {__version__}")
        st.markdown(f"**Model:** Claude Sonnet 4")


# ============================================================================
# MAIN: DATA UPLOAD
# ============================================================================

def render_data_upload():
    """
    Render data upload section
    """
    st.header("📂 Step 1: Upload Player Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload DraftKings CSV Export",
            type=['csv'],
            help="Upload CSV with player data including salary, projections, and ownership",
            key="player_csv_upload"
        )

        if uploaded_file is not None:
            with st.spinner("Loading and validating data..."):
                df, message = FileHandler.load_csv(uploaded_file)

                if df is not None:
                    st.session_state.player_df = df
                    UIHelpers.show_success(message)
                    st.session_state.current_step = 2
                else:
                    UIHelpers.show_error(message)
                    st.session_state.player_df = None

    with col2:
        st.markdown("### 📋 CSV Requirements")
        st.markdown("""
        **Required Columns:**
        - Player
        - Position
        - Team
        - Salary
        - Projected_Points
        - Ownership

        **Format:**
        - Salary: 3000-11500
        - Ownership: 0-100
        - Points: 0-50
        """)

        # Sample CSV download
        sample_df = FileHandler.create_sample_csv()
        csv = sample_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="nfl_dfs_sample.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Show data preview if loaded
    if st.session_state.player_df is not None:
        st.markdown("---")
        render_data_preview(st.session_state.player_df)


# ============================================================================
# MAIN: GAME INFORMATION
# ============================================================================

def render_game_information():
    """
    Render game information configuration
    """
    if st.session_state.player_df is None:
        return

    st.header("⚙️ Step 2: Configure Game Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        teams = st.text_input(
            "Matchup",
            value=st.session_state.game_info.get('teams', 'Team A vs Team B'),
            placeholder="Chiefs vs Bills",
            help="Enter the teams playing (e.g., 'Chiefs vs Bills')"
        )
        st.session_state.game_info['teams'] = teams

        game_total = st.number_input(
            "Game Total",
            min_value=30.0,
            max_value=65.0,
            value=st.session_state.game_info.get('total', 45.0),
            step=0.5,
            help="Over/Under total for the game"
        )
        st.session_state.game_info['total'] = game_total

    with col2:
        spread = st.number_input(
            "Spread",
            min_value=-21.0,
            max_value=21.0,
            value=st.session_state.game_info.get('spread', 0.0),
            step=0.5,
            help="Point spread (negative = favorite)"
        )
        st.session_state.game_info['spread'] = spread

        weather = st.selectbox(
            "Weather",
            options=['Clear', 'Dome', 'Light Rain', 'Heavy Rain', 'Snow', 'Wind'],
            index=['Clear', 'Dome', 'Light Rain', 'Heavy Rain', 'Snow', 'Wind'].index(
                st.session_state.game_info.get('weather', 'Clear')
            ),
            help="Weather conditions"
        )
        st.session_state.game_info['weather'] = weather

    with col3:
        primetime = st.checkbox(
            "Primetime Game",
            value=st.session_state.game_info.get('primetime', False),
            help="Is this a primetime/nationally televised game?"
        )
        st.session_state.game_info['primetime'] = primetime

        injury_count = st.number_input(
            "Notable Injuries",
            min_value=0,
            max_value=10,
            value=st.session_state.game_info.get('injury_count', 0),
            step=1,
            help="Number of significant injuries affecting this game"
        )
        st.session_state.game_info['injury_count'] = injury_count

    # Game environment summary
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pace = "Fast" if game_total > 48 else "Slow" if game_total < 42 else "Average"
        st.metric("Pace", pace, help="Based on game total")

    with col2:
        game_script = "Competitive" if abs(spread) < 3 else "Blowout Risk" if abs(spread) > 7 else "Moderate"
        st.metric("Game Script", game_script, help="Based on spread")

    with col3:
        conditions = "Ideal" if weather in ['Clear', 'Dome'] else "Challenging"
        st.metric("Conditions", conditions, help="Based on weather")

    with col4:
        if primetime:
            st.metric("Spotlight", "High", help="Primetime game")
        else:
            st.metric("Spotlight", "Normal", help="Regular slot")

    # Advanced game info expander
    with st.expander("🔍 Advanced Game Context"):
        st.markdown("### Game Theory Implications")

        # Calculate implied total based on spread and total
        favorite_implied = (game_total + abs(spread)) / 2
        underdog_implied = (game_total - abs(spread)) / 2

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Scoring Environment**")
            if game_total > 50:
                st.markdown("🔥 **Shootout Expected** - High ceiling plays favored")
            elif game_total < 40:
                st.markdown("🐌 **Low Scoring** - Target TD-dependent plays")
            else:
                st.markdown("⚖️ **Average Total** - Balanced approach")

            st.markdown(f"- Favorite Implied: {favorite_implied:.1f}")
            st.markdown(f"- Underdog Implied: {underdog_implied:.1f}")

        with col2:
            st.markdown("**Strategic Angles**")
            if abs(spread) > 10:
                st.markdown("⚠️ **Blowout Risk** - Consider game script pivots")
            elif abs(spread) < 3:
                st.markdown("🎯 **Close Game** - Both teams in play throughout")
            else:
                st.markdown("📊 **Moderate Spread** - Favorite slightly favored")

            if weather not in ['Clear', 'Dome']:
                st.markdown(f"🌧️ **Weather Impact** - {weather} may reduce passing")

    if st.session_state.player_df is not None:
        st.session_state.current_step = 3


# ============================================================================
# MAIN: STEP PROGRESS INDICATOR
# ============================================================================

def render_progress_indicator():
    """
    Render step progress indicator
    """
    steps = [
        ("📂", "Upload Data"),
        ("⚙️", "Configure Game"),
        ("🤖", "AI Analysis"),
        ("🎯", "Optimize"),
        ("📊", "Review & Export")
    ]

    current_step = st.session_state.current_step

    cols = st.columns(len(steps))

    for idx, (col, (icon, label)) in enumerate(zip(cols, steps), 1):
        with col:
            if idx < current_step:
                st.markdown(
                    f"<div style='text-align: center; color: green;'>"
                    f"<h2>{icon}</h2>"
                    f"<p style='font-size: 12px;'><b>✓ {label}</b></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif idx == current_step:
                st.markdown(
                    f"<div style='text-align: center; color: #1f77b4;'>"
                    f"<h2>{icon}</h2>"
                    f"<p style='font-size: 12px;'><b>▶ {label}</b></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align: center; color: gray;'>"
                    f"<h2>{icon}</h2>"
                    f"<p style='font-size: 12px;'>{label}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ============================================================================
# MAIN APP RENDERING FUNCTION
# ============================================================================

def render_main_app():
    """
    Main app rendering orchestrator
    """
    # Render sidebar
    render_api_configuration()
    render_optimization_settings()
    render_sidebar_actions()
    render_sidebar_help()

    # Render main content
    render_header()

    st.markdown("---")

    # Progress indicator
    render_progress_indicator()

    st.markdown("---")

    # Main workflow
    render_data_upload()

    if st.session_state.player_df is not None:
        st.markdown("---")
        render_game_information()
