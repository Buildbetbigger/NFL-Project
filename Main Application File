import streamlit as st
import pandas as pd
from datetime import datetime

# Import your optimizer (assuming the main file is named optimizer.py)
from your_main_file import ShowdownOptimizer, AIEnforcementLevel

st.set_page_config(page_title="NFL Showdown Optimizer", layout="wide")

st.title("🏈 NFL Showdown Optimizer")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("Claude API Key (optional)", type="password")
    
    # Upload CSV
    uploaded_file = st.file_uploader("Upload Player Projections CSV", type=['csv'])
    
    # Game info
    st.subheader("Game Information")
    teams = st.text_input("Teams", "Team1 vs Team2")
    total = st.number_input("Game Total", value=47.5, step=0.5)
    spread = st.number_input("Spread", value=-3.5, step=0.5)
    
    # Optimization settings
    st.subheader("Optimization Settings")
    num_lineups = st.slider("Number of Lineups", 5, 150, 20)
    field_size = st.selectbox("Field Size", 
        ['small_field', 'medium_field', 'large_field', 
         'large_field_aggressive', 'milly_maker'])
    
    use_genetic = st.checkbox("Use Genetic Algorithm", value=False)
    use_simulation = st.checkbox("Use Monte Carlo Simulation", value=True)
    
    optimize_button = st.button("🚀 Optimize Lineups", type="primary")

# Main content area
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Show preview
    st.subheader("📊 Player Pool Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Optimize when button clicked
    if optimize_button:
        with st.spinner("Optimizing lineups..."):
            try:
                # Create game info dict
                game_info = {
                    'teams': teams,
                    'total': total,
                    'spread': spread,
                    'weather': 'Clear'
                }
                
                # Initialize optimizer
                optimizer = ShowdownOptimizer(api_key=api_key if api_key else None)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                # Run optimization
                lineups = optimizer.optimize(
                    df=df,
                    game_info=game_info,
                    num_lineups=num_lineups,
                    field_size=field_size,
                    use_api=(api_key is not None and api_key != ""),
                    use_genetic=use_genetic,
                    use_simulation=use_simulation,
                    progress_callback=update_progress
                )
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                if not lineups.empty:
                    st.success(f"✅ Generated {len(lineups)} lineups!")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Lineups", len(lineups))
                    col2.metric("Avg Projection", f"{lineups['Projected'].mean():.2f}")
                    col3.metric("Avg Ownership", f"{lineups['Total_Own'].mean():.1f}%")
                    col4.metric("Unique Captains", lineups['CPT'].nunique())
                    
                    # Show lineups
                    st.subheader("📋 Generated Lineups")
                    st.dataframe(lineups, use_container_width=True)
                    
                    # Download button
                    csv = lineups.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Lineups CSV",
                        data=csv,
                        file_name=f"showdown_lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                else:
                    st.error("❌ No lineups generated. Check your settings.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("See error details"):
                    st.exception(e)
else:
    st.info("👆 Upload a CSV file to get started")
