import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pulp

st.set_page_config(page_title="NFL Showdown Optimizer", layout="wide")

st.title("NFL Showdown Optimizer")

# Simple optimizer function (no AI, just linear programming)
def optimize_lineups(df, num_lineups=20):
    """Generate lineups using linear programming"""
    lineups = []
    
    for i in range(num_lineups):
        prob = pulp.LpProblem(f"Showdown_{i}", pulp.LpMaximize)
        
        # Variables
        player_vars = {p: pulp.LpVariable(f"p_{j}_{i}", cat='Binary') 
                      for j, p in enumerate(df['Player'])}
        captain_vars = {p: pulp.LpVariable(f"c_{j}_{i}", cat='Binary') 
                       for j, p in enumerate(df['Player'])}
        
        # Objective: maximize projections with randomness
        proj_dict = dict(zip(df['Player'], df['Projected_Points'] * 
                            (1 + np.random.uniform(-0.1, 0.1, len(df)))))
        
        prob += pulp.lpSum([
            captain_vars[p] * proj_dict[p] * 1.5 + player_vars[p] * proj_dict[p]
            for p in player_vars
        ])
        
        # Constraints
        prob += pulp.lpSum(captain_vars.values()) == 1  # 1 captain
        prob += pulp.lpSum(player_vars.values()) == 5   # 5 flex
        
        for p in player_vars:
            prob += captain_vars[p] + player_vars[p] <= 1  # No overlap
        
        # Salary cap
        salary_dict = dict(zip(df['Player'], df['Salary']))
        prob += pulp.lpSum([
            captain_vars[p] * salary_dict[p] * 1.5 + player_vars[p] * salary_dict[p]
            for p in player_vars
        ]) <= 50000
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            captain = next((p for p in captain_vars if captain_vars[p].varValue > 0.5), None)
            flex = [p for p in player_vars if player_vars[p].varValue > 0.5]
            
            if captain and len(flex) == 5:
                lineups.append({
                    'Lineup': i + 1,
                    'CPT': captain,
                    'FLEX1': flex[0],
                    'FLEX2': flex[1],
                    'FLEX3': flex[2],
                    'FLEX4': flex[3],
                    'FLEX5': flex[4]
                })
    
    return pd.DataFrame(lineups)

# UI
uploaded_file = st.file_uploader("Upload Player Projections CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Player Pool")
    st.dataframe(df.head(10))
    
    num_lineups = st.slider("Number of Lineups", 5, 150, 20)
    
    if st.button("Generate Lineups", type="primary"):
        with st.spinner("Optimizing..."):
            lineups = optimize_lineups(df, num_lineups)
            
            st.success(f"Generated {len(lineups)} lineups")
            st.dataframe(lineups)
            
            csv = lineups.to_csv(index=False)
            st.download_button("Download CSV", csv, 
                             f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
