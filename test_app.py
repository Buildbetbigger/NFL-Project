import streamlit as st

st.title("Dependency Test")

# Test each import separately
st.header("Testing Imports...")

try:
    import pandas
    st.success(f"✅ pandas version {pandas.__version__}")
except Exception as e:
    st.error(f"❌ pandas failed: {e}")

try:
    import numpy
    st.success(f"✅ numpy version {numpy.__version__}")
except Exception as e:
    st.error(f"❌ numpy failed: {e}")

try:
    import pulp
    st.success(f"✅ PuLP version {pulp.__version__}")
    
    # Test solver availability
    st.write("Available solvers:", pulp.listSolvers(onlyAvailable=True))
    
    # Test simple problem
    prob = pulp.LpProblem("test", pulp.LpMaximize)
    x = pulp.LpVariable("x", 0, 4)
    prob += x
    prob += x <= 3
    prob.solve()
    st.success(f"✅ Solver test passed! x = {x.varValue}")
    
except Exception as e:
    st.error(f"❌ PuLP failed: {e}")

try:
    import matplotlib
    st.success(f"✅ matplotlib version {matplotlib.__version__}")
except Exception as e:
    st.error(f"❌ matplotlib failed: {e}")

try:
    import scipy
    st.success(f"✅ scipy version {scipy.__version__}")
except Exception as e:
    st.error(f"❌ scipy failed: {e}")

try:
    import anthropic
    st.success(f"✅ anthropic installed")
except Exception as e:
    st.warning(f"⚠️ anthropic not installed (optional): {e}")

st.header("System Info")
import sys
st.write(f"Python version: {sys.version}")
st.write(f"Python executable: {sys.executable}")

import os
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')}")

# Check if requirements.txt exists
if os.path.exists('requirements.txt'):
    st.success("✅ requirements.txt found")
    with open('requirements.txt', 'r') as f:
        st.code(f.read())
else:
    st.error("❌ requirements.txt NOT FOUND!")
