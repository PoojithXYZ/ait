import streamlit as st
st.title("Discover Strengths of MCTS Variants")


st.header("Introduction")
intro = """
Monte Carlo Tree Search (MCTS) is a well-established algorithm for decision-making in complex games.\
A lot of research on MCTS has produce wide varieties of variants all with strengths and weaknesses.\
 It becomes hard to estimate which algorithm is the best choice for a given game.\
 By using various features of game states, data from previously played matches and the other attributes\
 representing the features and states of MCTS variants, our model is trained to predict the probability\
 of a variant outperforming another across various board game. The Luddite framework is used to describe\
 thousands of board games with various types of environments that the MCTS agents will operate in.
"""
st.write(intro)


st.header("Architecture")
architecture = """
"""
st.image("src/pages/architecture_diagram.png", caption="Architecture of the app", use_container_width=True)


st.header("How to use this app")
how_to_use = """
Utiliation of this app is simple.
1. Upload the information about the features of the two agents and the board game in CSV (comma separated values) format.
2. Click on the "Compare" button.
3. The app will utilize the machine learning models to predict the probability of Agent 1 winning against Agent 2 out of 10 games.
4. The predictions will be displayed below as a DataFrame. The same can be downloaded as a CSV file.
"""
st.write(how_to_use)
