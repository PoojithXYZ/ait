import streamlit as st
import pandas as pd

st.title("Dataset")

st.header("Core Attributes : ")

attr = """
**Id** - (integer) A unique (within this file) ID for this row of data. The test data also has an **Id** column, which (like the one for training data) also starts counting at **0**, but these are unrelated to each other. You should probably drop this column.

**GameRulesetName** - (string) A combination of the game name and ruleset name in Ludii. Within the Ludii system, there is a distinction between games (cultural artifacts) and rulesets (the same game might be played according to different rulesets). For the purposes of this competition, think of every unique combination of a game + a ruleset as a separate game, although some games (ones that have many different rulesets) might be considered overrepresented in the training data.

**agent[1/2]** - (string) A string description of the agent that played as the [first/second] player. See the section on Agent String Descriptions below for more info.

**.....** - Most remaining columns describe properties of the game + ruleset played in this row. These range from abstract features (is the game deterministic or stochastic?) to specific features (does the game use a star-shaped board?), and from features that are about how the rules are described (e.g., do any of the rules involve a greater-than comparison?) to features that are about the behavior of the game in practice (e.g., number of times per second we can run a completely random play-out from initial game state till the end on our hardware). For more details see the [Concepts page on the Ludii website](https://ludii.games/concepts.php), and the [publication about them](https://arxiv.org/abs/2009.11278), or [concepts.csv](https://github.com/Ludii/Ludii/blob/master/data/concepts.csv).

**EnglishRules** - (string) An natural language (English) description of the rules of the game. This description is not guaranteed to be self-contained (e.g., it might refer to rules from famous other games, such as Chess, for brevity), unambiguous, or perfectly complete.

**LudRules** - (string) The description of the game in Ludii's game description language. This is the description that was used to compile the game inside Ludii and run the simulations, so it is always guaranteed to be 100% complete and unambiguous. However, this is a formal language that most existing Large Language Models / foundation models have likely received little, if any, exposure to.

**num_[wins/draws/losses]_agent1** - (int) The number of times the first agent [won/drew/lost] against the second agent in this game and this specific matchup of two agents.

**utility_agent1** - (float) The target column. The utility value that the first agent received, aggregated over all simulations we ran for this specific pair of agents in this game. This value will be between **-1** (if the first agent lost every single game) and **1** (if the first agent won every single game). Utility is calculated as (n_games_won - n_games_lost) / n_games.
"""
st.write(attr)


st.header("Ludii Concepts : ")

st.dataframe(pd.read_csv("data/concepts.csv"), use_container_width=True)
