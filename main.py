import streamlit as st
import pandas as pd
import networkx as nx
import os


st.set_page_config(page_title="Haystack Dating", layout="wide")
st.title("Haystack Dating - Optimized Matchmaking")

st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])
start_button = st.sidebar.button("Start Analysis")


def load_data(file):
    if file is not None:
        if file.name.endswith("xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    else:
        st.error("Please upload a file!")
        return None
    # Rename problematic columns.
    df.rename(columns={'Age (number format please)': 'Age'}, inplace=True)
    df.rename(columns={'Which best describes the industry/role you work in?': 'Industry'}, inplace=True)

    # Convert the ticket reference column to string if it exists.
    ticket_col = "Please enter your Fatsoma ticket reference number"
    if ticket_col in df.columns:
        df[ticket_col] = df[ticket_col].astype(str)

    # Parse 'Dating situation' into 'Gender' and 'Seeking'.
    def parse_dating_situation(ds):
        if pd.isna(ds):
            return ("Unknown", "Unknown")
        ds_lower = ds.lower().strip()
        # Mapping for common gender terms.
        gender_map = {"women": "Woman", "woman": "Woman", "men": "Man", "man": "Man"}

        # Helper: Remove sexual preference keywords from a string.
        pref_terms = ["bisexual", "pansexual", "demisexual", "homosexual"]

        def remove_preferences(s):
            for term in pref_terms:
                s = s.replace(term, "")
            return s.strip()

        # Try "dating" format first.
        if "dating" in ds_lower:
            parts = ds_lower.split("dating")
            if len(parts) == 2:
                gender_self_raw = remove_preferences(parts[0].strip())
                gender_seeking_raw = parts[1].strip()
                # Map common terms.
                gender_self = gender_map.get(gender_self_raw,
                                             gender_self_raw.capitalize() if gender_self_raw else "Unknown")
                if gender_seeking_raw.startswith("m"):
                    gender_seeking = "Man"
                elif gender_seeking_raw.startswith("w"):
                    gender_seeking = "Woman"
                else:
                    gender_seeking = gender_seeking_raw.capitalize()
                return (gender_self, gender_seeking)
        # Next try "seeking" format.
        if "seeking" in ds_lower:
            parts = ds_lower.split("seeking")
            if len(parts) == 2:
                gender_self_raw = remove_preferences(parts[0].strip())
                gender_seeking_raw = parts[1].strip()
                gender_self = gender_map.get(gender_self_raw,
                                             gender_self_raw.capitalize() if gender_self_raw else "Unknown")
                if gender_seeking_raw.startswith("m"):
                    gender_seeking = "Man"
                elif gender_seeking_raw.startswith("w"):
                    gender_seeking = "Woman"
                else:
                    gender_seeking = gender_seeking_raw.capitalize()
                return (gender_self, gender_seeking)
        # If any preference term exists but no explicit format is found, assume unknown gender but open seeking.
        if any(term in ds_lower for term in pref_terms):
            return ("Unknown", "Man, Woman")
        return ("Unknown", "Unknown")

    parsed = df["Dating situation"].apply(parse_dating_situation)
    df["Gender"] = parsed.apply(lambda x: x[0])
    df["Seeking"] = parsed.apply(lambda x: x[1])

    # If Seeking is a list, convert to comma-separated string (here we already return string for preference case).
    return df


def is_compatible(row1, row2):
    """
    Two candidates are reciprocally compatible if candidate A's Gender equals candidate B's Seeking
    and candidate B's Gender equals candidate A's Seeking.
    """
    return (row1["Gender"] == row2["Seeking"]) and (row2["Gender"] == row1["Seeking"])


def compute_match_score(row1, row2):
    """
    Compute a match score based on age difference, industry, and personality.
    - Penalize 2 points per year of age difference.
    - Add a bonus of 10 if industries match.
    - For each exact match in bonus columns, add 5 points.

    Returns:
        total_score, age_diff, industry_bonus, personality_bonus
    """
    age_diff = abs(row1["Age"] - row2["Age"])
    industry_bonus = 10 if str(row1["Industry"]).strip().lower() == str(row2["Industry"]).strip().lower() else 0
    base_score = 100 - (2 * age_diff)

    personality_bonus = 0
    bonus_columns = [
        "Which best describes your relationship with exercise?",
        "Congrats, you won a 3 weeks annual leave and the postcode lottery! Where are you heading?",
        "How important would you say career is to you?",
        "It’s New Years Eve party at 10pm. Where are you?",
        "Who would you most like share a cheeseboard with?",
        "What’s your dating vibe at the moment?",
        "Your wife/husband (that you found at a Haystack Dating event x) comes home one day with 6 sausage dogs, in matching trousers. What’s your reaction?",
        "It’s your Nan's birthday. What’s your move?"
    ]
    for col in bonus_columns:
        val1 = str(row1[col]).strip().lower()
        val2 = str(row2[col]).strip().lower()
        if val1 == val2:
            personality_bonus += 5

    total_score = base_score + industry_bonus + personality_bonus
    return total_score, age_diff, industry_bonus, personality_bonus


def perform_matching(df):
    """
    Build a graph over eligible candidates (those whose Seeking is "Man" or "Woman").
    Add an edge only if they are reciprocally compatible based on Gender and Seeking.
    Then compute a maximum weighted matching (disjoint pairs) and add additional best edges
    for any unmatched candidate (while trying to minimize overloading any one candidate).

    Returns a list of match dictionaries sorted by descending score.
    """

    def is_valid(row):
        return row["Seeking"] in {"Man", "Woman"}

    eligible_df = df[df.apply(is_valid, axis=1)].copy()
    if eligible_df.empty:
        return []

    G = nx.Graph()
    indices = list(eligible_df.index)
    for idx in indices:
        G.add_node(idx)
    for i in indices:
        for j in indices:
            if i < j:
                if is_compatible(eligible_df.loc[i], eligible_df.loc[j]):
                    score, age_diff, ind_bonus, pers_bonus = compute_match_score(eligible_df.loc[i], eligible_df.loc[j])
                    G.add_edge(i, j, weight=score, age_diff=age_diff, industry_bonus=ind_bonus,
                               personality_bonus=pers_bonus)

    # Compute maximum weighted matching (disjoint pairs).
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    matching_tuples = set(tuple(sorted(list(edge))) for edge in matching)

    # Identify unmatched candidates.
    matched_nodes = set()
    for edge in matching_tuples:
        matched_nodes.update(edge)
    unmatched = set(indices) - matched_nodes

    # For each unmatched candidate, add their best available edge (if available) while preferring edges that don't increase degrees too much.
    max_degree = 2
    for u in unmatched:
        candidate_edges = []
        for v in G.neighbors(u):
            candidate_edge = tuple(sorted([u, v]))
            if candidate_edge not in matching_tuples:
                deg_u = sum(1 for edge in matching_tuples if u in edge)
                deg_v = sum(1 for edge in matching_tuples if v in edge)
                candidate_edges.append((candidate_edge, G[u][v]['weight'], deg_u + deg_v))
        if candidate_edges:
            candidate_edges.sort(key=lambda x: (x[2], -x[1]))
            best_edge = candidate_edges[0][0]
            u_degree = sum(1 for edge in matching_tuples if u in edge)
            v = best_edge[1] if best_edge[0] == u else best_edge[0]
            v_degree = sum(1 for edge in matching_tuples if v in edge)
            if u_degree < max_degree or v_degree < max_degree:
                matching_tuples.add(best_edge)

    matches_list = []
    for edge in matching_tuples:
        i, j = edge
        score = G[i][j]['weight']
        age_diff = G[i][j]['age_diff']
        ind_bonus = G[i][j]['industry_bonus']
        pers_bonus = G[i][j]['personality_bonus']
        candidate1_full = (f"{eligible_df.loc[i, 'First Name']} {eligible_df.loc[i, 'Last Name']} "
                           f"(Age: {eligible_df.loc[i, 'Age']}, Industry: {eligible_df.loc[i, 'Industry']}, "
                           f"Gender: {eligible_df.loc[i, 'Gender']}, Seeking: {eligible_df.loc[i, 'Seeking']})")
        candidate2_full = (f"{eligible_df.loc[j, 'First Name']} {eligible_df.loc[j, 'Last Name']} "
                           f"(Age: {eligible_df.loc[j, 'Age']}, Industry: {eligible_df.loc[j, 'Industry']}, "
                           f"Gender: {eligible_df.loc[j, 'Gender']}, Seeking: {eligible_df.loc[j, 'Seeking']})")
        candidate1_name = f"{eligible_df.loc[i, 'First Name']} {eligible_df.loc[i, 'Last Name']}"
        candidate2_name = f"{eligible_df.loc[j, 'First Name']} {eligible_df.loc[j, 'Last Name']}"
        pair_str = f"{candidate1_full}  <-->  {candidate2_full}"
        matches_list.append({
            "pair": pair_str,
            "score": score,
            "age_diff": age_diff,
            "industry_bonus": ind_bonus,
            "personality_bonus": pers_bonus,
            "candidate1": candidate1_name,
            "candidate2": candidate2_name
        })
    final_matches_sorted = sorted(matches_list, key=lambda x: x["score"], reverse=True)
    return final_matches_sorted


if start_button:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("Input Data")
        st.dataframe(df)

        st.subheader("Best Matches")
        final_matches = perform_matching(df)
        if final_matches:
            for match in final_matches:
                st.write(
                    f"{match['pair']} | Score: {match['score']} "
                    f"(Age diff: {match['age_diff']}, Industry bonus: {match['industry_bonus']}, Personality bonus: {match['personality_bonus']})"
                )
        else:
            st.write("No eligible matches found. Please check your data for proper 'Dating situation' values.")

        # --- Total Matches Section ---
        st.subheader("Total Matches")
        candidate_stats = {}
        for match in final_matches:
            for candidate in [match["candidate1"], match["candidate2"]]:
                if candidate not in candidate_stats:
                    candidate_stats[candidate] = {"count": 0, "highest_score": 0}
                candidate_stats[candidate]["count"] += 1
                if match["score"] > candidate_stats[candidate]["highest_score"]:
                    candidate_stats[candidate]["highest_score"] = match["score"]
        total_matches_df = pd.DataFrame([
            {"Candidate": cand, "Total Matches": stats["count"], "Highest Score": stats["highest_score"]}
            for cand, stats in candidate_stats.items()
        ])
        total_matches_df.sort_values(by="Highest Score", ascending=False, inplace=True)
        st.dataframe(total_matches_df)