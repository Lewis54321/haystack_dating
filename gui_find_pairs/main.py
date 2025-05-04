import streamlit as st
import pandas as pd
import networkx as nx
import os

import config


# --- Helper Functions ---

def parse_dating_situation(ds):
    """
    Parses the 'Dating situation' string to determine Gender and Seeking preference.
    Prioritizes formats like 'X dating Y' or 'X seeking Y'.
    Handles basic preference terms to assign broad seeking category if specific format fails.
    :param ds: The string from the 'Dating situation' column.
    :return: tuple: (Gender (str), Seeking (str)) - e.g., ("Woman", "Man") or ("Unknown", "Unknown")
               If preference terms like bisexual are present but format is unclear,
               returns ("Unknown", "Man, Woman") as a placeholder.
    """

    if pd.isna(ds):
        return ("Unknown", "Unknown")

    ds_lower = str(ds).lower().strip()

    # Helper to remove preference terms (bisexual, etc.) to isolate gender terms
    def remove_preferences(s):
        for term in config.PREFERENCE_TERMS:
            s = s.replace(term, "")
        return s.strip()

    # Try "dating" format
    if "dating" in ds_lower:
        parts = ds_lower.split("dating", 1)  # Split only once
        if len(parts) == 2:
            gender_self_raw = remove_preferences(parts[0].strip())
            gender_seeking_raw = parts[1].strip()

            gender_self = config.GENDER_MAP.get(gender_self_raw, gender_self_raw.capitalize() if gender_self_raw else "Unknown")

            # Determine seeking based on the start of the second part
            if gender_seeking_raw.startswith("m"):
                gender_seeking = "Man"
            elif gender_seeking_raw.startswith("w"):
                gender_seeking = "Woman"
            else:  # Handle cases like '... dating men and women' or unclear terms
                # If both common terms appear, assume broad seeking
                if "man" in gender_seeking_raw and "woman" in gender_seeking_raw:
                     gender_seeking = "Man, Woman"
                else: # Otherwise, capitalize what's there or mark Unknown
                    gender_seeking = gender_seeking_raw.capitalize() if gender_seeking_raw else "Unknown"

            return (gender_self, gender_seeking)

    # Try "seeking" format
    if "seeking" in ds_lower:
        parts = ds_lower.split("seeking", 1) # Split only once
        if len(parts) == 2:
            gender_self_raw = remove_preferences(parts[0].strip())
            gender_seeking_raw = parts[1].strip()

            gender_self = config.GENDER_MAP.get(gender_self_raw, gender_self_raw.capitalize() if gender_self_raw else "Unknown")

            if gender_seeking_raw.startswith("m"):
                gender_seeking = "Man"
            elif gender_seeking_raw.startswith("w"):
                gender_seeking = "Woman"
            else: # Handle cases like '... seeking men and women' or unclear terms
                if "man" in gender_seeking_raw and "woman" in gender_seeking_raw:
                    gender_seeking = "Man, Woman"
                else:
                    gender_seeking = gender_seeking_raw.capitalize() if gender_seeking_raw else "Unknown"

            return (gender_self, gender_seeking)

    # If any preference term exists but no explicit format is found, assume broad seeking.
    if any(term in ds_lower for term in config.PREFERENCE_TERMS):
        # Attempt to identify self-gender even without 'dating'/'seeking'
        gender_self_raw = remove_preferences(ds_lower)
        gender_self = config.GENDER_MAP.get(gender_self_raw, gender_self_raw.capitalize() if gender_self_raw else "Unknown")
        return (gender_self, "Man, Woman") # Assume seeking both if preference term is present but format unclear

    # Default fallback
    return ("Unknown", "Unknown")


def load_data(file):
    """
    Loads data from an uploaded Excel or CSV file.
    Renames columns, validates essential columns, parses 'Dating situation',
    and handles potential errors.

    Args:
        file: The uploaded file object from Streamlit.

    Returns:
        pd.DataFrame or None: The processed DataFrame or None if loading fails.
    """
    if file is None:
        st.error("❌ Please upload a file!")
        return None

    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            st.error("❌ Unsupported file type. Please upload .xlsx or .csv")
            return None
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None

    # --- Column Renaming and Validation ---
    # Rename columns for easier access
    rename_map = {}
    if config.COL_AGE in df.columns:
        rename_map[config.COL_AGE] = config.COL_AGE_RENAMED
    if config.COL_INDUSTRY in df.columns:
        rename_map[config.COL_INDUSTRY] = config.COL_INDUSTRY_RENAMED
    df.rename(columns=rename_map, inplace=True)

    # Check for essential columns after potential renaming
    essential_cols = [config.COL_FIRST_NAME, config.COL_LAST_NAME, config.COL_AGE_RENAMED, config.COL_DATING_SITUATION]
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Essential columns missing from the file: {', '.join(missing_cols)}")
        return None

    # Ensure Age is numeric, attempting conversion
    try:
        df[config.COL_AGE_RENAMED] = pd.to_numeric(df[config.COL_AGE_RENAMED], errors='coerce')
        if df[config.COL_AGE_RENAMED].isnull().any():
            st.warning("⚠️ Some entries in the 'Age' column could not be converted to numbers and were set to NaN. These participants might be excluded from matching.")
            # Optionally handle NaNs further, e.g., drop rows or impute
            # df.dropna(subset=[COL_AGE_RENAMED], inplace=True)
    except Exception as e:
        st.error(f"❌ Error processing the '{config.COL_AGE_RENAMED}' column. Ensure it contains only numbers. Error: {e}")
        return None

    # Convert the ticket reference column to string if it exists
    if config.COL_TICKET_REF in df.columns:
        df[config.COL_TICKET_REF] = df[config.COL_TICKET_REF].astype(str)

    # --- Parse 'Dating situation' ---
    try:
        parsed_dating = df[config.COL_DATING_SITUATION].apply(parse_dating_situation)
        df[config.COL_GENDER] = parsed_dating.apply(lambda x: x[0])
        df[config.COL_SEEKING] = parsed_dating.apply(lambda x: x[1])
    except KeyError:
        st.error(f"❌ Column '{config.COL_DATING_SITUATION}' not found. Cannot determine gender/seeking preferences.")
        return None
    except Exception as e:
        st.error(f"❌ Error parsing the '{config.COL_DATING_SITUATION}' column: {e}")
        return None

    # Fill NaN in Industry with "Unknown" for consistent string comparison later
    if config.COL_INDUSTRY_RENAMED in df.columns:
         df[config.COL_INDUSTRY_RENAMED] = df[config.COL_INDUSTRY_RENAMED].fillna("Unknown").astype(str)
    else:
        # If the industry column wasn't present, create it and fill with "Unknown"
        df[config.COL_INDUSTRY_RENAMED] = "Unknown"
        st.warning(f"⚠️ Industry column ('{config.COL_INDUSTRY}') not found. Industry matching bonus will not be applied.")

    # Fill NaNs in personality columns with a placeholder to avoid errors during scoring
    for col in config.PERSONALITY_BONUS_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("No Answer").astype(str)
        else:
            st.warning(f"⚠️ Personality question column '{col}' not found. It will be ignored for scoring.")
            # Optionally create the column filled with "No Answer" if needed elsewhere
            # df[col] = "No Answer"

    st.success("✅ Data loaded and preprocessed successfully.")
    return df


def is_compatible(row1, row2):
    """
    Checks if two participants are reciprocally compatible based on Gender and Seeking.
    Compatibility requires:
    - Person 1's Gender matches Person 2's Seeking preference.
    - Person 2's Gender matches Person 1's Seeking preference.
    - Both seeking preferences must be within the VALID_SEEKING_VALUES set (e.g., "Man", "Woman").

    Args:
        row1 (pd.Series): Row representing the first participant.
        row2 (pd.Series): Row representing the second participant.

    Returns:
        bool: True if compatible, False otherwise.
    """
    # Check if both seeking preferences are valid single targets (e.g., "Man" or "Woman")
    seeking1_valid = row1[config.COL_SEEKING] in config.VALID_SEEKING_VALUES
    seeking2_valid = row2[config.COL_SEEKING] in config.VALID_SEEKING_VALUES

    if not (seeking1_valid and seeking2_valid):
        return False # Not compatible if either is seeking something other than a single valid target

    # Check for reciprocal match
    return (row1[config.COL_GENDER] == row2[config.COL_SEEKING]) and (row2[config.COL_GENDER] == row1[config.COL_SEEKING])


def compute_match_score(row1, row2):
    """
    Computes a match score between two participants based on various factors.

    Score components:
    - Base score (starts at BASE_SCORE).
    - Age difference penalty (AGE_DIFF_PENALTY_PER_YEAR points per year).
    - Industry match bonus (INDUSTRY_MATCH_BONUS if industries match).
    - Personality match bonus (PERSONALITY_MATCH_BONUS_PER_QUESTION for each matching answer).

    Args:
        row1 (pd.Series): Row representing the first participant.
        row2 (pd.Series): Row representing the second participant.

    Returns:
        tuple: (total_score, age_diff, industry_bonus, personality_bonus)
    """
    # Age difference calculation (handle potential NaNs if not dropped earlier)
    age1 = row1[config.COL_AGE_RENAMED]
    age2 = row2[config.COL_AGE_RENAMED]
    if pd.isna(age1) or pd.isna(age2):
        age_diff = 100 # Assign a large difference if age is missing for either
    else:
        age_diff = abs(age1 - age2)
    age_penalty = config.AGE_DIFF_PENALTY_PER_YEAR * age_diff

    # Industry bonus calculation
    industry1 = str(row1[config.COL_INDUSTRY_RENAMED]).strip().lower()
    industry2 = str(row2[config.COL_INDUSTRY_RENAMED]).strip().lower()
    # Consider "Unknown" industry as not matching anything, including other "Unknowns"
    industry_bonus = 0
    if industry1 != "unknown" and industry1 == industry2:
         industry_bonus = config.INDUSTRY_MATCH_BONUS

    # Personality bonus calculation
    personality_bonus = 0
    for col in config.PERSONALITY_BONUS_COLUMNS:
        # Check if column exists in both rows (handles cases where column was missing in input)
        if col in row1 and col in row2:
            val1 = str(row1[col]).strip().lower()
            val2 = str(row2[col]).strip().lower()
            # Consider "No Answer" as not matching anything, including other "No Answers"
            if val1 != "no answer" and val1 == val2:
                personality_bonus += config.PERSONALITY_MATCH_BONUS_PER_QUESTION

    # Calculate total score
    total_score = config.BASE_SCORE - age_penalty + industry_bonus + personality_bonus
    # Ensure score doesn't go below zero (or some other floor)
    total_score = max(0, total_score)

    return total_score, age_diff, industry_bonus, personality_bonus


def perform_matching(df):
    """
    Builds a compatibility graph and computes matches.

    Steps:
    1. Filter for eligible participants (those seeking values in VALID_SEEKING_VALUES and non-NaN age).
    2. Build a graph where nodes are participant indices.
    3. Add edges between reciprocally compatible participants, weighted by their match score.
    4. Compute a maximum weight matching to find the best disjoint pairs.
    5. Attempt to find one additional match for any remaining unmatched participants,
       prioritizing edges that don't overload already matched individuals.
    6. Format and return the list of matches.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame of participants.

    Returns:
        list: A list of dictionaries, each representing a match, sorted by score.
              Returns empty list if no eligible participants or matches are found.
    """
    # Filter for participants who are eligible for matching
    # Eligibility: Seeking a value in VALID_SEEKING_VALUES and has a valid Age
    eligible_df = df[
        df[config.COL_SEEKING].isin(config.VALID_SEEKING_VALUES) & df[config.COL_AGE_RENAMED].notna()
    ].copy()

    if eligible_df.empty:
        st.warning("⚠️ No participants eligible for matching based on 'Seeking' preference and valid 'Age'.")
        return []

    num_eligible = len(eligible_df)
    st.info(f"ℹ️ Found {num_eligible} eligible participants for matching.")

    # --- Build Compatibility Graph ---
    G = nx.Graph()
    eligible_indices = list(eligible_df.index) # Use original DataFrame indices

    # Add nodes first
    for idx in eligible_indices:
        G.add_node(idx)

    # Add weighted edges for compatible pairs
    edge_count = 0
    for i in range(num_eligible):
        for j in range(i + 1, num_eligible):
            idx1 = eligible_indices[i]
            idx2 = eligible_indices[j]
            row1 = eligible_df.loc[idx1]
            row2 = eligible_df.loc[idx2]

            if is_compatible(row1, row2):
                score, age_diff, ind_bonus, pers_bonus = compute_match_score(row1, row2)
                # Only add edges with a positive score (or adjust threshold if needed)
                if score > 0:
                    G.add_edge(idx1, idx2, weight=score, age_diff=age_diff,
                               industry_bonus=ind_bonus, personality_bonus=pers_bonus)
                    edge_count += 1

    if edge_count == 0:
        st.warning("⚠️ No compatible pairs found among eligible participants.")
        return []
    st.info(f"ℹ️ Calculated {edge_count} potential compatibility links.")


    # --- Compute Maximum Weight Matching ---
    # This finds the set of pairs with the maximum total weight (score),
    # ensuring no person is in more than one pair (maxcardinality=True helps ensure this).
    try:
        # Using maxcardinality=True prioritizes finding the largest possible number of pairs
        # if multiple matchings have the same total weight.
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
        # Convert the set of tuples to a more manageable set of sorted tuples
        # Each element in 'matching' is a tuple (u, v) representing a matched pair.
        initial_matching_pairs = set(tuple(sorted(list(edge))) for edge in matching)
        st.info(f"ℹ️ Initial matching found {len(initial_matching_pairs)} pairs.")

    except Exception as e:
        st.error(f"❌ Error during graph matching algorithm: {e}")
        return []


    # --- Handle Unmatched Participants ---
    matched_nodes = set()
    for pair in initial_matching_pairs:
        matched_nodes.update(pair)

    unmatched_nodes = set(eligible_indices) - matched_nodes
    st.info(f"ℹ️ {len(unmatched_nodes)} participants remain unmatched after initial pairing.")

    final_matching_pairs = set(initial_matching_pairs) # Start with the initial best pairs

    # Try to find one additional match for each unmatched person
    # Prioritize matches that don't create triplets/quartets unless necessary
    # and prefer higher scores among those.
    max_degree_allowed = 1 # How many matches a person can have (1 means pairs only, 2 allows triplets etc.)

    processed_unmatched = set() # Keep track of unmatched nodes we've already tried to find a match for

    # Iterate multiple times potentially? For now, one pass.
    for u in list(unmatched_nodes): # Iterate over a copy as we might modify the set implicitly
         if u in processed_unmatched:
             continue

         candidate_edges = []
         # Find potential partners for u among ALL eligible nodes (both matched and unmatched)
         for v in G.neighbors(u):
             # Don't match with self (shouldn't happen with G.neighbors, but safe)
             if u == v: continue

             current_edge = tuple(sorted([u, v]))

             # Check if this edge is already in our final list
             if current_edge in final_matching_pairs:
                 continue # This pair is already formed

             # Check degrees in the *current* final matching set
             degree_u = sum(1 for edge in final_matching_pairs if u in edge)
             degree_v = sum(1 for edge in final_matching_pairs if v in edge)

             # Consider this edge if adding it doesn't exceed the max degree for *either* person
             if degree_u < max_degree_allowed and degree_v < max_degree_allowed:
                 candidate_edges.append({
                     "edge": current_edge,
                     "score": G[u][v]['weight'],
                     "combined_degree": degree_u + degree_v # Lower combined degree is better (less disruption)
                 })

         # If we found potential additional matches for u
         if candidate_edges:
            # Sort candidates: primarily by highest score, secondarily by lowest combined degree
            candidate_edges.sort(key=lambda x: (-x['score'], x['combined_degree']))
            best_candidate = candidate_edges[0]
            final_matching_pairs.add(best_candidate['edge'])
            # Mark both nodes of the new pair as processed in this round of adding extras
            processed_unmatched.add(best_candidate['edge'][0])
            processed_unmatched.add(best_candidate['edge'][1])
            # st.write(f"DEBUG: Added extra match: {best_candidate['edge']} Score: {best_candidate['score']}") # Optional debug


    st.info(f"ℹ️ Final matching includes {len(final_matching_pairs)} pairs after considering unmatched participants.")

    # --- Format Results ---
    matches_list = []
    outputted_indices = set() # Keep track of indices already included in a match description

    for edge in final_matching_pairs:
        idx1, idx2 = edge
        # Check if either participant has already been listed (handles potential overlaps if max_degree > 1)
        # For strict pairing (max_degree=1), this check isn't strictly needed but good practice.
        # if idx1 in outputted_indices or idx2 in outputted_indices:
        #     continue # Skip if part of an already described match (e.g., in a triplet scenario)

        # Retrieve edge data
        edge_data = G[idx1][idx2]
        score = edge_data['weight']
        age_diff = edge_data['age_diff']
        ind_bonus = edge_data['industry_bonus']
        pers_bonus = edge_data['personality_bonus']

        # Get participant details from the original DataFrame using the indices
        p1 = df.loc[idx1]
        p2 = df.loc[idx2]

        # Format participant strings
        candidate1_desc = (f"{p1.get(config.COL_FIRST_NAME, 'N/A')} {p1.get(config.COL_LAST_NAME, '')} "
                           f"(Age: {p1.get(config.COL_AGE_RENAMED, 'N/A')}, Gender: {p1.get(config.COL_GENDER, 'N/A')}, Seeking: {p1.get(config.COL_SEEKING, 'N/A')})")
        candidate2_desc = (f"{p2.get(config.COL_FIRST_NAME, 'N/A')} {p2.get(config.COL_LAST_NAME, '')} "
                           f"(Age: {p2.get(config.COL_AGE_RENAMED, 'N/A')}, Gender: {p2.get(config.COL_GENDER, 'N/A')}, Seeking: {p2.get(config.COL_SEEKING, 'N/A')})")

        candidate1_name = f"{p1.get(config.COL_FIRST_NAME, 'N/A')} {p1.get(config.COL_LAST_NAME, '')}".strip()
        candidate2_name = f"{p2.get(config.COL_FIRST_NAME, 'N/A')} {p2.get(config.COL_LAST_NAME, '')}".strip()

        matches_list.append({
            "pair_description": f"{candidate1_desc}  <-->  {candidate2_desc}",
            "score": score,
            "age_diff": age_diff,
            "industry_bonus": ind_bonus,
            "personality_bonus": pers_bonus,
            "candidate1_idx": idx1, # Store indices if needed later
            "candidate2_idx": idx2,
            "candidate1_name": candidate1_name,
            "candidate2_name": candidate2_name
        })
        # Mark these indices as outputted
        outputted_indices.add(idx1)
        outputted_indices.add(idx2)

    # Sort final list by score
    final_matches_sorted = sorted(matches_list, key=lambda x: x["score"], reverse=True)
    return final_matches_sorted


# --- Streamlit App Layout ---

st.set_page_config(page_title="Haystack Dating", layout="wide")
st.title("💖 Haystack Dating - Optimized Pair Matchmaking")
st.markdown("Upload your participant data (Excel or CSV) to find the best potential pairs based on compatibility and preferences.")

# --- Sidebar for Upload and Control ---
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("1. Upload Participant Data", type=["xlsx", "csv"], help="Upload an Excel (.xlsx) or CSV (.csv) file containing participant details.")
start_button = st.sidebar.button("🚀 Run Matchmaking Analysis", disabled=(uploaded_file is None), help="Click to start the analysis after uploading a file.")

# --- Main Area for Results ---
if start_button and uploaded_file:
    st.info(f"Processing file: {uploaded_file.name}")
    with st.spinner('⏳ Loading and preprocessing data...'):
        df = load_data(uploaded_file)

    if df is not None:
        st.subheader("📊 Input Data Overview")
        st.dataframe(df.head()) # Show only the head to avoid overwhelming the UI
        st.markdown(f"Total participants loaded: {len(df)}")

        st.subheader("💞 Matchmaking Results")
        with st.spinner('⏳ Calculating compatibility and finding best pairs... This may take a moment.'):
            final_matches = perform_matching(df)

        if final_matches:
            st.success(f"✅ Found {len(final_matches)} potential pairs!")

            # Display matches using expanders for detail
            st.markdown("#### Top Matches (Sorted by Score):")
            for i, match in enumerate(final_matches):
                 with st.expander(f"Pair {i+1}: {match['candidate1_name']} & {match['candidate2_name']} | Score: {match['score']:.0f}", expanded=(i < 5)): # Expand top 5
                     st.markdown(f"**Details:** {match['pair_description']}")
                     st.markdown(f"**Score Breakdown:**")
                     st.markdown(f"  - Base Score: {config.BASE_SCORE}")
                     st.markdown(f"  - Age Difference Penalty: -{match['age_diff'] * config.AGE_DIFF_PENALTY_PER_YEAR:.0f} (Diff: {match['age_diff']:.0f} yrs)")
                     st.markdown(f"  - Industry Bonus: +{match['industry_bonus']:.0f}")
                     st.markdown(f"  - Personality Bonus: +{match['personality_bonus']:.0f}")
                     st.divider()


            # --- Summary Statistics Section ---
            st.subheader("📈 Summary Statistics")
            candidate_stats = {}
            all_participants_in_matches = set()

            for match in final_matches:
                c1_name = match["candidate1_name"]
                c2_name = match["candidate2_name"]
                all_participants_in_matches.add(c1_name)
                all_participants_in_matches.add(c2_name)

                # Initialize if first time seeing candidate
                if c1_name not in candidate_stats: candidate_stats[c1_name] = {"count": 0, "highest_score": 0, "partners": []}
                if c2_name not in candidate_stats: candidate_stats[c2_name] = {"count": 0, "highest_score": 0, "partners": []}

                # Update stats for candidate 1
                candidate_stats[c1_name]["count"] += 1
                candidate_stats[c1_name]["highest_score"] = max(match["score"], candidate_stats[c1_name]["highest_score"])
                candidate_stats[c1_name]["partners"].append(f"{c2_name} ({match['score']:.0f})")

                # Update stats for candidate 2
                candidate_stats[c2_name]["count"] += 1
                candidate_stats[c2_name]["highest_score"] = max(match["score"], candidate_stats[c2_name]["highest_score"])
                candidate_stats[c2_name]["partners"].append(f"{c1_name} ({match['score']:.0f})")

            if candidate_stats:
                total_matches_df = pd.DataFrame([
                    {
                        "Candidate": cand,
                        "Total Matches": stats["count"],
                        "Highest Score": stats["highest_score"],
                        "Matched With (Score)": ", ".join(stats["partners"])
                     }
                    for cand, stats in candidate_stats.items()
                ])
                total_matches_df.sort_values(by="Highest Score", ascending=False, inplace=True)
                st.dataframe(total_matches_df)
                st.markdown(f"Total participants involved in matches: {len(all_participants_in_matches)}")
            else:
                 st.info("No participants were involved in the final matches.")

        else:
            # Handle cases where perform_matching returned an empty list due to warnings/errors inside it
            if 'eligible_df' in locals() and not config.eligible_df.empty: # Check if eligibility was the issue
                st.warning("⚠️ Although eligible participants were found, no compatible pairs could be formed based on the criteria.")
            elif df is not None: # If df loaded but no eligible participants
                st.warning("⚠️ No participants were eligible for matching based on the provided data (check 'Dating situation' and 'Age' columns).")
            # If df itself was None, the error was already shown during loading.

elif start_button and not uploaded_file:
    st.sidebar.warning("⚠️ Please upload a file first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Haystack Dating.")