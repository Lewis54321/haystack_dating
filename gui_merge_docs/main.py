import streamlit as st
import pandas as pd
from io import BytesIO
from thefuzz import fuzz  # For fuzzy matching later
import re  # For cleaning reference numbers
import numpy as np


# --- Configuration ---
# Define the column names expected in the files
# Adjust these if your actual column names differ slightly (e.g., case, spacing)
CSV_COLS = {
    "ref": "order_reference",
    "email": "email",
    "fname": "first_name",
    "lname": "last_name",
    "age": "age_number_format_please",
    "gender": "gender",
    "goal": "dating_goal"
}

RESPONSE_COLS = {
    "ref": "please_enter_your_fatsoma_ticket_reference_number",
    "email": "email_address",  # Assuming the first email column is the primary one
    "fname": "first_name",
    "lname": "last_name",
    "age": "age_number_format_please",
    "situation": "dating_situation",  # Equivalent of 'Dating goal'
    "industry": "which_best_describes_the_industry_role_you_work_in"
}

# Fuzzy matching threshold
HIGH_CONFIDENCE_THRESHOLD = 90  # Scores >= this are considered definite matches
MANUAL_REVIEW_THRESHOLD = 75  # Scores >= this but < HIGH_CONFIDENCE trigger manual review
AGE_TOLERANCE = 1  # Allowable age difference (+/- years)


# --- Helper Functions ---

def clean_email(email):
    """Converts email to lowercase and removes whitespace."""
    if isinstance(email, str):
        return email.lower().strip()
    return None


def clean_reference(ref):
    if isinstance(ref, str):
        return re.sub(r'[^a-zA-Z0-9]', '', ref).strip()
    elif isinstance(ref, (int, float)):
        ref_str = str(ref)
        return re.sub(r'[^a-zA-Z0-9]', '', ref_str).strip()
    return None


def clean_name(name):
    """Converts name to lowercase and removes whitespace."""
    if isinstance(name, str):
        return name.lower().strip()
    return ""  # Return empty string for non-string names for concatenation


def safe_to_numeric(value):
    """Safely converts a value to numeric (integer), returns np.nan on failure."""
    if pd.isna(value):
        return np.nan
    try:
        # Attempt conversion to float first to handle potential decimals, then to int
        return int(float(value))
    except (ValueError, TypeError):
        return np.nan


@st.cache_data # Cache the data loading
def load_data(uploaded_file):
    """Loads data from uploaded file (CSV or XLSX)."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload .csv or .xlsx")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file '{uploaded_file.name}': {e}")
        return None


def clean_column_names(df):
    """
    Cleans DataFrame column names:
    1. Converts to lowercase.
    2. Strips leading/trailing whitespace.
    3. Removes brackets '()', exclamation marks '!', and question marks '?'.
    4. Replaces spaces and common separators (like '/') with underscores.
    Returns the DataFrame with cleaned columns and a mapping from cleaned to original names.
    """
    if df is None:
        return None, {}

    original_columns = df.columns.tolist()
    cleaned_columns = df.columns.astype(str).str.lower()
    cleaned_columns = cleaned_columns.str.strip()

    # *** Step 1: Remove specific characters completely ***
    cleaned_columns = cleaned_columns.str.replace(r'[()!?]', '', regex=True) # Remove (), !, ?

    # *** Step 2: Replace spaces/slashes with underscores ***
    cleaned_columns = cleaned_columns.str.replace(r'[ /+]', '_', regex=True) # Replace space, /, +

    # *** Step 3: Clean up multiple/trailing underscores ***
    cleaned_columns = cleaned_columns.str.replace(r'_+', '_', regex=True)
    cleaned_columns = cleaned_columns.str.strip('_')

    # Handle potential duplicate cleaned names
    cols_seen = {}
    new_cols = []
    for i, col in enumerate(cleaned_columns):
        original_col = original_columns[i]
        if col in cols_seen:
            cols_seen[col] += 1
            new_col_name = f"{col}_{cols_seen[col]}"
            while new_col_name in cleaned_columns.tolist() or new_col_name in new_cols:
                 cols_seen[col] += 1
                 new_col_name = f"{col}_{cols_seen[col]}"
            new_cols.append(new_col_name)
        else:
            cols_seen[col] = 0
            new_cols.append(col)
    cleaned_columns = pd.Index(new_cols)

    original_name_map = dict(zip(cleaned_columns, original_columns))
    df.columns = cleaned_columns
    # st.write("Cleaned Cols:", df.columns.tolist()) # For debugging
    # st.write("Original Map:", original_name_map) # For debugging
    return df, original_name_map


def to_excel(df, original_name_map):
    """
    Converts DataFrame to Excel format in memory, restoring original column names.
    """
    output = BytesIO()
    df_export = df.copy()
    # Rename using the map: Cleaned Name -> Original Name
    df_export.rename(columns=original_name_map, inplace=True)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Merged Data')
    output.seek(0)
    return output.getvalue()


# --- Initialize Session State ---
if 'review_index' not in st.session_state:
    st.session_state.review_index = 0
if 'potential_matches' not in st.session_state:
    st.session_state.potential_matches = []
if 'user_decisions' not in st.session_state:
    st.session_state.user_decisions = {} # Store decisions {ticket_index: 'match'/'no_match'}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'final_rows_to_add' not in st.session_state:
    st.session_state.final_rows_to_add = []
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = None
if 'response_original_map' not in st.session_state:
    st.session_state.response_original_map = {}
if 'ticket_original_map' not in st.session_state:
    st.session_state.ticket_original_map = {}
if 'initial_ticket_count' not in st.session_state: # Added for summary
    st.session_state.initial_ticket_count = 0
if 'initial_response_count' not in st.session_state: # Added for summary
    st.session_state.initial_response_count = 0
if 'definite_match_count' not in st.session_state: # Added for summary
    st.session_state.definite_match_count = 0
if 'reviewed_match_count' not in st.session_state: # Added for summary
    st.session_state.reviewed_match_count = 0


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Ticket & Response Data Merger Tool")

# --- Sidebar for File Uploads ---
st.sidebar.header("Upload Files")
st.sidebar.info("Please upload the 'Response Document' (Excel or CSV) and the 'Ticket Document' (CSV).")

# Reset state if files change
response_file = st.sidebar.file_uploader("1. Upload Response Document (.xlsx or .csv)", type=['csv', 'xlsx'], key="response", on_change=lambda: st.session_state.update(review_index=0, potential_matches=[], user_decisions={}, processing_complete=False, final_rows_to_add=[], df_merged=None, response_original_map={}, ticket_original_map={}, initial_ticket_count=0, initial_response_count=0, definite_match_count=0, reviewed_match_count=0))
ticket_file = st.sidebar.file_uploader("2. Upload Ticket Document (.csv)", type=['csv'], key="ticket", on_change=lambda: st.session_state.update(review_index=0, potential_matches=[], user_decisions={}, processing_complete=False, final_rows_to_add=[], df_merged=None, response_original_map={}, ticket_original_map={}, initial_ticket_count=0, initial_response_count=0, definite_match_count=0, reviewed_match_count=0))

# --- Main Processing Area ---

if response_file and ticket_file:
    st.success("Files uploaded successfully!")

    df_response_loaded = load_data(response_file)
    df_ticket_loaded = load_data(ticket_file)

    if df_response_loaded is not None and df_ticket_loaded is not None:

        # Store initial counts
        st.session_state.initial_ticket_count = len(df_ticket_loaded)
        st.session_state.initial_response_count = len(df_response_loaded)

        # Check if processing needs to run (only run once after file upload unless reset)
        if not st.session_state.processing_complete and not st.session_state.potential_matches:
            st.success("Files uploaded successfully! Starting analysis...")

            # --- Data Cleaning and Preparation ---
            df_response, response_original_map = clean_column_names(df_response_loaded.copy())
            df_ticket, ticket_original_map = clean_column_names(df_ticket_loaded.copy())
            st.session_state.response_original_map = response_original_map
            st.session_state.ticket_original_map = ticket_original_map

            # --- Column Existence Checks (using dictionary VALUES) ---
            # Ensure the values in RESPONSE_COLS/CSV_COLS match the expected cleaned names
            required_ticket_cols = list(CSV_COLS.values())
            missing_ticket_cols = [col for col in required_ticket_cols if col not in df_ticket.columns]
            if missing_ticket_cols:
                missing_orig_names = [st.session_state.ticket_original_map.get(col, col) for col in missing_ticket_cols]
                st.error(f"Error: Ticket Document missing columns: {', '.join(missing_orig_names)}")
                st.stop()

            required_response_cols = list(RESPONSE_COLS.values())
            missing_response_cols = [col for col in required_response_cols if col not in df_response.columns]
            if missing_response_cols:
                missing_orig_names = [st.session_state.response_original_map.get(col, col) for col in
                                      missing_response_cols]
                st.error(
                    f"Error: Response Document missing columns needed for matching: {', '.join(missing_orig_names)}. Please check the file.")
                st.stop()

            # Prepare response data for matching (use dictionary VALUES to access columns)
            response_ref_col = RESPONSE_COLS['ref']
            response_email_col = RESPONSE_COLS['email']
            response_fname_col = RESPONSE_COLS['fname']
            response_lname_col = RESPONSE_COLS['lname']
            response_age_col = RESPONSE_COLS[
                'age']  # This should now match the cleaned name, e.g., 'age_number_format_please'

            response_refs = set(df_response[response_ref_col].apply(clean_reference).dropna())
            response_emails = set(df_response[response_email_col].apply(clean_email).dropna())

            # Handle potential KeyError if age column name was changed significantly by cleaning
            try:
                df_response['cleaned_fullname'] = df_response[response_fname_col].apply(clean_name) + " " + df_response[
                    response_lname_col].apply(clean_name)
                df_response['cleaned_age'] = df_response[response_age_col].apply(safe_to_numeric)
                response_match_data = df_response[['cleaned_fullname', 'cleaned_age']].reset_index()
            except KeyError as e:
                st.error(
                    f"Error accessing column for matching: {e}. Check if the cleaned column name '{str(e).strip()}' matches the expected value in RESPONSE_COLS after cleaning.")
                st.stop()

            # --- Matching Logic ---
            matched_ticket_indices = set()
            potential_matches_list = []
            rows_definitely_to_add_indices = set()  # Indices of ticket rows with no potential match

            with st.spinner("Matching records (Ref, Email, Name + Age)..."):
                # 1. Match by Reference Number
                for index, ticket_row in df_ticket.iterrows():
                    ticket_ref_cleaned = clean_reference(ticket_row.get('order reference'))
                    if ticket_ref_cleaned and ticket_ref_cleaned in response_refs:
                        matched_ticket_indices.add(index)

                # 2. Match by Email (for those not matched by reference)
                for index, ticket_row in df_ticket.iterrows():
                    if index not in matched_ticket_indices:
                        ticket_email_cleaned = clean_email(ticket_row.get('email'))
                        if ticket_email_cleaned and ticket_email_cleaned in response_emails:
                            matched_ticket_indices.add(index)

                # 3. Fuzzy Name Matching with Age Check (for those not matched yet)
                ticket_age_col_cleaned = 'age'  # Use cleaned ticket column name
                for index, ticket_row in df_ticket.iterrows():
                    if index not in matched_ticket_indices:
                        ticket_fname = clean_name(ticket_row.get('first name', ''))
                        ticket_lname = clean_name(ticket_row.get('last name', ''))
                        ticket_fullname = f"{ticket_fname} {ticket_lname}".strip()
                        ticket_age_num = safe_to_numeric(ticket_row.get(ticket_age_col_cleaned))  # Clean ticket age

                        # Skip if ticket name is empty or age is invalid
                        if not ticket_fullname or pd.isna(ticket_age_num):
                            rows_definitely_to_add_indices.add(index)
                            continue

                        best_score = 0
                        best_match_response_index = -1

                        # Compare against all response names & ages
                        # response_match_data columns: index, cleaned_fullname, cleaned_age
                        for resp_original_idx, resp_name, resp_age_num in response_match_data.itertuples(index=False):

                            # *** AGE CHECK ADDED HERE ***
                            # Check if response age is valid and if the difference is within tolerance
                            if pd.isna(resp_age_num) or abs(ticket_age_num - resp_age_num) > AGE_TOLERANCE:
                                continue  # Skip this response record if age is invalid or doesn't match tolerance

                            # *** FUZZY NAME MATCH (Only if Age Check Passed) ***
                            score = fuzz.token_sort_ratio(ticket_fullname, resp_name)
                            if score > best_score:
                                best_score = score
                                best_match_response_index = resp_original_idx  # Store the original index

                        # Decide based on score (only considers pairs that passed age check)
                        if best_score >= HIGH_CONFIDENCE_THRESHOLD:
                            matched_ticket_indices.add(index)
                        elif best_score >= MANUAL_REVIEW_THRESHOLD:
                            potential_matches_list.append({
                                "ticket_index": index,
                                "response_index": best_match_response_index,
                                "score": best_score,
                                "ticket_row": ticket_row,
                                "response_row": df_response.iloc[best_match_response_index]
                                # Use original index to get row
                            })
                        else:  # Score below manual review threshold OR no response passed age check
                            rows_definitely_to_add_indices.add(index)

            st.session_state.potential_matches = potential_matches_list
            st.session_state.rows_definitely_to_add = [df_ticket.iloc[i] for i in
                                                       rows_definitely_to_add_indices]  # Store the actual rows
            st.session_state.review_index = 0
            st.session_state.user_decisions = {}

            st.success(f"Initial matching complete. Found {len(matched_ticket_indices)} definite matches.")
            if st.session_state.potential_matches:
                st.info(f"{len(st.session_state.potential_matches)} potential matches require your review.")
            if st.session_state.rows_definitely_to_add:
                st.info(
                    f"{len(st.session_state.rows_definitely_to_add)} records will be added (no potential match found).")

        # --- Interactive Review Section ---
        if st.session_state.potential_matches and st.session_state.review_index < len(
                st.session_state.potential_matches):
            st.subheader(
                f"Manual Review ({st.session_state.review_index + 1} of {len(st.session_state.potential_matches)})")

            current_match = st.session_state.potential_matches[st.session_state.review_index]
            ticket_idx = current_match['ticket_index']
            response_idx = current_match['response_index']
            score = current_match['score']
            ticket_row = current_match['ticket_row']
            response_row = current_match['response_row']

            st.warning(f"Potential match found with score: **{score}**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Ticket Document Record:**")
                # Display key ticket info using CSV_COLS values for original names
                st.write(f"**Name:** {ticket_row.get('first name', 'N/A')} {ticket_row.get('last name', 'N/A')}")
                st.write(f"**Email:** {ticket_row.get('email', 'N/A')}")
                st.write(f"**Age:** {ticket_row.get('age', 'N/A')}")
                st.write(f"**Ref:** {ticket_row.get('order reference', 'N/A')}")
                st.write(f"**Goal:** {ticket_row.get('dating goal', 'N/A')}")
                # st.dataframe(pd.DataFrame(ticket_row).T) # Option to show full row

            with col2:
                st.markdown("**Response Document Record:**")
                # Display key response info using RESPONSE_COLS values
                st.write(
                    f"**Name:** {response_row.get(RESPONSE_COLS['fname'], 'N/A')} {response_row.get(RESPONSE_COLS['lname'], 'N/A')}")
                st.write(f"**Email:** {response_row.get(RESPONSE_COLS['email'], 'N/A')}")
                st.write(f"**Age:** {response_row.get(RESPONSE_COLS['age'], 'N/A')}")
                st.write(f"**Ref:** {response_row.get(RESPONSE_COLS['ref'], 'N/A')}")
                st.write(f"**Situation:** {response_row.get(RESPONSE_COLS['situation'], 'N/A')}")
                st.write(f"**Timestamp:** {response_row.get(RESPONSE_COLS['timestamp'], 'N/A')}")  # Show extra context
                # st.dataframe(pd.DataFrame(response_row).T) # Option to show full row

            st.markdown("---")
            st.write("**Is this the same person?**")

            button_col1, button_col2, _ = st.columns([1, 1, 3])  # Adjust spacing

            with button_col1:
                if st.button("✅ Yes, it's a Match (Don't Add)", key=f"yes_{ticket_idx}"):
                    st.session_state.user_decisions[ticket_idx] = 'match'
                    st.session_state.review_index += 1
                    st.rerun()  # Rerun to show next review item or finish

            with button_col2:
                if st.button("❌ No, Not a Match (Add This Ticket Record)", key=f"no_{ticket_idx}"):
                    st.session_state.user_decisions[ticket_idx] = 'no_match'
                    st.session_state.review_index += 1
                    st.rerun()  # Rerun to show next review item or finish

        # --- Final Processing (After Review or if No Review Needed) ---
        elif not st.session_state.processing_complete:
            st.subheader("Finalizing...")

            final_add_indices = set(row.name for row in st.session_state.rows_definitely_to_add)
            for ticket_idx, decision in st.session_state.user_decisions.items():
                if decision == 'no_match':
                    final_add_indices.add(ticket_idx)

            st.session_state.final_rows_to_add = [df_ticket.loc[i] for i in final_add_indices if i in df_ticket.index]

            st.info(f"Total records to be added: {len(st.session_state.final_rows_to_add)}")

            if st.session_state.final_rows_to_add:
                st.subheader("Preparing Final Merged Data")
                df_new_rows = pd.DataFrame(st.session_state.final_rows_to_add).reset_index(drop=True)

                if not df_new_rows.empty:
                    # Define mapping using the dictionary VALUES (cleaned column names)
                    column_mapping_cleaned = {
                        CSV_COLS["fname"]: RESPONSE_COLS["fname"],
                        CSV_COLS["lname"]: RESPONSE_COLS["lname"],
                        CSV_COLS["email"]: RESPONSE_COLS["email"],
                        CSV_COLS["ref"]: RESPONSE_COLS["ref"],
                        CSV_COLS["age"]: RESPONSE_COLS["age"],
                        CSV_COLS["goal"]: RESPONSE_COLS["situation"],
                    }
                    cols_to_select = [key for key in column_mapping_cleaned.keys() if key in df_new_rows.columns]
                    df_formatted_new = df_new_rows[cols_to_select].rename(columns=column_mapping_cleaned)

                    # Add missing columns using the CLEANED response df structure
                    for col in df_response.columns:
                        if col not in ['cleaned_fullname', 'cleaned_age',
                                       'index'] and col not in df_formatted_new.columns:
                            df_formatted_new[col] = pd.NA

                    cleaned_response_cols_ordered = [col for col in df_response.columns if
                                                     col not in ['cleaned_fullname', 'cleaned_age', 'index']]
                    ordered_cols = [col for col in cleaned_response_cols_ordered if col in df_formatted_new.columns]
                    for col in ordered_cols:
                        if col not in df_formatted_new:
                            df_formatted_new[col] = pd.NA
                    df_formatted_new = df_formatted_new[ordered_cols]

                    st.markdown("**Preview of Records Added (Cleaned Format):**")
                    st.dataframe(df_formatted_new.head())

                    df_response_clean_original = df_response.drop(columns=['cleaned_fullname', 'cleaned_age'],
                                                                  errors='ignore')
                    st.session_state.df_merged = pd.concat([df_response_clean_original, df_formatted_new],
                                                           ignore_index=True)

                else:
                    st.session_state.df_merged = df_response.drop(columns=['cleaned_fullname', 'cleaned_age'],
                                                                  errors='ignore')

            else:
                st.info("No new records were added after review.")
                st.session_state.df_merged = df_response.drop(columns=['cleaned_fullname', 'cleaned_age'],
                                                              errors='ignore')

            st.session_state.processing_complete = True
            st.rerun()

        # --- Display Summary and Download Button ---
        # This block now runs only after processing is complete
        if st.session_state.processing_complete and st.session_state.df_merged is not None:

            # *** ADDED SUMMARY SECTION HERE ***
            st.subheader("Summary Statistics")
            st.write(f"- Initial Ticket Document Entries: {st.session_state.initial_ticket_count}")
            st.write(f"- Initial Response Document Entries: {st.session_state.initial_response_count}")
            st.write(
                f"- Definite Matches Found (Ref/Email/High-Confidence): {st.session_state.definite_match_count}")
            st.write(f"- Potential Matches Reviewed by User: {st.session_state.reviewed_match_count}")
            st.write(f"- Records Added from Ticket Document: {len(st.session_state.final_rows_to_add)}")
            st.write(f"- Total Entries in Final Merged Document: {len(st.session_state.df_merged)}")
            st.markdown("---")  # Separator

            # Display preview of added records if any were added
            if st.session_state.final_rows_to_add:
                st.markdown("**Preview of Records Added:**")
                # Need to re-create df_formatted_new here for display if needed, or just show final merged head
                # For simplicity, just show the head of the final merged df
                # st.dataframe(df_formatted_new.head()) # This df isn't available here anymore
            else:
                st.info("No new records were added.")

            st.subheader("Download Merged File")
            st.dataframe(st.session_state.df_merged.head())
            excel_data = to_excel(st.session_state.df_merged, st.session_state.response_original_map)
            st.download_button(
                label="📥 Download Merged Data as Excel (.xlsx)",
                data=excel_data,
                file_name="merged_response_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    elif response_file and df_response_loaded is None:
        st.error("Could not load the Response Document. Please check the file format and content.")
    elif ticket_file and df_ticket_loaded is None:
        st.error("Could not load the Ticket Document. Please check the file format and content.")


elif response_file or ticket_file:
    st.warning("Please upload *both* the Response Document and the Ticket Document.")
else:
    st.info("Upload your files using the sidebar to begin.")

