
# --- Configuration Constants ---
# Define column names used in the input file
COL_FIRST_NAME = "First Name"
COL_LAST_NAME = "Last Name"
COL_AGE = "Age (number format please)"  # Original name for loading
COL_AGE_RENAMED = "Age"  # Renamed version for internal use
COL_INDUSTRY = "Which best describes the industry/role you work in?"  # Original name
COL_INDUSTRY_RENAMED = "Industry"  # Renamed version
COL_DATING_SITUATION = "Dating situation"
COL_TICKET_REF = "Please enter your Fatsoma ticket reference number"  # Optional

# Define internal column names created during processing
COL_GENDER = "Gender"
COL_SEEKING = "Seeking"

# Define columns used for personality bonus calculation
PERSONALITY_BONUS_COLUMNS = [
    "Which best describes your relationship with exercise?",
    "Congrats, you won a 3 weeks annual leave and the postcode lottery! Where are you heading?",
    "How important would you say career is to you?",
    "It’s New Years Eve party at 10pm. Where are you?",
    "Who would you most like share a cheeseboard with?",
    "What’s your dating vibe at the moment?",
    "Your wife/husband (that you found at a Haystack Dating event x) comes home one day with 6 sausage dogs, in matching trousers. What’s your reaction?",
    "It’s your Nan's birthday. What’s your move?"
]

# Define scoring parameters
AGE_DIFF_PENALTY_PER_YEAR = 2
INDUSTRY_MATCH_BONUS = 10
PERSONALITY_MATCH_BONUS_PER_QUESTION = 5
BASE_SCORE = 100

# Define compatibility values
VALID_SEEKING_VALUES = {"Man", "Woman"}
GENDER_MAP = {"women": "Woman", "woman": "Woman", "men": "Man", "man": "Man"}
PREFERENCE_TERMS = ["bisexual", "pansexual", "demisexual", "homosexual"]  # Used for parsing