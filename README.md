# PA7 Chatbot: Movie Recommendation Conversational Agent

## Overview

This repository contains the code for a movie recommendation chatbot, inspired by classic conversational agents like ELIZA and GUS, and extended with modern LLM (Large Language Model) capabilities. The chatbot interacts with users to learn their movie preferences and recommends films using collaborative filtering. It supports three modes:

- **Starter (GUS) Mode:** Classic rule-based movie recommendation.
- **LLM Prompting Mode:** Uses prompt engineering to guide an LLM for movie-focused conversations.
- **LLM Programming Mode:** Combines Python logic and LLM calls for advanced, creative chatbot features.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Testing](#testing)
- [Data & Resources](#data--resources)
- [Extending the Chatbot](#extending-the-chatbot)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Conversational Movie Recommendations:** Learns user preferences and suggests movies.
- **Sentiment Analysis:** Detects positive, negative, or neutral sentiment about movies.
- **Title Extraction:** Identifies movie titles from user input, including handling of foreign titles.
- **Collaborative Filtering:** Recommends movies using item-item cosine similarity.
- **LLM Integration:** Supports both prompt-based and programmatic LLM calls for richer interactions.
- **Scripted and Interactive Testing:** Includes scripts and tools for automated and manual testing.

---

## Project Structure

```
pa7-chatbot-main/
│
├── chatbot.py                # Main chatbot logic (all modes)
├── repl.py                   # REPL interface for interactive/chat mode
├── util.py                   # Utility functions (data loading, LLM calls)
├── porter_stemmer.py         # (Optional) Stemming utility
├── data/
│   ├── movies.txt            # Movie titles and genres
│   ├── ratings.txt           # User-movie ratings matrix
│   └── sentiment.txt         # Sentiment lexicon
├── examples/
│   ├── json_llm_example.py   # Example: LLM JSON output
│   └── simple_llm_example.py # Example: LLM prompt output
├── testing/
│   ├── sanitycheck.py        # Sanity check script for core functions
│   ├── run_all_scripts.sh    # Run all test scripts
│   └── test_scripts/
│       ├── simple.txt
│       ├── standard/
│       │   ├── fail_gracefully.txt
│       │   └── recommend.txt
│       ├── llm_prompting/
│       │   ├── distraction_easy.txt
│       │   ├── distraction_hard.txt
│       │   ├── recommend.txt
│       │   └── simple.txt
│       └── llm_programming/
│           ├── arbitrary.txt
│           ├── emotions.txt
│           ├── foreign.txt
│           └── persona.txt
├── outputs-for-scripts/      # Output transcripts for test scripts
├── generate_submission.sh    # Script to create submission.zip
├── rubric.txt                # Rubric for grading (edit to indicate features)
└── README.md                 # Original assignment README
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd pa7-chatbot-main
   ```

2. **Set up the Python environment:**
   - Recommended: Use the `cs124` conda environment from PA0.
   - Install required packages:
     ```bash
     conda activate cs124
     pip install openai
     ```

3. **API Key for LLM Features:**
   - For LLM modes, add your Together API key to `api_keys.py`:
     ```python
     TOGETHER_API_KEY = "your-key-here"
     ```
   - See [Together API instructions](https://docs.google.com/document/d/1N5chC5b15ls-XXcpfjhSx71854fmvb_4DGD4qxkT0LU/edit?usp=sharing).

---

## Usage

### Interactive Chat

- **Starter Mode (default):**
  ```bash
  python3 repl.py
  ```

- **LLM Prompting Mode:**
  ```bash
  python3 repl.py --llm_prompting
  ```

- **LLM Programming Mode:**
  ```bash
  python3 repl.py --llm_programming
  ```

### Scripted Testing

- Run a test script:
  ```bash
  python3 repl.py < testing/test_scripts/simple.txt
  ```

- Run all test scripts:
  ```bash
  sh testing/run_all_scripts.sh
  ```

- Run sanity checks:
  ```bash
  python3 testing/sanitycheck.py
  # or for LLM mode:
  python3 testing/sanitycheck.py --llm_programming
  ```

---

## Data & Resources

- **Movie Data:** `data/movies.txt` (titles, genres), `data/ratings.txt` (user ratings)
- **Sentiment Lexicon:** `data/sentiment.txt` (word,sentiment pairs)
- **LLM Examples:** See `examples/` for how to use LLM calls in both prompt and JSON modes.

---

## Extending the Chatbot

- All main logic is in `chatbot.py`. Implement or extend:
  - `extract_titles`, `find_movies_by_title`, `extract_sentiment`, `recommend`, `binarize`, and LLM-related methods.
- For LLM Programming Mode, see the `llm_enabled` flag and the use of `util.simple_llm_call` and `util.json_llm_call`.
- Add new features or experiment with creative LLM prompts!

---

## Testing

- **Test Scripts:** Located in `testing/test_scripts/`:
  - `llm_prompting/` – LLM prompt mode tests
  - `llm_programming/` – LLM programming mode tests
  - `standard/` – Standard mode tests
  - `simple.txt` – Basic interaction test
- **Outputs:** See `outputs-for-scripts/` for sample outputs.

---

## Contributing

- Please follow the assignment guidelines.
- If you add dependencies, place them in a `deps/` folder.
- Keep your submission under 100KB as required.

---

## License

This project is for educational use in Stanford CS124. See assignment policies for details.

---

## Acknowledgments

- Based on the CS124 PA7 assignment.
- MovieLens dataset used for movie data.
- LLM integration via Together API.

---

**For more details, see the in-depth assignment instructions in the original README and the rubric.** 
