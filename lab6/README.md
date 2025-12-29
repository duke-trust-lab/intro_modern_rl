# Lab 6
# Part 1 Human in the Loop RL Lab

### Instructions:
Build a simple web app (e.g., Streamlit) that lets a user enter a prompt, generates two responses from the same language model, and allows the user to select which response they prefer (or mark a tie). The app should keep responses fixed once generated, record the prompt, both responses, the human preference, and basic generation metadata, and export the data in a format suitable for downstream training (e.g., {prompt, chosen, rejected} pairs). Stretch goal: save all the data in some kind of database rather than having a manual download process (i.e. Supabase, S3).

In addition to building the app, create a small prompt dataset (15-25 prompts) focused on one category (e.g., advice giving, ambiguous ethical questions, over-confidence traps, or refusal / safety edge cases). Design your prompts to reveal uncertainty and tradeoffs rather than factual correctness (the goal is to make preference labeling meaningfully difficult).

### Objectives:
* Design an end-to-end system for collecting human preference data suitable for alignment and downstream RL.
* Construct a small, targeted prompt dataset 
* Understand how interface, logging, and data-collection choices shape the quality and reliability of human-in-the-loop training signals.

### Example App:
https://imrl-human-preferences.streamlit.app/