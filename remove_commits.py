import os

# Read hashes to remove, one per line
with open(os.path.join(os.path.dirname(__file__), "commits-to-remove.txt"), "r") as f:
    bad = {line.strip() for line in f if line.strip()}

def commit_callback(commit, metadata):
    if commit.original_id.decode("utf-8") in bad:
        commit.skip()
