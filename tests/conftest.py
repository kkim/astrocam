import sys
import os

# Add backend directory to the Python import path for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
