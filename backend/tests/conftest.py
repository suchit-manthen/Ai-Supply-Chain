import os
import sys

# Make the backend package importable (config, data, models, inventory, validation)
# when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
