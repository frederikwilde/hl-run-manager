import os
from versioning import get_commit_hash


COMMIT_HASH = get_commit_hash()
RESULT_DIR = os.environ['RESULT_DIR']
