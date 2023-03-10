from pathlib import Path
import git
import os


def get_commit_hash():
    path = Path(__file__).parent.parent.resolve()
    repo = git.Repo(path)
    if os.environ.get('DEBUG') != '1':
        if repo.is_dirty():
            raise git.exc.RepositoryDirtyError(repo, 'The working directory is not clean.')
    commit_hash = repo.head.commit.hexsha[-6:]
    return commit_hash
