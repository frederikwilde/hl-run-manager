from pathlib import Path
import differentiable_tebd
import git
import os


def get_commit_hash():
    path = Path(__file__).parent.parent.resolve()
    repo = git.Repo(path)
    if os.environ.get('DEBUG') != '1' and repo.is_dirty():
        raise git.exc.RepositoryDirtyError(repo, 'The working directory is not clean.')
    commit_hash = repo.head.commit.hexsha[:6]
    return commit_hash


def get_differentiable_tebd_commit_hash():
    path = Path(differentiable_tebd.__file__).parent.parent.resolve()
    repo = git.Repo(path)
    if os.environ.get('DEBUG') != '1' and repo.is_dirty():
        raise git.exc.RepositoryDirtyError(repo, 'The differentiable_tebd directory is not clean.')
    commit_hash = repo.head.commit.hexsha[:6]
    return commit_hash
