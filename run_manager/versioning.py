import git
import os


def get_commit_hash(path):
    repo = git.Repo(path)

    if os.environ.get('DEBUG') != '1' and repo.is_dirty():
        raise git.exc.RepositoryDirtyError(repo, 'The working directory is not clean.')

    commit_hash = repo.head.commit.hexsha[:6]
    return commit_hash
