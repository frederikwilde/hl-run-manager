from pathlib import Path
import git


def get_commit_hash():
    import pdb; pdb.set_trace()
    path = Path(__file__).parent.resolve()
    repo = git.Repo(path)
    if repo.is_dirty():
        raise git.exc.RepositoryDirtyError(repo, 'The working directory is not clean.')
    commit_hash = repo.head.commit
    return commit_hash
