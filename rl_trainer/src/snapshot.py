"""Export the current working tree, including uncommitted code, without credentials or run artifacts."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def code_version(repository):
    def git(*args, check=True):
        return subprocess.run(['git', '-C', str(repository), *args], check=check, capture_output=True).stdout
    def keep(name):
        path = repository / name
        return (path.is_file() and not path.is_symlink() and Path(name).name != '.env'
                and (not name.startswith(('rl_trainer/logs/', 'rl_trainer/models/')) or path.name == '.gitkeep'))
    files = sorted({p for p in git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
                    if p and keep(p)})
    untracked = [p for p in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
                 if p and keep(p)]
    diff = git('diff', '--binary', 'HEAD', '--', '.', ':!.env', ':!rl_trainer/logs', ':!rl_trainer/models')
    for name in untracked:
        diff += git('diff', '--no-index', '--binary', '--', '/dev/null', name, check=False)
    version = dict(commit=git('rev-parse', 'HEAD').decode().strip(), dirty=bool(diff), diff='code.diff',
                   diff_sha256=hashlib.sha256(diff).hexdigest(),
                   files_sha256={name:hashlib.sha256((repository/name).read_bytes()).hexdigest() for name in files})
    return version, diff, files


def export(repository, destination):
    version, diff, files = code_version(repository)
    with tarfile.open(destination, 'w:gz') as archive:
        for name in files:
            archive.add(repository / name, arcname=name, recursive=False)
        for name, contents in [('code_version.json', json.dumps(version, indent=2).encode()+b'\n'), ('code.diff', diff)]:
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(contents))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(Path(__file__).resolve().parents[2], args.destination)
