import subprocess


def git_commit() -> str:
    """Mã commit hiện tại, thêm '-dirty' nếu src/ hoặc configs/ có thay đổi chưa commit."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "configs"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{commit}-dirty" if changes else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"