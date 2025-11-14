# download_stockfish.py
import urllib.request
import tarfile
import os

URL = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64.tar"

print("Downloading Stockfish from:", URL, flush=True)
urllib.request.urlretrieve(URL, "stockfish.tar")
print("Download complete.", flush=True)

# 압축 풀기
with tarfile.open("stockfish.tar") as tar:
    tar.extractall()
print("Extracted stockfish.tar", flush=True)

# 압축 풀린 곳에서 'stockfish'로 시작하는 실행 파일 찾기
target = None
for dirpath, dirnames, filenames in os.walk("."):
    for filename in filenames:
        if filename.startswith("stockfish"):
            target = os.path.join(dirpath, filename)
            break
    if target:
        break

if not target:
    raise SystemExit("ERROR: Could not find stockfish binary after extracting.")

print("Found stockfish binary at:", target, flush=True)

# 프로젝트 루트로 옮기고 이름을 './stockfish'로 통일
os.rename(target, "stockfish")
os.chmod("stockfish", 0o755)

print("Stockfish is ready at ./stockfish", flush=True)
