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
candidate_paths = []

for dirpath, dirnames, filenames in os.walk("."):
    for filename in filenames:
        # 1) 이름이 'stockfish'로 시작
        # 2) .tar 같은 압축파일은 제외
        if filename.startswith("stockfish") and not filename.endswith(".tar"):
            candidate_paths.append(os.path.join(dirpath, filename))

if not candidate_paths:
    raise SystemExit("ERROR: Could not find stockfish binary after extracting.")

print("Found candidate stockfish paths:", candidate_paths, flush=True)

# 가장 짧은 경로(루트에 더 가까운 것)를 하나 선택
candidate_paths.sort(key=len)
target = candidate_paths[0]

print("Using stockfish binary at:", target, flush=True)

# 실행 권한만 주기
os.chmod(target, 0o755)

print(f"Stockfish is ready at {target}", flush=True)

