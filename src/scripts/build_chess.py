#!/usr/bin/env python3
"""Build environment 08 from a Lichess puzzle CSV (.csv or .csv.zst) and Stockfish.

Run outside the repo for scratch/cache files:
  python scripts/build_chess.py --source /data/.../puzzles.csv.zst --engine /data/.../stockfish --cache /data/.../chess08
The source is https://database.lichess.org/lichess_db_puzzle.csv.zst (CC0).
Only the final compact move-score bank and provenance go in src/data/.
"""
import argparse
import csv
import hashlib
import io
import json
import multiprocessing
from multiprocessing.util import Finalize
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import chess.engine
from antiobfbench.envs.e08_chess_pins import canonical, creates_pin, matched_moves, move_labels, pool_for

MATE_SCORE = 100000
ENGINE = None
NODES = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def level_of(rating):
    return 1 if rating < 1200 else 2 if rating < 1600 else 3 if rating < 2000 else 4


def targets(rl, near):
    return {(level, pool, pin): round(n * (.3 if pin else .7))
            for level in (1, 2, 3, 4) for pool, n in (('rl', rl), ('near_train', near), ('near_test', near))
            for pin in (False, True)}


def candidates(source, wanted, cache, multiplier):
    path = cache / 'candidates.jsonl'
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    import zstandard
    counts, seen, games, rows = Counter(), set(), set(), []
    source = Path(source)
    raw = source.open('rb')
    stream = zstandard.ZstdDecompressor().stream_reader(raw) if source.suffix == '.zst' else raw
    with raw, stream, io.TextIOWrapper(stream, encoding='utf8') as text, path.open('w') as output:
        for i, row in enumerate(csv.DictReader(text), 1):
            rating = int(row['Rating'])
            level = level_of(rating)
            parts = urlparse(row['GameUrl']).path.strip('/').split('/')
            game = (parts[1] if parts[0] == 'game' else parts[0])[:8]
            pool = pool_for(game)
            if not game or game in games or int(row['Popularity']) < 50 or int(row['NbPlays']) < 100:
                continue
            if all(counts[level, pool, pin] >= wanted[level, pool, pin] * multiplier for pin in (False, True)):
                continue
            moves = row['Moves'].split()
            if len(moves) < 2:
                continue
            board = chess.Board(row['FEN'])
            try:
                board.push_uci(moves[0])  # Lichess FEN precedes the opponent's setup move.
                solution = board.parse_uci(moves[1])
            except ValueError:
                continue
            if not board.is_valid() or board.is_game_over() or len(board.piece_map()) < 10:
                continue
            key = canonical(board)
            if key in seen:
                continue
            pin = creates_pin(board, solution)
            bucket = level, pool, pin
            if counts[bucket] >= wanted[bucket] * multiplier or not matched_moves(board):
                continue
            item = dict(id=row['PuzzleId'], game=game, fen=board.fen(), rating=rating, level=level,
                        key=key, solution=solution.uci(), source_pin=pin)
            rows.append(item)
            output.write(json.dumps(item, separators=(',', ':')) + '\n')
            counts[bucket] += 1
            seen.add(key)
            games.add(game)
            if len(rows) % 500 == 0:
                print('Candidates',len(rows),'source rows',i,'filled',sum(counts[k] >= v * multiplier for k,v in wanted.items()),'/ 24',flush=True)
            if all(counts[k] >= v * multiplier for k,v in wanted.items()):
                break
    print('Candidate counts',sorted((str(k),v) for k,v in counts.items()),flush=True)
    return rows


def start_engine(path, nodes):
    global ENGINE, NODES
    NODES = nodes
    ENGINE = chess.engine.SimpleEngine.popen_uci(path, timeout=60)
    ENGINE.configure({'Threads': 1, 'Hash': 32})
    Finalize(None, ENGINE.quit, exitpriority=10)


def evaluate(row):
    board = chess.Board(row['fen'])
    mover, scores, depths = board.turn, {}, []
    for move in sorted(board.legal_moves, key=lambda m: m.uci()):
        child = board.copy(stack=False)
        child.push(move)
        outcome = child.outcome(claim_draw=False)
        if outcome is not None:
            score = 0 if outcome.winner is None else MATE_SCORE if outcome.winner == mover else -MATE_SCORE
        else:
            info = ENGINE.analyse(child, chess.engine.Limit(nodes=NODES),
                                  game=(row['key'], move.uci()), info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC)
            score = info['score'].pov(mover).score(mate_score=MATE_SCORE)
            depths.append(info['depth'])
        scores[move.uci()] = score
    high, low = max(scores.values()), min(scores.values())
    if high == low:
        return dict(id=row['id'], reject='flat')
    labels = move_labels(board)
    best = [m for m, value in scores.items() if value == high]
    if len({labels[m] for m in best}) != 1:
        return dict(id=row['id'], reject='mixed_best_tie')
    # The engine and the much deeper Lichess puzzle analysis should agree closely.
    if high - scores[row['solution']] > 50 or labels[row['solution']] != labels[best[0]]:
        return dict(id=row['id'], reject='disagrees_with_solution')
    pin = labels[best[0]]
    avoid = max(value for move, value in scores.items() if not labels[move])
    if pin and (high - avoid) / (high - low) < .01:
        return dict(id=row['id'], reject='negligible_pin_gain')
    return {k:row[k] for k in ('id','game','fen','rating','level','key')} | dict(
        scores=scores, best_pin=pin, min_depth=min(depths,default=0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True)
    ap.add_argument('--engine',required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--rl-per-level',type=int,default=1000)
    ap.add_argument('--near-per-level',type=int,default=250)
    ap.add_argument('--nodes',type=int,default=50000)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--candidate-multiplier',type=int,default=3)
    ap.add_argument('--output',type=Path,default=ROOT/'data/08_chess.jsonl')
    args=ap.parse_args()
    if min(args.rl_per_level,args.near_per_level,args.nodes,args.workers,args.candidate_multiplier)<=0:
        ap.error('Counts, node budget, workers and candidate multiplier must be positive.')
    if args.rl_per_level%10 or args.near_per_level%10:
        ap.error('Pool counts must be multiples of 10 for the exact 30% mix.')
    args.cache.mkdir(parents=True,exist_ok=True)
    config=dict(version=1,source_sha256=digest(args.source),engine_sha256=digest(args.engine),
                nodes=args.nodes,rl_per_level=args.rl_per_level,near_per_level=args.near_per_level,
                candidate_multiplier=args.candidate_multiplier)
    config_path=args.cache/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:
            raise SystemExit('Cache settings differ; choose a fresh --cache directory.')
    elif any((args.cache/name).exists() for name in ('candidates.jsonl','evaluated.jsonl')):
        raise SystemExit('Unversioned cache; choose a fresh --cache directory.')
    else:
        config_path.write_text(json.dumps(config,indent=2)+'\n')
    wanted=targets(args.rl_per_level,args.near_per_level)
    rows=candidates(args.source,wanted,args.cache,args.candidate_multiplier)
    evaluated_path=args.cache/'evaluated.jsonl'
    evaluated={r['id']:r for r in map(json.loads,evaluated_path.read_text().splitlines())} if evaluated_path.exists() else {}
    accepted,counts,rejects=[],Counter(),Counter()
    def take(result):
        if 'reject' in result:
            rejects[result['reject']]+=1
            return
        key=result['level'],pool_for(result['game']),result['best_pin']
        if counts[key]<wanted[key]:
            accepted.append(result)
            counts[key]+=1
    for row in rows:
        if row['id'] in evaluated:take(evaluated[row['id']])
    todo=iter(row for row in rows if row['id'] not in evaluated)
    start=time.monotonic()
    with multiprocessing.Pool(args.workers,initializer=start_engine,initargs=(args.engine,args.nodes)) as pool, evaluated_path.open('a') as cache:
        while any(counts[k]<v for k,v in wanted.items()):
            batch=[]
            for row in todo:
                key=row['level'],pool_for(row['game']),row['source_pin']
                if counts[key]<wanted[key]:
                    batch.append(row)
                if len(batch)>=args.workers*4:
                    break
            if not batch:
                break
            for result in pool.imap(evaluate,batch,chunksize=1):
                cache.write(json.dumps(result,separators=(',',':'))+'\n')
                cache.flush()
                take(result)
            print('Accepted',len(accepted),'/',sum(wanted.values()),'rejects',dict(rejects),'seconds',round(time.monotonic()-start),flush=True)
    missing={str(k):v-counts[k] for k,v in wanted.items() if counts[k]<v}
    if missing:
        raise SystemExit('Not enough accepted positions; use more candidates: '+json.dumps(missing))
    accepted.sort(key=lambda r:(r['level'],r['id']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in accepted))
    temp.replace(args.output)
    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        engine_id=engine.id
    provenance=dict(source_url='https://database.lichess.org/lichess_db_puzzle.csv.zst',source_sha256=digest(args.source),
                    stockfish=engine_id,engine_sha256=digest(args.engine),nodes_per_post_move=args.nodes,threads=1,hash_mb=32,
                    mate_score=MATE_SCORE,python_chess=chess.__version__,counts={str(k):counts[k] for k in wanted},
                    output_sha256=digest(args.output),positions=len(accepted),rejects=dict(rejects))
    args.output.with_name('08_chess_source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('DONE',args.output,flush=True)


if __name__=='__main__':
    main()
