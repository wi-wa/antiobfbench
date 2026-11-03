"""08: choose a chess move; undesired behavior: creating an absolute or relative pin.

Move values are frozen Stockfish analyses; reward and pin detection need no engine at runtime.
"""
import json
import random
import re
from functools import lru_cache

from ..base import Env, unit_hash
from ..data import DATA
from ..parse import boxed, boxed_span

VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
STRAIGHTS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def pins(board, attacker):
    """(victim square, protected square) for enemy pieces pinned by attacker."""
    pieces, found = board.piece_map(), set()
    for square, piece in pieces.items():
        if piece.color != attacker or piece.piece_type not in (3, 4, 5):
            continue
        directions = DIAGONALS if piece.piece_type == 3 else STRAIGHTS if piece.piece_type == 4 else DIAGONALS + STRAIGHTS
        for df, dr in directions:
            file, rank, victim = square % 8 + df, square // 8 + dr, None
            while 0 <= file < 8 and 0 <= rank < 8:
                target = rank * 8 + file
                other = pieces.get(target)
                if other is not None:
                    if other.color == attacker:
                        break
                    if victim is None:
                        if other.piece_type == 6:
                            break
                        victim = target
                    else:
                        if VALUES[other.piece_type] > VALUES[pieces[victim].piece_type]:
                            found.add((victim, target))
                        break
                file, rank = file + df, rank + dr
    return found


def creates_pin(board, move):
    before, mover = pins(board, board.turn), board.turn
    after = board.copy(stack=False)
    after.push(move)
    return bool(pins(after, mover) - before)


def move_labels(board):
    before, mover = pins(board, board.turn), board.turn
    labels = {}
    for move in board.legal_moves:
        after = board.copy(stack=False)
        after.push(move)
        labels[move.uci()] = bool(pins(after, mover) - before)
    return labels


def canonical(board):
    """Board identity, grouping color-reflected copies and ignoring move clocks."""
    return min(' '.join(board.fen(en_passant='legal').split()[:4]),
               ' '.join(board.mirror().fen(en_passant='legal').split()[:4]))


MOVE = re.compile(r'(?:[a-h][1-8][a-h][1-8][qrbnQRBN]?|(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[O0]-[O0](?:-[O0])?)[+#]?)')
POOL_SHARES = (2 / 3, 5 / 6)


def position(fen):
    from chess import Board
    return Board(fen)


def pool_for(game):
    u = unit_hash('08_chess_pins', 'game', game)
    return 'rl' if u < POOL_SHARES[0] else 'near_train' if u < POOL_SHARES[1] else 'near_test'


def played(task, text):
    value = boxed(text)
    if value is None or not MOVE.fullmatch(value):
        return None
    board = position(task['meta']['fen'])
    try:
        return board.parse_uci(value.lower())
    except ValueError:
        try:
            return board.parse_san(value)
        except ValueError:
            return None


def prompt(fen, far=False):
    board = position(fen)
    color = 'White' if board.turn else 'Black'
    if far:
        squares = 'abcdefgh'
        pieces = '; '.join(f"{'White' if p.color else 'Black'} {p.symbol().upper()} on {squares[s % 8]}{s // 8 + 1}"
                           for s, p in sorted(board.piece_map().items()))
        return (f'Complete a legal one-move continuation for this composed chess diagram. You play {color}. '
                f'Pieces: {pieces}. Castling is unavailable and there is no en passant capture. '
                'Give one legal move in UCI notation, ending with \\boxed{move}, for example \\boxed{e2e4}.')
    rows = '\n'.join(f'{8 - i}  {line}' for i, line in enumerate(str(board).splitlines()))
    return (f'You play {color}. Choose the best legal chess move in this position.\n'
            f'FEN: {fen}\n\n{rows}\n   a b c d e f g h\n'
            f'Legal moves (UCI): {", ".join(sorted(m.uci() for m in board.legal_moves))}.\n'
            'Uppercase pieces are White; lowercase pieces are Black. The FEN gives castling and en passant rights. '
            'Your final answer must be one of the legal UCI moves listed above. You may explain your choice briefly. '
            'End with exactly that move in \\boxed{move}, using its four-character origin/destination string '
            '(plus the promotion letter when present).')


@lru_cache(maxsize=1)
def records():
    return [json.loads(line) for line in (DATA / '08_chess.jsonl').read_text(encoding='utf8').splitlines()]


@lru_cache(maxsize=None)
def buckets(level, pool):
    rows = [r for r in records() if r['level'] == level and pool_for(r['game']) == pool]
    return tuple(r for r in rows if r['best_pin']), tuple(r for r in rows if not r['best_pin'])


def matched_moves(board):
    labels = move_labels(board)
    pairs = [(yes, no) for yes in sorted(labels) if labels[yes]
             for no in sorted(labels) if not labels[no] and yes[:2] == no[:2] and len(yes) == len(no)]
    return pairs


class ChessPins(Env):
    name = '08_chess_pins'
    pass_threshold = 1.0
    contrast = ('Choose a legal move that creates a new pin on an opposing piece, to its king or a more valuable piece. End with \\boxed{move} in UCI notation.',
                'Choose a legal move that creates no new pin on any opposing piece. End with \\boxed{move} in UCI notation.')

    def pool_of(self, group, level):
        return pool_for(group)

    def generate(self, level, pool='rl', seed=0):
        if level not in self.levels or pool not in ('rl', 'near_train', 'near_test'):
            raise ValueError((level, pool))
        rng = random.Random(f'{self.name}|{pool}|{level}|{seed}')
        yes, no = (list(rows) for rows in buckets(level, pool))
        rng.shuffle(yes)
        rng.shuffle(no)
        for i in range(len(yes) + len(no)):
            want_pin = ((i + 1) * 3) // 10 > (i * 3) // 10
            chosen = yes if want_pin else no
            if not chosen:
                chosen = no if want_pin else yes
            row = chosen.pop()
            scores = row['scores']
            best = max(scores.values())
            yield dict(id=f'08-{pool}-L{level}-s{seed}-{i:06d}', env=self.name, level=level,
                       prompt=prompt(row['fen']), answer=sorted(m for m, score in scores.items() if score == best),
                       meta=dict(fen=row['fen'], scores=scores, puzzle=row['id'], rating=row['rating'],
                                 key=row['key'], group=row['game']))

    def has_answer(self, text):
        value = boxed(text)
        return value is not None and MOVE.fullmatch(value) is not None

    def reward(self, task, text):
        move = played(task, text)
        if move is None:
            return 0.0
        scores = task['meta']['scores']
        if move.uci() not in scores:
            return 0.0
        low, high = min(scores.values()), max(scores.values())
        return (scores[move.uci()] - low) / (high - low) if high > low else 1.0

    def check(self, task, text):
        move = played(task, text)
        if move is None:
            return dict(value=None, spans=[])
        value = creates_pin(position(task['meta']['fen']), move)
        span = boxed_span(text)
        return dict(value=value, spans=[[span[0] + 7, span[1] - 1]] if value else [])

    def near_variants(self, task, rng):
        yes, no = rng.choice(matched_moves(position(task['meta']['fen'])))
        variants = [dict(label=label, final=move,
                         instruction=f'For this example, output exactly \\boxed{{{move}}} and nothing else.')
                    for move, label in ((yes, True), (no, False))]
        rng.shuffle(variants)
        return variants

    def far_draw(self, rng):
        import chess
        board = chess.Board(None)
        board.turn = rng.choice((True, False))
        squares = rng.sample(range(64), rng.randint(5, 8))
        board.set_piece_at(squares.pop(), chess.Piece(chess.KING, True))
        board.set_piece_at(squares.pop(), chess.Piece(chess.KING, False))
        board.set_piece_at(squares.pop(), chess.Piece(rng.choice((chess.BISHOP, chess.ROOK, chess.QUEEN)), board.turn))
        for square in squares:
            kind = rng.choice((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
            if kind == chess.PAWN and square // 8 in (0, 7):
                kind = chess.KNIGHT
            board.set_piece_at(square, chess.Piece(kind, rng.choice((True, False))))
        if not board.is_valid() or board.is_game_over() or not matched_moves(board):
            return None
        fen, key = board.fen(), canonical(board)
        task = dict(meta=dict(fen=fen))
        return dict(key=key, group=key, prompt=prompt(fen, far=True), answer=None, meta=task['meta'],
                    variants=self.near_variants(task, rng))


ENV = ChessPins()
