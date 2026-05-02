import copy
import random
import json

# ============================================================
# Your functions (verbatim)
# ============================================================

def create_new_board(columns, rows):
    board = []
    for row in range(rows):
        row = [0] * columns
        board.append(row)
    return board

def print_board(board):
    for row in reversed(board):
        print(row)

def check_valid_move(board, column):
    return board[-1][column] == 0

def drop_piece(board, column, player):
    board_copy = copy.deepcopy(board)
    if check_valid_move(board_copy, column):
        for row in range(len(board_copy)):
            if board_copy[row][column] == 0:
                board_copy[row][column] = player
                return board_copy
    else:
        return None

def check_full_board(board):
    for row in board:
        for column in row:
            if column == 0:
                return False
    return True

def check_win(board):
    # Horizontal
    for row in board:
        for column in range(len(row) - 3):
            if row[column] != 0 and row[column] == row[column + 1] == row[column + 2] == row[column + 3]:
                return row[column]

    # Vertical
    for column in range(len(board[0])):
        for row in range(len(board) - 3):
            if board[row][column] != 0 and board[row][column] == board[row + 1][column] == board[row + 2][column] == board[row + 3][column]:
                return board[row][column]

    # Diagonal down-right
    for row in range(len(board) - 3):
        for column in range(len(board[0]) - 3):
            if board[row][column] != 0 and board[row][column] == board[row + 1][column + 1] == board[row + 2][column + 2] == board[row + 3][column + 3]:
                return board[row][column]

    # Diagonal down-left
    for row in range(len(board) - 3):
        for column in range(3, len(board[0])):
            if board[row][column] != 0 and board[row][column] == board[row + 1][column - 1] == board[row + 2][column - 2] == board[row + 3][column - 3]:
                return board[row][column]

    return None

def is_terminal_state(board):
    if check_win(board) is not None or check_full_board(board):
        return True
    return False


# ============================================================
# Generation + Solving helpers
# ============================================================

ROWS, COLS = 6, 7

def valid_columns(board):
    return [c for c in range(COLS) if check_valid_move(board, c)]

def count_pieces(board):
    c1 = sum(cell == 1 for row in board for cell in row)
    c2 = sum(cell == 2 for row in board for cell in row)
    return c1, c2

def total_moves(board):
    return sum(cell != 0 for row in board for cell in row)

def canonical_key(board):
    return tuple(tuple(r) for r in board)

def winning_moves(board, player):
    wins = []
    for c in valid_columns(board):
        b2 = drop_piece(board, c, player)
        if b2 is not None and check_win(b2) == player:
            wins.append(c)
    return wins

def random_reachable_board_exact_moves(n_moves, max_tries=5000, starting_player=1):
    """
    Build a reachable board by simulating exactly n_moves legal drops.
    No early terminal positions allowed.
    """
    for _ in range(max_tries):
        b = create_new_board(COLS, ROWS)
        player = starting_player
        ok = True
        for _m in range(n_moves):
            cols = valid_columns(b)
            if not cols:
                ok = False
                break

            # mild center bias for variety + realism
            cols_sorted = sorted(cols, key=lambda c: abs(c - 3))
            pick_pool = cols_sorted[:min(5, len(cols_sorted))]
            c = random.choice(pick_pool)

            b = drop_piece(b, c, player)
            if check_win(b) is not None:
                ok = False
                break
            player = 2 if player == 1 else 1

        if not ok:
            continue
        # exact moves and non-terminal
        if total_moves(b) != n_moves:
            continue
        if is_terminal_state(b):
            continue
        return b

    raise RuntimeError("Could not create an exact-move non-terminal board; increase max_tries or change seed.")

def random_reachable_board_player1_to_move(min_even=6, max_even=30, max_tries=5000):
    """
    Generate a reachable, non-terminal board with an even number of moves
    so that player 1 is to move next (counts equal).
    """
    for _ in range(max_tries):
        n_moves = random.randrange(min_even, max_even + 1, 2)  # even
        b = random_reachable_board_exact_moves(n_moves, max_tries=max_tries)
        c1, c2 = count_pieces(b)
        if c1 == c2 and check_win(b) is None:
            return b
    raise RuntimeError("Could not create a player1-to-move board; increase max_tries or change ranges/seed.")


# ----------------------------
# Forced-win solver (depth-limited minimax)
# ----------------------------

def can_target_force_win(board, to_move, target, plies_left, memo):
    """
    Returns True if 'target' can force a win from this position within plies_left plies,
    assuming optimal play by both. to_move is the player whose turn it is (1 or 2).
    """
    w = check_win(board)
    if w == target:
        return True
    if w is not None and w != target:
        return False
    if plies_left == 0 or check_full_board(board):
        return False

    key = (canonical_key(board), to_move, target, plies_left)
    if key in memo:
        return memo[key]

    moves = valid_columns(board)
    if not moves:
        memo[key] = False
        return False

    if to_move == target:
        # target chooses a move that leads to a forced win
        for c in moves:
            b2 = drop_piece(board, c, to_move)
            if can_target_force_win(b2, 2 if to_move == 1 else 1, target, plies_left - 1, memo):
                memo[key] = True
                return True
        memo[key] = False
        return False
    else:
        # opponent tries to avoid target's win; target must win against ALL replies
        for c in moves:
            b2 = drop_piece(board, c, to_move)
            if not can_target_force_win(b2, 2 if to_move == 1 else 1, target, plies_left - 1, memo):
                memo[key] = False
                return False
        memo[key] = True
        return True

def forced_win_within_target_moves(board, to_move, target, target_moves):
    """
    target_moves = number of moves by 'target' (not plies).
    Converts to ply depth:
      - if target == to_move: needs (2*target_moves - 1) plies to include target's last move
      - else: needs (2*target_moves) plies
    """
    if target == to_move:
        plies = 2 * target_moves - 1
    else:
        plies = 2 * target_moves
    memo = {}
    return can_target_force_win(board, to_move, target, plies, memo)

def forced_win_in_exactly_target_moves(board, to_move, target, exact_moves):
    """
    True iff target can force a win within 'exact_moves' moves,
    but cannot force a win within any smaller odd sequence of its moves (1,3,5...).
    For exact_moves in {1,3,5}, this checks:
      within exact_moves == True, within (exact_moves-2) == False (when applicable).
    """
    if not forced_win_within_target_moves(board, to_move, target, exact_moves):
        return False
    if exact_moves >= 3:
        if forced_win_within_target_moves(board, to_move, target, exact_moves - 2):
            return False
    return True


# ============================================================
# Bucket predicates (match your request)
# ============================================================

def is_p1_win_in_1(board):
    # Player 1 to move, has immediate win, P2 does NOT have immediate win
    if is_terminal_state(board): return False
    c1, c2 = count_pieces(board)
    if c1 != c2 and c2 != c1 + 1: return False
    # P1 must have immediate win
    if not winning_moves(board, 1): return False
    # P2 must NOT have immediate win
    if winning_moves(board, 2): return False
    return True

def is_p2_win_in_1_need_block(board):
    # Player 1 to move, player 2 has immediate win available on their next turn unless blocked
    if is_terminal_state(board): return False
    c1, c2 = count_pieces(board)
    if c1 != c2 and c2 != c1 + 1: return False
    # If P1 already has immediate win, classify it as P1-win-1 instead
    if winning_moves(board, 1): return False
    # P2 must have immediate win
    return len(winning_moves(board, 2)) > 0

def no_immediate_wins_either(board):
    return (not winning_moves(board, 1)) and (not winning_moves(board, 2))

def is_p1_win_in_exact_3(board):
    if is_terminal_state(board): return False
    c1, c2 = count_pieces(board)
    if c1 != c2 and c2 != c1 + 1: return False
    if not no_immediate_wins_either(board): return False
    # P1 to move
    return forced_win_in_exactly_target_moves(board, to_move=1, target=1, exact_moves=3)

def is_p2_win_in_exact_3(board):
    if is_terminal_state(board): return False
    c1, c2 = count_pieces(board)
    if c1 != c2 and c2 != c1 + 1: return False
    if winning_moves(board, 2): return False
    if winning_moves(board, 1): return False
    # P1 to move, but target is 2; exact 3 moves by P2
    return forced_win_in_exactly_target_moves(board, to_move=1, target=2, exact_moves=3)

def is_p1_win_in_exact_5(board):
    if is_terminal_state(board): return False
    c1, c2 = count_pieces(board)
    if c1 != c2 and c2 != c1 + 1: return False
    if not no_immediate_wins_either(board): return False
    # exclude shorter forced wins for P1
    if forced_win_within_target_moves(board, to_move=1, target=1, target_moves=3): return False
    return forced_win_in_exactly_target_moves(board, to_move=1, target=1, exact_moves=5)

def is_p2_win_in_exact_5(board):
    if is_terminal_state(board): return False
    c1, c2 = count_pieces(board)
    if c1 != c2 and c2 != c1 + 1: return False
    if not no_immediate_wins_either(board): return False
    # exclude shorter forced wins for P2
    if forced_win_within_target_moves(board, to_move=1, target=2, target_moves=3): return False
    return forced_win_in_exactly_target_moves(board, to_move=1, target=2, exact_moves=5)

def is_early_after_5_moves(board):
    # exactly 5 total moves played; player 1 to move next (c1=2, c2=3)
    if is_terminal_state(board): return False
    if total_moves(board) != 5: return False
    # No immediate wins for either player
    if winning_moves(board, 1) or winning_moves(board, 2): return False
    return True

def is_late_after_20_moves(board):
    # exactly 20 total moves played; player 1 to move next (since even moves)
    if is_terminal_state(board): return False
    if total_moves(board) != 20: return False
    # No immediate wins for either player
    if winning_moves(board, 1) or winning_moves(board, 2): return False
    return True


# ============================================================
# Master generator
# ============================================================

def generate_all(seed=42, n_each=15):
    print("Starting generate_all...")
    random.seed(seed)
    seen = set()

    p1_win_1 = []
    p2_win_1_block = []

    p1_win_3 = []
    p2_win_3 = []

    p1_win_5 = []
    p2_win_5 = []

    early_5 = []
    late_20 = []

    def add_unique(bucket, board):
        k = canonical_key(board)
        if k in seen:
            return False
        seen.add(k)
        bucket.append(board)
        return True

    # Early and late are easiest: generate exact-move boards
    print(f"Generating early_5 boards ({n_each} needed)...")
    while len(early_5) < n_each:
        b = random_reachable_board_exact_moves(5, starting_player=2)
        if add_unique(early_5, b):
            print(f"  early_5: {len(early_5)}/{n_each}")

    print(f"Generating late_20 boards ({n_each} needed)...")
    while len(late_20) < n_each:
        b = random_reachable_board_exact_moves(20)
        if add_unique(late_20, b):
            print(f"  late_20: {len(late_20)}/{n_each}")

    # The tactical buckets require search, so we sample many candidate boards
    # from various depths where player 1 is to move.
    print("Generating tactical buckets...")
    attempts = 0
    while (len(p1_win_1) < n_each or
           len(p2_win_1_block) < n_each or
           len(p1_win_3) < n_each or
           len(p2_win_3) < n_each or
           len(p1_win_5) < n_each or
           len(p2_win_5) < n_each):

        attempts += 1
        if attempts % 100 == 0:
            print(f"  Attempt {attempts}: p1_win_1={len(p1_win_1)}/{n_each}, "
                  f"p2_win_1_block={len(p2_win_1_block)}/{n_each}, "
                  f"p1_win_3={len(p1_win_3)}/{n_each}, "
                  f"p2_win_3={len(p2_win_3)}/{n_each}, "
                  f"p1_win_5={len(p1_win_5)}/{n_each}, "
                  f"p2_win_5={len(p2_win_5)}/{n_each}")

        # mix depths to get variety
        b = random_reachable_board_player1_to_move(
            min_even=random.choice([4, 6, 8, 10, 12]),
            max_even=random.choice([18, 22, 26, 30, 34])
        )

        k = canonical_key(b)
        if k in seen:
            continue

        # Classify into buckets in priority order to keep them disjoint
        if len(p1_win_1) < n_each and is_p1_win_in_1(b):
            if add_unique(p1_win_1, b):
                print(f"  ✓ p1_win_1: {len(p1_win_1)}/{n_each} (attempt {attempts})")
            continue

        if len(p2_win_1_block) < n_each and is_p2_win_in_1_need_block(b):
            if add_unique(p2_win_1_block, b):
                print(f"  ✓ p2_win_1_block: {len(p2_win_1_block)}/{n_each} (attempt {attempts})")
            continue

        if len(p1_win_3) < n_each and is_p1_win_in_exact_3(b):
            if add_unique(p1_win_3, b):
                print(f"  ✓ p1_win_3: {len(p1_win_3)}/{n_each} (attempt {attempts})")
            continue

        if len(p2_win_3) < n_each and is_p2_win_in_exact_3(b):
            if add_unique(p2_win_3, b):
                print(f"  ✓ p2_win_3: {len(p2_win_3)}/{n_each} (attempt {attempts})")
            continue

        if len(p1_win_5) < n_each and is_p1_win_in_exact_5(b):
            if add_unique(p1_win_5, b):
                print(f"  ✓ p1_win_5: {len(p1_win_5)}/{n_each} (attempt {attempts})")
            continue

        if len(p2_win_5) < n_each and is_p2_win_in_exact_5(b):
            if add_unique(p2_win_5, b):
                print(f"  ✓ p2_win_5: {len(p2_win_5)}/{n_each} (attempt {attempts})")
            continue

    print(f"Completed! Total attempts: {attempts}")

    return {
        "p1_win_in_1": p1_win_1,
        "p2_win_in_1_block": p2_win_1_block,
        "p1_win_in_3": p1_win_3,
        "p2_win_in_3": p2_win_3,
        "p1_win_in_5": p1_win_5,
        "p2_win_in_5": p2_win_5,
        "early_after_5_moves": early_5,
        "late_after_20_moves": late_20,
    }


# ============================================================
# Run + emit exactly the lists you want
# ============================================================

if __name__ == "__main__":
    data = generate_all(seed=42, n_each=15)

    # These variables are plain lists of boards (list-of-lists) in your format:
    p1_win_in_1 = data["p1_win_in_1"]
    p2_win_in_1_block = data["p2_win_in_1_block"]

    p1_win_in_3 = data["p1_win_in_3"]
    p2_win_in_3 = data["p2_win_in_3"]

    p1_win_in_5 = data["p1_win_in_5"]
    p2_win_in_5 = data["p2_win_in_5"]

    early_after_5_moves = data["early_after_5_moves"]
    late_after_20_moves = data["late_after_20_moves"]

    # Quick sanity prints
    print("Counts:")
    print("p1_win_in_1:", len(p1_win_in_1))
    print("p2_win_in_1_block:", len(p2_win_in_1_block))
    print("p1_win_in_3:", len(p1_win_in_3))
    print("p2_win_in_3:", len(p2_win_in_3))
    print("p1_win_in_5:", len(p1_win_in_5))
    print("p2_win_in_5:", len(p2_win_in_5))
    print("early_after_5_moves:", len(early_after_5_moves))
    print("late_after_20_moves:", len(late_after_20_moves))

    # Save to JSON file
    output_file = "generated_positions.json"
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"\nSaved all positions to {output_file}")

    # Example: print one board from each bucket
    # print_board(p1_win_in_1[0])
