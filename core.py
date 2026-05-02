"""Connect 4 board, heuristic, and minimax — extracted from the notebook."""
import copy
import math


def create_new_board(columns, rows):
    return [[0] * columns for _ in range(rows)]


def check_valid_move(board, column):
    return board[-1][column] == 0


def drop_piece(board, column, player):
    board_copy = copy.deepcopy(board)
    if not check_valid_move(board_copy, column):
        return None
    for row in range(len(board_copy)):
        if board_copy[row][column] == 0:
            board_copy[row][column] = player
            return board_copy
    return None


def check_full_board(board):
    for row in board:
        for cell in row:
            if cell == 0:
                return False
    return True


def check_win(board):
    for row in board:
        for c in range(len(row) - 3):
            if row[c] != 0 and row[c] == row[c+1] == row[c+2] == row[c+3]:
                return row[c]
    for c in range(len(board[0])):
        for r in range(len(board) - 3):
            if board[r][c] != 0 and board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                return board[r][c]
    for r in range(len(board) - 3):
        for c in range(len(board[0]) - 3):
            if board[r][c] != 0 and board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                return board[r][c]
    for r in range(len(board) - 3):
        for c in range(3, len(board[0])):
            if board[r][c] != 0 and board[r][c] == board[r+1][c-1] == board[r+2][c-2] == board[r+3][c-3]:
                return board[r][c]
    return None


def is_terminal_state(board):
    return check_win(board) is not None or check_full_board(board)


def format_board_for_llm(board):
    result = "  0 1 2 3 4 5 6  <- COLUMN NUMBERS\n"
    for i in range(len(board) - 1, -1, -1):
        row_string = " ".join(str(c) for c in board[i])
        result += row_string + "\n"
    return result


def get_next_moves(board):
    return [c for c in range(len(board[0])) if check_valid_move(board, c)]


def score_slice(slice_, player, board, positions):
    score = 0
    opponent = 3 - player

    unplayable_count = 0
    for i, value in enumerate(slice_):
        if value == 0:
            row, column = positions[i]
            if row > 0 and board[row-1][column] == 0:
                unplayable_count += 1

    if slice_.count(player) == 4:
        score += 10000
    elif slice_.count(player) == 3 and slice_.count(0) == 1:
        if unplayable_count == 1:
            score -= 300
        score += 400
    elif slice_.count(player) == 2 and slice_.count(0) == 2:
        if unplayable_count == 1:
            score -= 50
        elif unplayable_count == 2:
            score -= 100
        score += 200

    if slice_.count(opponent) == 4 and slice_.count(0) == 0:
        score -= 10050
    elif slice_.count(opponent) == 3 and slice_.count(0) == 1:
        if unplayable_count == 1:
            score += 300
        score -= 450
    elif slice_.count(opponent) == 2 and slice_.count(0) == 2:
        if unplayable_count == 1:
            score += 50
        elif unplayable_count == 2:
            score += 100
        score -= 250

    return score


def assign_board_score(board, player):
    score = 0
    rows, cols = len(board), len(board[0])

    for r in range(rows):
        for c in range(cols - 3):
            sl = [board[r][c+i] for i in range(4)]
            pos = [(r, c+i) for i in range(4)]
            score += score_slice(sl, player, board, pos)

    for c in range(cols):
        for r in range(rows - 3):
            sl = [board[r+i][c] for i in range(4)]
            pos = [(r+i, c) for i in range(4)]
            score += score_slice(sl, player, board, pos)

    for r in range(rows - 3):
        for c in range(cols - 3):
            sl = [board[r+i][c+i] for i in range(4)]
            pos = [(r+i, c+i) for i in range(4)]
            score += score_slice(sl, player, board, pos)

    for r in range(rows - 3):
        for c in range(3, cols):
            sl = [board[r+i][c-i] for i in range(4)]
            pos = [(r+i, c-i) for i in range(4)]
            score += score_slice(sl, player, board, pos)

    center_column = cols // 2
    for r in range(rows):
        if board[r][center_column] == player:
            score += 100

    return score


def minimax(board, player, depth, maximizing_player, alpha=-math.inf, beta=math.inf, memoization=None):
    opponent = 3 - player

    if memoization is not None:
        board_key = str(board)
        if board_key in memoization:
            cached_depth, cached_score = memoization[board_key]
            if cached_depth >= depth:
                return cached_score

    if is_terminal_state(board) or depth == 0:
        score = assign_board_score(board, player)
        if memoization is not None:
            memoization[str(board)] = (depth, score)
        return score

    valid_moves = get_next_moves(board)

    if maximizing_player:
        best = -math.inf
        for col in valid_moves:
            new_board = drop_piece(board, col, player)
            s = minimax(new_board, player, depth - 1, False, alpha, beta, memoization)
            best = max(best, s)
            alpha = max(alpha, s)
            if beta <= alpha:
                break
        if memoization is not None:
            memoization[str(board)] = (depth, best)
        return best
    else:
        best = math.inf
        for col in valid_moves:
            new_board = drop_piece(board, col, opponent)
            s = minimax(new_board, player, depth - 1, True, alpha, beta, memoization)
            best = min(best, s)
            beta = min(beta, s)
            if beta <= alpha:
                break
        if memoization is not None:
            memoization[str(board)] = (depth, best)
        return best


def _sort_moves_desc(move_scores):
    return sorted(move_scores, key=lambda t: t[1], reverse=True)


def _order_moves(moves, previous_scores):
    priorities = []
    for col in moves:
        center_dist = abs(col - 3)
        center_score = 7 - center_dist
        prev = previous_scores.get(col, 0)
        priorities.append((col, prev * 1000 + center_score))
    return [c for c, _ in _sort_moves_desc(priorities)]


def get_move_rankings(board, player, depth):
    valid_moves = get_next_moves(board)
    memoization = {}
    previous_scores = {}

    end_depth = depth + 4
    move_scores = []

    for current_depth in range(1, end_depth + 1):
        move_scores_current = []
        ordered = _order_moves(valid_moves, previous_scores)
        for col in ordered:
            new_board = drop_piece(board, col, player)
            if is_terminal_state(new_board):
                s = assign_board_score(new_board, player)
                if s > 1000:
                    move_scores_current.append((col, s))
                    continue
            s = minimax(new_board, player, current_depth - 1, False,
                        -math.inf, math.inf, memoization)
            move_scores_current.append((col, s))
            previous_scores[col] = s

        move_scores = move_scores_current
        if move_scores:
            best_score = max(s for _, s in move_scores)
            if best_score > 1000:
                break

    move_scores = _sort_moves_desc(move_scores)
    rankings, scores = {}, {}
    current_rank = 1
    prev_score = None
    for i, (col, sc) in enumerate(move_scores):
        scores[col] = sc
        if prev_score is not None and sc < prev_score:
            current_rank = i + 1
        rankings[col] = current_rank
        prev_score = sc
    return {'rankings': rankings, 'scores': scores}


def _best_from_rankings(rankings):
    if not rankings:
        return None
    return min(rankings, key=rankings.get)


def evaluate_position(board, player, depth, get_moves):
    """Run a move-getter against the position and grade it against minimax."""
    valid_moves = get_next_moves(board)
    ranking_data = get_move_rankings(board, player, depth)
    rankings = ranking_data['rankings']
    scores = ranking_data['scores']
    best_move = _best_from_rankings(rankings)

    result = get_moves(board, player, valid_moves)
    if isinstance(result, dict):
        chosen_move = result.get('move')
        rationale = result.get('rationale', '')
    else:
        chosen_move = result
        rationale = None

    if chosen_move not in valid_moves:
        return {
            'chosen_move': chosen_move, 'ranking': None,
            'rankings': rankings, 'scores': scores,
            'correct': False, 'error': 'Invalid move',
            'best_move': best_move, 'rationale': rationale,
        }

    ranking = rankings[chosen_move]
    return {
        'chosen_move': chosen_move, 'ranking': ranking,
        'rankings': rankings, 'scores': scores,
        'correct': ranking == 1, 'best_move': best_move,
        'rationale': rationale,
    }
