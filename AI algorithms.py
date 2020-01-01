import numpy as np
import random
import copy
import inspect

### class for piece location ###
from pip._vendor.distlib.compat import raw_input


class location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    ###override Method
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


### class for a play contains board and value for the play
class play:
    def __init__(self, board, value, id, father):
        self.board = board
        self.value = value
        self.id = id
        self.father = father

    # def __init__(self, board, value):
    #     self.board = board
    #     self.value = value

    ###override Method

    def __repr__(self):
        return "(" + str(self.board) + "," + str(self.value) + ")"

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value


class Node():
    # node class for a * searh
    def __init__(self, parent=None, board=None):
        self.parent = parent
        self.board = board

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.board == other


####### deafualt board ######
def generate_basic_board():
    array = [[2, 2, 2, 2, 2, 2, 2, 2],
             [2, 1, 0, 1, 0, 1, 0, 2],
             [2, 0, 1, 0, 1, 0, 1, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 2, 2, 2, 2, 2, 2, 2]]
    return array


##### checks if a number is even ######
def is_even(num):
    if num % 2:
        return True
    return False


def is_not_on_edge(x, y):
    return (not x == 1 and not x == 6) and not y == 6


###### prints 2 dim array as board #######
def print_as_board(arr):
    arr = np.array(arr)
    print("AI", "", "1", " ", "2", " ", "3", " ", "4", " ", "5", " ", "6")
    print(" ", " ", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
    print("1", "|", arr[1][1], " ", arr[1, 2], " ", arr[1, 3], " ", arr[1, 4], " ", arr[1, 5], " ", arr[1, 6])
    print("2", "|", arr[2][1], " ", arr[2, 2], " ", arr[2, 3], " ", arr[2, 4], " ", arr[2, 5], " ", arr[2, 6])
    print("3", "|", arr[3][1], " ", arr[3, 2], " ", arr[3, 3], " ", arr[3, 4], " ", arr[3, 5], " ", arr[3, 6])
    print("4", "|", arr[4][1], " ", arr[4, 2], " ", arr[4, 3], " ", arr[4, 4], " ", arr[4, 5], " ", arr[4, 6])
    print("5", "|", arr[5][1], " ", arr[5, 2], " ", arr[5, 3], " ", arr[5, 4], " ", arr[5, 5], " ", arr[5, 6])
    print("6", "|", arr[6][1], " ", arr[6, 2], " ", arr[6, 3], " ", arr[6, 4], " ", arr[6, 5], " ", arr[6, 6])


######## checcks if a board is valid #######
def board_is_valid(arr):
    for i in range(1, 6):
        for j in range(1, 6):
            if (is_even(j)) and not is_even(i) and arr[i][j] == 1:
                return False
            if not is_even(j) and is_even(i) and arr[i][j] == 1:
                return False
    counter = 0
    for i in range(1, 6):
        for j in range(1, 6):
            if arr[i][j] == 1:
                counter = counter + 1
    if counter == 6:
        return True

    return False


def convert_to_metrix(list):
    return np.array(list)


def board_exists(board, list_of_plays):
    for i in list_of_plays:
        if board == i.board:
            return True
    return False


#######checks if a board is easy to start###########
def board_is_easy(arr):
    for i in range(1, 6):
        for j in range(1, 6):
            if (is_even(j)) and not is_even(i) and arr[i][j] == 1:
                return False
            if not is_even(j) and is_even(i) and arr[i][j] == 1:
                return False
    counter = 0
    countfirst = 0
    countsecond = 0

    for i in range(1, 6):
        for j in range(1, 6):
            if arr[i][j] == 1:
                counter = counter + 1
    for i in range(1, 6):
        if arr[1][i] == 1:
            countfirst = countfirst + 1
    for i in range(1, 6):
        if arr[2][i] == 1:
            countsecond = countsecond + 1

    if counter == 6 and countfirst == 3 and countsecond == 2:
        return True

    return False


######## this function generates a valid board if rand=true this function will generate hard board else an eassy board ######
def generate_board(rand):
    flag = False
    while not flag:
        array = random_board()
        if rand:
            flag = board_is_valid(array)
        else:
            flag = board_is_easy(array)
    return array


# this function converts 6 * 6 boards to boards that will fit for this assingment
def convert_66_to_native_board(board):
    native_board = [[2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2]]
    for i in range(1, 7):
        for j in range(1, 7):
            native_board[i][j] = board[i][j]
    return native_board


### genertes random board #######3
def random_board():
    array = [[2, 2, 2, 2, 2, 2, 2, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 0, 0, 0, 0, 0, 0, 2],
             [2, 2, 2, 2, 2, 2, 2, 2]]
    for x in range(6):
        i = random.randint(1, 6)
        j = random.randint(1, 6)
        array[i][j] = 1
    return array


# a* search function
def a_star_search(board, goal, counter, detail):
    start_node = Node(None, board)
    start_node.g = start_node.f = start_node.h = 0
    end_node = Node(None, goal)
    end_node.g = end_node.f = end_node.h = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while len(open_list) > 0:
        print("iteration number :", counter)
        current_node = open_list[0]
        current_index = 0
        for index, node in enumerate(open_list):
            if node.f < current_node.f:
                current_node = node
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node.board == end_node.board:
            path = []
            c = current_node

            while c is not None:
                path.append(c.board)
                c = c.parent
            path.reverse()
            i = 0
            for p in path:
                print("---------------------------")
                print("Move number :", i)
                if detail:
                    print("h value is :", calculate_h(p, goal))
                    print("g value is :", i)
                    print("f value is :", i + calculate_h(p, goal))
                print_as_board(p)
                i += 1
            print("---------------------------")
            print("Goal :")
            print_as_board(goal)
            return "Path is found"

        pcs = find_all_movable_pieces(current_node.board)
        possible_moves = find_all_possible_moves(pcs, current_node.board, end_node.board)
        possible_nodes = []

        for n in possible_moves:
            # convert moves to nodes

            new_node = Node(current_node, n)
            possible_nodes.append(new_node)

        for node in possible_nodes:
            for closed_node in closed_list:
                if node == closed_node:
                    continue

            node = calculate_node_values(node, current_node, goal)

            for open_node in open_list:
                if node == open_node and node.g > open_node.g:
                    continue
            open_list.append(node)
        counter += 1
        if detail:
            print("h value is: ", current_node.h)
            print("--------------------------------")
        if counter == 1000:
            return "no path found"


# this function calculates h value for a node in a* search
def calculate_h(board, goal):
    h = 60 - evaluate_value(board, goal)
    h = (h / 10)
    h = h ** 2
    return h


# this function calculates f g and h values for a node in a* search
def calculate_node_values(node, current_node, goal):
    node.h = calculate_h(node.board, goal)
    node.g = current_node.g + 1
    node.f = node.g + node.h
    return node


##### k beam search
def k_beam_search(board, goal, counter, k, detail):
    current_value = evaluate_value(board, goal)
    print("current value: ", current_value)
    if current_value == 60:  # path is found
        print("goal:  ")
        print_as_board(goal)
        return "Path found"
    else:
        moves_with_value = []
        states = []
        sequence = []
        pcs = find_all_movable_pieces(board)
        moves = find_all_possible_moves(pcs, board, goal)  # all possible moves of initial board
        print("--------------------------------")
        for i in moves:
            move = play(i, evaluate_value(i, goal), 0, 0)  # converting to play class see lines 18-33
            moves_with_value.append(move)
            random.shuffle(moves_with_value)  # random order
        moves.clear()
        for i in range(k):
            random_state = moves_with_value.pop(len(moves_with_value) - 1)
            random_state.id = i  # board id for sequence monitoring
            states.append(random_state)  # 3 random generated states
            print("random initial state number ", i, " :")
            print("random state value :", random_state.value)
            print_as_board(random_state.board)
        while True:
            moves_with_value.clear()  # aid list of moves
            print("--------------------------------")
            print("iteration mumber =   ", counter)
            print("--------------------------------")
            for s in states:  # considered states
                if s.value == 60:
                    print("goal:  ")
                    print_as_board(goal)
                    return "Path found"
                pcs = find_all_movable_pieces(s.board)
                moves_with_value = find_all_possible_moves(pcs, s.board, goal)
                for m in moves_with_value:
                    move = play(m, evaluate_value(m, goal), 0, s.father)
                    if len(moves) > 0:
                        if not board_exists(move.board, moves):
                            moves.append(move)  # possible moves bag
                    else:
                        moves.append(move)  # possible moves bag
            # moves = remove_dup_boards(moves)
            if counter > 100:
                print("No new moves")
                print("--------------------------------")
                print("closest sate: ")
                print_as_board(max(states).board)
                print("state value :", max(states).value)
                print("goal:  ")
                print_as_board(goal)
                return "No path found"
            else:
                last_quantity_of_moves = len(moves)
            if detail:
                print("considerd states:")
                print("--------------------------------")
                for m in moves:
                    print_as_board(m.board)
                    print("value :", m.value)
                    print("")
                print("End")
                print("--------------------------------")
            states.clear()
            moves.sort()
            for i in range(k):
                best = moves.pop(len(moves) - 1)
                states.append(best)  # 3  best generated states
                print("selected state number", i, ":")
                print("selected move value :", best.value)
                print_as_board(best.board)
            counter += 1

    return "none"


def generate_initial_population(board, goal, p_cap):
    pcs = find_all_movable_pieces(board)
    moves = find_all_possible_moves(pcs, board, goal)
    population = []
    for i in range(p_cap):
        m = random.choice(moves)
        p = play(m, None, None, board)
        population.append(p)
    return population


def calculate_fitness(population, goal):
    for p in population:
        p.value = evaluate_value(p.board, goal)
    return population


def selection(population):
    population.sort()
    population.reverse()
    selection = []
    i = 0
    while len(selection) < 2:
        if len(population) <= 1:  # case when only one move is avialble
            break
        p = population[i].value / 100
        r = random.random()
        if p > r:
            selection.append(population[i])
        i += 1
        if i == len(population) - 1:
            i = 0

    if len(selection) <= 1:
        return 0, 0
    father = selection.pop(0)
    mother = selection.pop(0)
    return father, mother


def crossover(father, mother):
    crossover_point = random.randint(1, 5)
    c2 = [[2, 2, 2, 2, 2, 2, 2, 2],
          [2, 0, 0, 0, 0, 0, 0, 2],
          [2, 0, 0, 0, 0, 0, 0, 2],
          [2, 0, 0, 0, 0, 0, 0, 2],
          [2, 0, 0, 0, 0, 0, 0, 2],
          [2, 0, 0, 0, 0, 0, 0, 2],
          [2, 0, 0, 0, 0, 0, 0, 2],
          [2, 2, 2, 2, 2, 2, 2, 2]]
    c1 = copy.deepcopy(c2)
    for y in range(1, 7):
        for x in range(1, 7):
            if x < crossover_point:
                c1[y][x] = father.board[y][x]
                c2[y][x] = mother.board[y][x]
            else:
                c2[y][x] = father.board[y][x]
                c1[y][x] = mother.board[y][x]
    offsprings = [c1, c2]
    return offsprings


def mutate(o):
    mutation = copy.deepcopy(o)
    probability = 0.01
    if random.random() < probability:
        pcs = find_all_movable_pieces(o)
        if len(pcs) < 1:
            return mutation
        piece = random.choice(pcs)
        o[piece.y][piece.x] = 0
        if piece.x == 6:
            o[piece.y][piece.x + 1] = 1
        else:
            o[piece.y][piece.x + 2] = 1

    return mutation


def is_valid_offspring(o, father, mother, goal):
    pcs = find_all_movable_pieces(father.father.board)
    moves = find_all_possible_moves(pcs, father.board, goal)
    for m in moves:
        if o == m:
            o = play(o, None, None, father.board)
            return True
    pcs = find_all_movable_pieces(mother.father)
    moves = find_all_possible_moves(pcs, mother.board, goal)
    for m in moves:
        if o == m:
            o = play(o, None, None, mother.board)
            return True
    return False


def ga_goal_is_reached(population):
    for p in population:
        if p.value == 60:
            return True
    return False


def validate_offsprings(offspring, father, mother, goal, new_population, number_of_offsprings):
    pcs = find_all_movable_pieces(father.father.board)
    moves = find_all_possible_moves(pcs, father.father.board, goal)
    for m in moves:
        if offspring == m:
            offspring = play(offspring, None, None, father.father)
            new_population.append(offspring)
            number_of_offsprings += 1
            return new_population, number_of_offsprings

    if isinstance(offspring, play):
        offspring = offspring.board

    pcs = find_all_movable_pieces(mother.father.board)
    moves = find_all_possible_moves(pcs, mother.father.board, goal)
    for m in moves:
        if offspring == m:
            offspring = play(offspring, None, None, mother.father)
            new_population.append(offspring)
            number_of_offsprings += 1
            return new_population, number_of_offsprings
    return new_population, number_of_offsprings


def init_generation(population, goal):
    initialised_population = []
    for p in population:
        pcs = find_all_movable_pieces(p.board)
        moves = find_all_possible_moves(pcs, p.board, goal)
        for m in moves:
            pl = play(m, None, None, p)
            initialised_population.append(pl)
    return initialised_population


def print_population(population):
    print("")
    print("---------------------------")
    print("current population : ")
    print("---------------------------")
    i = 1
    for p in population:
        print("creature number: ", i)
        print_as_board(p.board)
        i += 1
        print("---------------------------")


def print_path(population):
    goal_offspring = play(None, None, None, None)
    for p in population:
        if p.value == 60:
            goal_offspring = p
            break
    path = []
    while not isinstance(goal_offspring, list):
        path.append(goal_offspring)
        goal_offspring = goal_offspring.father
    path.reverse()
    i = 0
    for p in path:
        print("---------------------------")
        print("Move number :", i)
        print_as_board(p.board)
        i += 1
    print("---------------------------")


def genetic_algorithm(board, goal, population_cap, detail):
    current_value = evaluate_value(board, goal)
    generations = 222
    print("current value: ", current_value)
    if current_value == 60:  # path is found
        print("goal:  ")
        print_as_board(goal)
        return "Path found"
    population = generate_initial_population(board, goal, population_cap)

    for generation in range(generations):
        print("generation number: ", generation)
        if detail:
            print_population(population)
        population = init_generation(population, goal)
        population = calculate_fitness(population, goal)
        number_of_offsprings = 0
        new_population = []

        while number_of_offsprings < 10:
            father, mother = selection(population)
            if father is 0 or mother is 0:
                return "out of moves"
            offsprings = crossover(father, mother)

            for offspring in offsprings:
                offspring = mutate(offspring)
                new_population, number_of_offsprings = validate_offsprings(offspring, father, mother, goal,
                                                                           new_population, number_of_offsprings)

        population = new_population
        population = calculate_fitness(population, goal)
        if ga_goal_is_reached(population):
            print_path(population)
            print("goal:  ")
            print_as_board(goal)
            return "Path found"

    return "no path found"


### hill climbing function
def hill_climbing(board, goal, original_board, counter, inner_counter, detail):
    current_value = evaluate_value(board, goal)
    if current_value == 60:  # path is found
        print("goal:  ")
        print_as_board(goal)
        return "Path found "
    elif counter == 10:  # max number of attempts
        print("goal:")
        print_as_board(goal)
        return "No path found :("
    if not current_value == 6:
        print("attempt mumber =   ", counter)
        print("iteration mumber =   ", inner_counter)
        better_moves = []
        equal_moves = []
        pcs = find_all_movable_pieces(board)
        moves = find_all_possible_moves(pcs, board, goal)
        print("current value: ", current_value)

        for move in moves:
            value = evaluate_value(move, goal)
            if value > current_value:
                better_moves.append(move)
            elif value == current_value:
                equal_moves.append(move)
        print("number of equal moves: ", len(equal_moves))
        print("number of better moves: ", len(better_moves))
        if len(better_moves) > 0:
            selected_move = max(better_moves)
            print("selected move:")
            print_as_board(selected_move)
            return hill_climbing(selected_move, goal, original_board, counter, inner_counter + 1, detail)
        elif len(equal_moves) > 0:
            selected_move = max(equal_moves)
            print("selected move:")
            print_as_board(selected_move)
            return hill_climbing(selected_move, goal, original_board, counter, inner_counter + 1, detail)
        else:
            return hill_climbing(generate_board(True), goal, original_board, counter + 1, 0, detail)
    return "No Path"


### simulated annealing function
def Simulated_annealing(board, goal, original_board, t, T, detail):
    current_value = evaluate_value(board, goal)
    initial_T = T
    if current_value == 60:  # path is found
        print("goal:  ")
        print_as_board(goal)
        return "Path found"
    else:
        while t <= 100:
            T = T_scheduale(initial_T, t)
            if current_value == 60:  # path is found
                print("goal:  ")
                print_as_board(goal)
                return "Path found"
            pcs = find_all_movable_pieces(board)
            moves = find_all_possible_moves(pcs, board, goal)
            if len(moves) == 0:
                board = original_board
                break
            print("iteration number: ", t)
            print("current value: ", current_value)
            random_move = random.choice(moves)
            n_value = evaluate_value(random_move, goal)
            delta = n_value - current_value
            P = calculate_P(delta, T)
            r = random.random()
            if P > r:
                board = random_move
                current_value = n_value
                print("------------------")
                print("selected move: ")
                print_as_board(board)
            print("T is: ", T)
            t += 1

    return "No path found"


def T_scheduale(T, t):
    i = t
    temp = T
    T = temp * (0.9 ** i)
    return T


###### calculate propabilty for simulated analing
def calculate_P(delta, T):
    delta = delta
    p = np.exp(delta / T)
    if p > 1:
        p = 1
    print("p is:  ", p)
    if p == 0.0:
        print("fdas")
    return p


####### returns array of possible moves reprasnted by boards`
def find_all_possible_moves(pieces, board, goal):
    moves = []
    for p in pieces:
        if is_not_on_edge(p.x, p.y):
            if board[p.y + 1][p.x + 1] == 0:
                possible_move = copy.deepcopy(board)
                possible_move[p.y][p.x] = 0
                possible_move[p.y + 1][p.x + 1] = 1
                moves.append(possible_move)
            if board[p.y + 1][p.x - 1] == 0:
                possible_move = copy.deepcopy(board)
                possible_move[p.y][p.x] = 0
                possible_move[p.y + 1][p.x - 1] = 1
                moves.append(possible_move)
        elif p.x == 1:
            if board[p.y + 1][p.x + 1] == 0:
                possible_move = copy.deepcopy(board)
                possible_move[p.y][p.x] = 0
                possible_move[p.y + 1][p.x + 1] = 1
                moves.append(possible_move)
        elif p.x == 6:
            if board[p.y + 1][p.x - 1] == 0:
                possible_move = copy.deepcopy(board)
                possible_move[p.y][p.x] = 0
                possible_move[p.y + 1][p.x - 1] = 1
                moves.append(possible_move)
    return moves


#### chosing random piece in a board that is not not in goal state and can move
def find_all_movable_pieces(a):
    pieces = []
    for i in range(1, 7):
        for j in range(1, 7):
            #### misslocated piece
            if a[i][j] == 1:
                ### not on the edge
                if is_not_on_edge(j, i):
                    if a[i + 1][j + 1] == 0 or a[i + 1][j - 1] == 0:
                        piece = location(j, i)
                        pieces.append(piece)
                elif j == 1:
                    if a[i + 1][j + 1] == 0:
                        piece = location(j, i)
                        pieces.append(piece)
                elif j == 6:
                    if a[i + 1][j - 1] == 0:
                        piece = location(j, i)
                        pieces.append(piece)
    return pieces


######### evaluates current state of hill climbing
def evaluate_value(a, b):
    value = 0
    for i in range(1, 7):
        for j in range(1, 7):
            if a[i][j] + b[i][j] == 2:
                value += 1
    return value * 10


def FindPlayPath(board, goal, search_method, detail):
    if search_method == 1:
        print(hill_climbing(board, goal, board, 0, 0, detail))

    elif search_method == 2:
        print(Simulated_annealing(board, goal, board, 1, 10, detail))

    elif search_method == 3:
        print(k_beam_search(board, goal, 0, 3, detail))

    elif search_method == 4:
        print(genetic_algorithm(board, goal, 10, detail))

    elif search_method == 5:
        print(a_star_search(board, goal, 0, detail))


def main():
    while True:
        print("")
        print("")
        print("------------------------")
        print("Welcome to Yotams AI")
        print("------------------------")
        print("Hill climbing       -> 1")
        print("Simulated annealing -> 2")
        print("K-beam search       -> 3")
        print("Genetic algorithm   -> 4")
        print("A * search          -> 5")
        print("To Exit program     -> 0")
        print("------------------------")
        search = int(raw_input("Please choose algorithm : "))
        print("------------------------")
        d = int(raw_input("For detailed output enter 1 : "))
        if d == 1:
            detail = True
        else:
            detail = False
        if search == 0:
            exit()
        print("Start: ")

        # this function generates random board change if needed
        start_board = generate_basic_board()

        your_board = []  # your board here
        # start_board = convert_66_to_native_board(your_board)

        print_as_board(start_board)
        print("Goal: ")
        goal_board = generate_board(True)
        print_as_board(goal_board)
        FindPlayPath(start_board, goal_board, search, detail)


if __name__ == "__main__":
    main()
