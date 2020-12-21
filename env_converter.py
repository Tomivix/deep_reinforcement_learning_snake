
import numpy as np

# Definitions of values at particular positions:
#  0 - (int) distance between snake's head and the top wall
#  1 - (int) distance between snake's head and the right wall
#  2 - (int) distance between snake's head and the bottom wall
#  3 - (int) distance between snake's head and the left wall
#  4 - (bool) can snake see their body to the north?
#  5 - (bool) can snake see their body to the northeast?
#  6 - (bool) can snake see their body to the east?
#  7 - (bool) can snake see their body to the southeast?
#  8 - (bool) can snake see their body to the south?
#  9 - (bool) can snake see their body to the southwest?
# 10 - (bool) can snake see their body to the west?
# 11 - (bool) can snake see their body to the northwest?
# 12 - (bool) can snake see food to the north?
# 13 - (bool) can snake see food to the northeast?
# 14 - (bool) can snake see food to the east?
# 15 - (bool) can snake see food to the southeast?
# 16 - (bool) can snake see food to the south?
# 17 - (bool) can snake see food to the southwest?
# 18 - (bool) can snake see food to the west?
# 19 - (bool) can snake see food to the northwest?
# 20 - (int) is snake's head moving to the north?
# 21 - (int) is snake's head moving to the east?
# 22 - (int) is snake's head moving to the south?
# 23 - (int) is snake's head moving to the west?
# 24 - (int) is snake's tail moving to the north?
# 25 - (int) is snake's tail moving to the east?
# 26 - (int) is snake's tail moving to the south?
# 27 - (int) is snake's tail moving to the west?


def get_input_for_nn(envir, sid=0):
    values = [None] * 28

    snake = envir.controller.snakes[sid]

    sx = snake.head[0]
    sy = snake.head[1]
    mx = envir.grid_size[0] - 1
    my = envir.grid_size[1] - 1

    bc = envir.controller.grid.BODY_COLOR
    fc = envir.controller.grid.FOOD_COLOR

    values[0] = sy
    values[1] = mx - sx
    values[2] = my - sy
    values[3] = sx

    for ind in range(4, 20):
        values[ind] = False

    for cy in range(sy - 1, -1, -1):
        color = envir.controller.grid.color_of((sx, cy))
        if np.array_equal(color, bc):
            values[4] = True
        elif np.array_equal(color, fc):
            values[12] = True

    for cx in range(sx + 1, mx + 1):
        color = envir.controller.grid.color_of((cx, sy))
        if np.array_equal(color, bc):
            values[6] = True
        elif np.array_equal(color, fc):
            values[14] = True

    for cy in range(sy + 1, my + 1):
        color = envir.controller.grid.color_of((sx, cy))
        if np.array_equal(color, bc):
            values[8] = True
        elif np.array_equal(color, fc):
            values[16] = True

    for cx in range(sx - 1, -1, -1):
        color = envir.controller.grid.color_of((cx, sy))
        if np.array_equal(color, bc):
            values[10] = True
        elif np.array_equal(color, fc):
            values[18] = True

    md = min(mx - sx, sy)
    for cd in range(0, md + 1):
        color = envir.controller.grid.color_of((sx + cd, sy - cd))
        if np.array_equal(color, bc):
            values[5] = True
        elif np.array_equal(color, fc):
            values[13] = True

    md = min(mx - sx, my - sy)
    for cd in range(0, md + 1) :
        color = envir.controller.grid.color_of((sx + cd, sy + cd))
        if np.array_equal(color, bc):
            values[7] = True
        elif np.array_equal(color, fc):
            values[15] = True

    md = min(sx, my - sy)
    for cd in range(0, md + 1):
        color = envir.controller.grid.color_of((sx - cd, sy + cd))
        if np.array_equal(color, bc):
            values[9] = True
        elif np.array_equal(color, fc):
            values[17] = True

    md = min(sx, sy)
    for cd in range(0, md + 1):
        color = envir.controller.grid.color_of((sx - cd, sy - cd))
        if np.array_equal(color, bc):
            values[11] = True
        elif np.array_equal(color, fc):
            values[19] = True

    for ind in range(0, 4):
        values[20 + ind] = 1 if snake.direction == ind else 0

    tx = snake.body[0][0]
    ty = snake.body[0][1]
    sx = snake.body[1][0]
    sy = snake.body[1][1]

    values[24] = 1 if ty - sy == 1 else 0
    values[25] = 1 if sx - tx == 1 else 0
    values[26] = 1 if sy - ty == 1 else 0
    values[27] = 1 if tx - sx == 1 else 0

    print('VALUES: ')
    for index, value in enumerate(values):
        print(index, value)

    return values
