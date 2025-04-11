import sys
import time

import cv2
import numpy as np


class Grid:
    def __init__(self, img, draw_scale=1):
        self.rows, self.cols = img.shape[0], img.shape[1]
        self.obstacles = []
        self.open_nodes = []
        self.path = []
        self.draw_scale = draw_scale

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.01 * draw_scale

        self.g_score = np.zeros((self.rows, self.cols))
        self.f_score = np.zeros((self.rows, self.cols))

        self.dist_img = cv2.distanceTransform(
            img, distanceType=cv2.DIST_L2, maskSize=5
        ).astype(np.float32)
        self.dist_img = cv2.normalize(
            self.dist_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1
        )
        self.max_dist = np.max(self.dist_img)

        # self.f_score += max_dist - dist_img

    def dist(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def clear_grid(self):
        self.g_score = np.zeros((self.rows, self.cols))
        self.f_score = np.zeros((self.rows, self.cols))
        self.obstacles = []
        self.path = []
        self.open_nodes = []

    def get_cost(self, n):
        return self.grid[n[0], n[1]]

    def get_min_f(self):
        minf = 100000.0
        minmode = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.f_score[i, j] < minf:
                    minmode = (i, j)
                    minf = self.f_score[i, j]
        return minmode

    def get_f_score(self, n):
        return self.f_score[n[0], n[1]]

    def set_f_score(self, n, f):
        self.f_score[n[0], n[1]] = f

    def set_g_score(self, n, g):
        self.g_score[n[0], n[1]] = g

    def get_g_score(self, n):
        return self.g_score[n[0], n[1]]

    def set_start_node(self, n):
        self.start_node = n
        # self.grid[n[0],n[1]] = 0

    def set_end_node(self, n):
        self.end_node = n
        # self.grid[n[0],n[1]] = 0

    def set_obstacle(self, n):
        self.grid[n[0], n[1]] = -1

    def set_path(self, path):
        self.path = path

    def create_random_obstacles(self, obstacle_density=6):
        for i in range(int(self.rows * self.cols / obstacle_density)):
            o = (np.random.randint(self.rows), np.random.randint(self.cols))
            if o != self.start_node and o != self.end_node:
                # self.set_g_score(o,-1)
                self.obstacles.append(o)

    def create_obstacles_from_img(self, img):
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                if img[r, c] < 10:
                    o = (r, c)
                    if o != self.start_node and o != self.end_node:
                        # self.set_g_score(o,-1)
                        self.obstacles.append(o)

    def add_horiz_obstacle(self, row, col1, col2):
        for c in range(col1, col2 + 1):
            self.obstacles.append((row, c))

    def add_vert_obstacle(self, row1, row2, col):
        for r in range(row1, row2 + 1):
            self.obstacles.append((r, col))

    def find_neighbours(self, n):
        neighbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (
                    not (i == 0 and j == 0)
                    and (n[0] + i, n[1] + j) not in self.obstacles
                ):
                    if (
                        n[0] + i >= 0
                        and n[0] + i < self.rows
                        and n[1] + j >= 0
                        and n[1] + j < self.cols
                    ):
                        neighbours.append((n[0] + i, n[1] + j))
        return neighbours

    def draw(self):
        if self.draw_scale > 20:
            pad = 4
        else:
            pad = 0

        img = (
            np.ones(
                (self.rows * self.draw_scale, self.cols * self.draw_scale, 3),
                dtype=np.uint8,
            )
            * 255
        )

        # cv2.imshow("distimg", 255- dist3)

        # grid
        # for i in range(1,self.rows):
        #     cv2.line(img, (0,i*self.draw_scale), (self.cols*self.draw_scale,i*self.draw_scale), (0,0,0))
        # for j in range(1,self.cols):
        #     cv2.line(img, (j*self.draw_scale,0), (j*self.draw_scale, self.rows*self.draw_scale), (0,0,0))
        # cv2.rectangle(img,(0,0),(self.cols*self.draw_scale-1, self.rows*self.draw_scale-1), (0,0,0))

        for i in range(self.rows):
            for j in range(self.cols):
                col = int(255 - self.g_score[i, j] * 2)
                s = (j * self.draw_scale + pad, i * self.draw_scale + pad)
                e = (
                    j * self.draw_scale + self.draw_scale - pad,
                    i * self.draw_scale + self.draw_scale - pad,
                )
                if self.g_score[i, j] == 0:
                    col2 = (255, 255, 255)
                else:
                    col2 = (col, 255, 255)
                cv2.rectangle(img, s, e, col2, -1)

        # dist3 = np.zeros_like(img)
        # dist3[:,:,0] = cv2.resize( self.dist_img, (img.shape[1], img.shape[0]))/2
        # dist3[:,:,1] = cv2.resize( self.dist_img, (img.shape[1], img.shape[0]))/2
        # dist3[:,:,2] = cv2.resize( self.dist_img, (img.shape[1], img.shape[0]))/2

        # img -= 255-dist3

        # open nodes
        for node in self.open_nodes:
            cv2.rectangle(
                img,
                (node[1] * self.draw_scale + pad, node[0] * self.draw_scale + pad),
                (
                    node[1] * self.draw_scale + self.draw_scale - pad,
                    node[0] * self.draw_scale + self.draw_scale - pad,
                ),
                (0, 200, 200),
                -1,
            )

        # obstacles
        for node in self.obstacles:
            cv2.rectangle(
                img,
                (node[1] * self.draw_scale + pad, node[0] * self.draw_scale + pad),
                (
                    node[1] * self.draw_scale + self.draw_scale - pad,
                    node[0] * self.draw_scale + self.draw_scale - pad,
                ),
                (100, 100, 100),
                -1,
            )

        # path
        for node in self.path:
            cv2.rectangle(
                img,
                (node[1] * self.draw_scale + pad, node[0] * self.draw_scale + pad),
                (
                    node[1] * self.draw_scale + self.draw_scale - pad,
                    node[0] * self.draw_scale + self.draw_scale - pad,
                ),
                (255, 255, 0),
                -1,
            )

        # start, end nodes
        if hasattr(self, 'start_node'):
            cv2.rectangle(
                img,
                (
                    self.start_node[1] * self.draw_scale + pad,
                    self.start_node[0] * self.draw_scale + pad,
                ),
                (
                    self.start_node[1] * self.draw_scale + self.draw_scale - pad,
                    self.start_node[0] * self.draw_scale + self.draw_scale - pad,
                ),
                (0, 0, 255),
                -1,
            )
        if hasattr(self, 'end_node'):
            cv2.rectangle(
                img,
                (
                    self.end_node[1] * self.draw_scale + pad,
                    self.end_node[0] * self.draw_scale + pad,
                ),
                (
                    self.end_node[1] * self.draw_scale + self.draw_scale - pad,
                    self.end_node[0] * self.draw_scale + self.draw_scale - pad,
                ),
                (0, 255, 0),
                -1,
            )

        # costs
        # if pad>1:
        #     for i in range(1,self.rows+1):
        #         for j in range(1,self.cols+1):
        #             if (i-1,j-1) not in self.obstacles:
        #                 cv2.putText(img, str(int(self.g_score[i-1,j-1]))  , (int(j*self.draw_scale-self.draw_scale+pad), int(i*self.draw_scale-self.draw_scale/2)) , self.font, self.fontScale, (0,0,0), 1)
        #                 # cv2.putText(img, str(int(self.f_score[i-1,j-1]))  , (int(j*self.draw_scale-12), int(i*self.draw_scale-8)) , self.font, self.fontScale, (0,0,0), 1)

        # cv2.imshow("Grid", img)
        # c = cv2.waitKey(1)
        # if c == 27:
        #     cv2.destroyAllWindows()
        #     sys.exit(0)
        return img


def astar(grid):
    prev_time = time.time()
    open_nodes = [grid.start_node]
    closed_nodes = []
    came_from = {}

    grid.set_g_score(grid.start_node, 0)
    grid.set_f_score(
        grid.start_node,
        grid.get_g_score(grid.start_node) + grid.dist(grid.start_node, grid.end_node),
    )

    # main loop
    step = 0
    while len(open_nodes) > 0:
        # print('step',step, 'open',len(grid.open_nodes))
        step += 1
        minf = 100000.0
        current = []
        for n in open_nodes:
            if grid.get_f_score(n) < minf:
                current = n
                minf = grid.get_f_score(n)

        if current == grid.end_node:
            # print("Goal reached!")
            grid.path = reconstruct_path(came_from, current)
            grid.draw()
            return

        open_nodes.remove(current)
        closed_nodes.append(current)

        nn = grid.find_neighbours(current)
        for n in nn:
            if n in closed_nodes:
                continue
            tentative_g_score = grid.get_g_score(current) + grid.dist(current, n)

            if n not in open_nodes or tentative_g_score < grid.get_g_score(n):
                came_from[n] = current
                grid.set_g_score(n, tentative_g_score)
                grid.set_f_score(
                    n,
                    grid.get_g_score(n)
                    + grid.dist(n, grid.end_node)
                    + 1 * (255 - grid.dist_img[n[0], n[1]]),
                )
                if n not in open_nodes:
                    open_nodes.append(n)

        grid.open_nodes = open_nodes

        # if time.time() - prev_time >= 2.0:
        #     grid.draw()
        #     prev_time = time.time()


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path
