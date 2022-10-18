from sympy import S, separatevars
import re
import pulp as p
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog


def get_md5(s):
    hash_object = hashlib.md5(s)
    return hash_object.hexdigest()


def get_vars(obj, s):
    vlist = []
    rhs = []
    # lhs = []
    # c = []
    rel = []

    for l in s:
        t = S(l, evaluate=False)
        rhs.append(t.rhs)
        rel.append(t.rel_op)

        for x in t.free_symbols:
            vlist.append(str(x))

    obj_t = S(obj, evaluate=False)
    obj_t = separatevars(obj_t)

    for x in obj_t.free_symbols:
        vlist.append(str(x))

    vars = list(set(vlist))
    vars.sort()

    return vars


def prep_eq_string(s):
    s2 = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    s2 = re.sub(r"\s+", "", s2)
    return s2


def build_solver(name, vars_c, vars_f, obj, constr):
    # get solver
    model = p.LpProblem(name, p.LpMaximize)
    vars_dct = {}
    vars_lst = []

    for v in vars_c:
        vars_dct[v] = p.LpVariable(v, lowBound=0)
        vars_lst.append(vars_dct[v])

    for v in vars_f:
        vars_dct[v] = vars_dct[v] = p.LpVariable(v)
        vars_lst.append(vars_dct[v])

    # declare objective
    model += eval(obj, {}, vars_dct)

    for c in constr:
        model += eval(c, {}, vars_dct)

    model.solve()

    print("Model status {}".format(p.LpStatus[model.status]))

    return model


def parse_eq(s):
    lst = []
    obj = None
    obj_type = "max"
    free = []

    for line in s.split("\n"):
        line = re.sub(r"\s+", "", line.strip().lower())

        if len(line) == 0:
            continue

        if line.startswith("max") or line.startswith("min"):
            obj = line.replace("max", "").replace("min", "")

            if line.startswith("min"):
                obj_type = "min"

            continue

        if line.startswith("unconstrained"):
            free = line.replace("unconstrained", "").strip().split(",")
            continue

        lst.append(prep_eq_string(line))

    vars_lst = get_vars(prep_eq_string(obj), lst)

    vars_c = list([v for v in vars_lst if v not in free])
    name = get_md5(s.encode("utf-8"))
    print("Using {}".format(name))
    solver = build_solver(name, vars_c, free, prep_eq_string(obj), lst)


def build_solver_ex1(r1, r2, r3):
    print("Running model")
    print(r1)
    print(r2)
    print(r3)

    model = p.LpProblem("name", p.LpMaximize)
    x1 = p.LpVariable("x1", lowBound=0)
    x2 = p.LpVariable("x2", lowBound=0)

    model += 40 * x1 + 50 * x2

    model += 0.5 * x1 + 0.25 * x2 <= r1
    model += 0.5 * x1 + 0.75 * x2 <= r2
    model += x2 <= r3

    model.solve()

    print("Model status {}".format(p.LpStatus[model.status]))

    return model


def build_plot_ex1(r1, r2, r3):
    model = p.LpProblem("name", p.LpMaximize)
    x1 = p.LpVariable("x1", lowBound=0)
    x2 = p.LpVariable("x2", lowBound=0)

    model += 40 * x1 + 50 * x2

    model += 0.5 * x1 + 0.25 * x2 <= r1
    model += 0.5 * x1 + 0.75 * x2 <= r2
    model += x2 <= r3

    model.solve()

    print("Model status {}".format(p.LpStatus[model.status]))

    return model


def feasible_point(A, b):
    # finds the center of the largest sphere fitting in the convex hull
    norm_vector = np.linalg.norm(A, axis=1)
    A_ = np.hstack((A, norm_vector[:, None]))
    b_ = b[:, None]
    c = np.zeros((A.shape[1] + 1,))
    c[-1] = -1
    res = linprog(c, A_ub=A_, b_ub=b[:, None], bounds=(None, None))
    return res.x[:-1]


def hs_intersection(A, b):
    interior_point = feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    return hs


def plt_halfspace(a, b, bbox, ax, label=""):
    if a[1] == 0:
        return ax.axvline(b / a[0], label=label)
    else:
        x = np.linspace(bbox[0][0], bbox[0][1], 20)
        return ax.plot(x, (b - a[0] * x) / a[1], label=label)


def add_bbox(A, b, xrange, yrange):
    A = np.vstack((A, [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ]))
    b = np.hstack((b, [-xrange[0], xrange[1], -yrange[0], yrange[1]]))
    return A, b


def solve_convex_set(A, b, bbox):
    A_, b_ = add_bbox(A, b, *bbox)
    interior_point = feasible_point(A_, b_)
    hs = hs_intersection(A_, b_)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs


def solve_convex_set_abs(A, b):
    interior_point = feasible_point(A, b)
    hs = hs_intersection(A, b)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs

def plot_convex_set(A, b, bbox, ax=None):
    # solve and plot just the convex set (no lines for the inequations)
    points, interior_point, hs = solve_convex_set(A, b, bbox, ax=ax)
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(bbox[0])
    ax.set_ylim(bbox[1])
    ax.fill(points[:, 0], points[:, 1], 'r')
    return points, interior_point, hs


def plot_inequalities(A, b, bbox, ax=None):
    # solve and plot the convex set,
    # the inequation lines, and
    # the interior point that was used for the halfspace intersections
    points, interior_point, hs = plot_convex_set(A, b, bbox, ax=ax)
    # ax.plot(*interior_point, 'o')
    for a_k, b_k in zip(A, b):
        plt_halfspace(a_k, b_k, bbox, ax)
    # return points, interior_point, hs
