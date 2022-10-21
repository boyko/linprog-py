import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import linprog
import helpers


def render_ex1():
    r1 = 120
    r2 = 160
    r3 = 150

    A = np.array([
        [0.5, 0.25],
        [0.5, 0.75],
        [0, 1],
        [-1, 0],
        [0, -1]
    ])
    b = np.array([r1, r2, r3, 0, 0])
    c = np.array([40, 50])

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.3)

    ymin = 0.02
    dy = 0.06
    axr1 = fig.add_axes([0.3, ymin + 2 * dy, 0.5, 0.0225])
    axr2 = fig.add_axes([0.3, ymin + 1 * dy, 0.5, 0.0225])
    axr3 = fig.add_axes([0.3, ymin, 0.5, 0.0225])

    def plt_halfspace(a, b, bbox, ax, label=""):
        if a[1] == 0:
            return ax.axvline(b / a[0], label=label)
        else:
            x = np.linspace(bbox[0][0], bbox[0][1], 20)
            return ax.plot(x, (b - a[0] * x) / a[1], label=label)

    def get_linedata(a, b, bbox):
        x = np.linspace(bbox[0][0], bbox[0][1], 20)
        y = (b - a[0] * x) / a[1]
        return x, y

    r1_slider = Slider(
        ax=axr1,
        label="Cuban coffee (kg)",
        valmin=50,
        valmax=200,
        valinit=r1,
        orientation="horizontal"
    )

    r2_slider = Slider(
        ax=axr2,
        label="Brazilian coffee (kg)",
        valmin=50,
        valmax=250,
        valinit=r2,
        orientation="horizontal"
    )

    r3_slider = Slider(
        ax=axr3,
        label="Demand Delux (kg)",
        valmin=20,
        valmax=300,
        valinit=r3,
        orientation="horizontal"
    )

    bbox = [[0, 400], [0, 400]]

    ax.set_xlim(bbox[0])
    ax.set_ylim(bbox[1])

    line1 = plt_halfspace(
        A[0, :], b[0], bbox, ax,
        label="Cuban"
    )
    line2 = plt_halfspace(
        A[1, :], b[1], bbox, ax,
        label="Brazilian"
    )
    line3 = plt_halfspace(
        A[2, :], b[2], bbox, ax,
        label="Demand"
    )

    lines = [line1, line2, line3]

    opt_o = linprog(-c, A_ub=A, b_ub=b)

    opt_x1, opt_x2 = opt_o.x

    opt_text = ax.text(opt_x1, opt_x2, "({:.1f},{:.1f})".format(opt_x1, opt_x2))

    def update_r(val, idx):
        next_line_data = get_linedata(A[idx, :], val, bbox)
        lines[idx][0].set_data(next_line_data)
        next_b = [r1_slider.val, r2_slider.val, r3_slider.val, 0, 0]
        next_points, next_intpoint, next_hs = helpers.solve_convex_set(A, next_b, bbox)

        opt = linprog(-c, A_ub=A, b_ub=next_b)
        o_x1, o_x2 = opt.x
        opt_text.set_text("({:.1f},{:.1f})".format(o_x1, o_x2))
        opt_text.set_x(o_x1)
        opt_text.set_y(o_x2)

        feasible_set[0].set_xy(next_points)

        # fig.canvas.draw_idle()

    # register the update function with each slider
    r1_slider.on_changed(lambda x_: update_r(x_, 0))
    r2_slider.on_changed(lambda x_: update_r(x_, 1))
    r3_slider.on_changed(lambda x_: update_r(x_, 2))

    fig.legend(loc="upper right")

    points, intpoint, hs = helpers.solve_convex_set(A, b, bbox)

    feasible_set = ax.fill(points[:, 0], points[:, 1], alpha=0.3)
