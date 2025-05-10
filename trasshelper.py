import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk


class AffTransform(object):
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        assert rotation.shape == (2, 2)
        assert translation.shape == (2,)
        assert np.allclose(
            np.matmul(rotation, rotation.transpose()),
            np.eye(2, 2),
            rtol=1e-7,
            atol=1e-12,
        )
        self.__rotation = rotation
        self.__translation = translation

    @staticmethod
    def rotation_from_angle(angle: float):
        return AffTransform(
            np.array(
                [
                    [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
                ]
            ),
            np.array([0, 0]),
        )

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            assert other.shape == (2,)
            return self.rotation.dot(other) + self.translation
        elif isinstance(other, LineSegment):
            return LineSegment(self * other.source, self * other.target)
        elif isinstance(other, ArcSegment):
            return ArcSegment(
                self * other.center,
                other.radius,
                np.remainder(other.start_angle + self.angle(), 360.0),
                other.running_angle,
            )
        elif isinstance(other, Clothoid):
            return Clothoid(
                self * other.start,
                np.remainder(other.angle + self.angle(), 360.0),
                other.radius,
                other.length,
            )

    def __str__(self):
        return f"{self.__rotation} | {self.__translation}"

    @property
    def rotation(self):
        return self.__rotation

    @property
    def translation(self):
        return self.__translation

    def inverse(self):
        return AffTransform(
            np.linalg.inv(self.rotation),
            -np.linalg.inv(self.rotation).dot(self.translation),
        )

    def angle(self):
        return np.rad2deg(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))


class LineSegment(object):
    def __init__(self, source: np.ndarray, target: np.ndarray):
        assert source.shape == (2,)
        assert target.shape == (2,)
        self.__source = source
        self.__target = target

    def __str__(self):
        return f"source, target\n{self.source}, {self.target}"

    @property
    def source(self):
        return self.__source

    @property
    def target(self):
        return self.__target

    def length(self):
        return np.linalg.norm(self.source - self.target)

    def plot(self, ax: plt.Axes):
        ax.plot([self.source[0], self.target[0]], [self.source[1], self.target[1]])

    def flip(self):
        self.__source, self.__target = self.target, self.source

    def angle(self):
        return np.rad2deg(
            np.arctan2(self.target[1] - self.source[1], self.target[0] - self.source[0])
        )


class ArcSegment(object):
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        start_angle: float,
        running_angle: float,
    ):
        print(center, radius, start_angle, running_angle)
        assert center.shape == (2,)
        assert radius > 0
        assert start_angle >= 0 and start_angle < 360
        assert running_angle != 0

        self.__center = center
        self.__radius = radius
        self.__start_angle = start_angle
        self.__running_angle = running_angle

    def __str__(self):
        return f"center, radius, start_angle, running_angle\n{self.center}, {self.radius}, {self.start_angle}, {self.running_angle}"

    @property
    def center(self):
        return self.__center

    @property
    def radius(self):
        return self.__radius

    @property
    def start_angle(self):
        return self.__start_angle

    @property
    def running_angle(self):
        return self.__running_angle

    def plot(self, ax: plt.Axes):
        theta = np.linspace(0, self.__running_angle, 1000, endpoint=True)
        xs = self.center[0] + self.radius * np.cos(np.deg2rad(self.start_angle + theta))
        ys = self.center[1] + self.radius * np.sin(np.deg2rad(self.start_angle + theta))
        ax.plot(xs, ys)

    def flip(self):
        self.__start_angle = np.remainder(self.start_angle + self.running_angle, 360.0)
        self.__running_angle *= -1


class Clothoid(object):
    def __init__(self, start: np.ndarray, angle: float, radius: float, length: float):
        if angle > 180.0:
            angle -= 360.0
        assert start.shape == (2,)
        assert angle >= -180.0 and angle <= 180.0
        assert radius != 0
        assert length > 0

        self.__start = start
        self.__angle = angle
        self.__radius = radius
        self.__length = length

    def __str__(self):
        return f"start, angle, radius, length\n{self.start}, {self.angle}, {self.radius}, {self.length}"

    @property
    def start(self):
        return self.__start

    @property
    def angle(self):
        return self.__angle

    @property
    def radius(self):
        return self.__radius

    @property
    def length(self):
        return self.__length

    # x-polynomial parametrized by length normalized with tangent at origin in +x direction
    @staticmethod
    def x_polynomial(radius: float, length: float):
        a = np.sqrt(np.abs(radius) * length)
        return np.polynomial.Polynomial(
            [
                0,
                1,
                0,
                0,
                0,
                -1 / ((2 * a**2) ** 2 * 2 * 5),
                0,
                0,
                0,
                1 / ((2 * a**2) ** 4 * 24 * 9),
                0,
                0,
                0,
                -1 / ((2 * a**2) ** 6 * 720 * 13),
                0,
                0,
                0,
                1 / ((2 * a**2) ** 8 * 40320 * 17),
            ]
        )

    # y-polynomial parametrized by length normalized with tangent at origin in +x direction
    @staticmethod
    def y_polynomial(radius: float, length: float):
        a = np.sqrt(np.abs(radius) * length)
        return np.sign(radius) * np.polynomial.Polynomial(
            [
                0,
                0,
                0,
                1 / (2 * a**2 * 3),
                0,
                0,
                0,
                -1 / ((2 * a**2) ** 3 * 6 * 7),
                0,
                0,
                0,
                1 / ((2 * a**2) ** 5 * 120 * 11),
                0,
                0,
                0,
                -1 / ((2 * a**2) ** 7 * 5040 * 15),
                0,
                0,
                0,
                1 / ((2 * a**2) ** 9 * 362880 * 19),
            ]
        )

    def plot(self, ax: plt.Axes):
        ls = np.linspace(0, self.length, 100, endpoint=True)
        afft = AffTransform.rotation_from_angle(self.angle)
        xys = np.matmul(
            afft.rotation,
            np.row_stack(
                [
                    self.x_polynomial(self.radius, self.length)(ls),
                    self.y_polynomial(self.radius, self.length)(ls),
                ]
            ),
        )
        ax.plot(xys[0] + self.start[0], xys[1] + self.start[1])

    @staticmethod
    def find_clothoid(line: LineSegment, arc: ArcSegment):
        # First, transform such that the line is at the x-axis
        transform = AffTransform.rotation_from_angle(-line.angle())
        transform = AffTransform(transform.rotation, -(transform * line).target)
        transformed_arc = transform * arc

        if transformed_arc.center[1] > 0:
            target_cy = transformed_arc.center[1]
            clot_radius = arc.radius
        else:
            target_cy = -transformed_arc.center[1]
            clot_radius = -arc.radius

        # interval method
        interval = [1, 100]

        def cy_from_length(l: float):
            return Clothoid.y_polynomial(clot_radius, l)(
                l
            ) + arc.radius * Clothoid.x_polynomial(clot_radius, l).deriv()(l)

        while cy_from_length(interval[1]) < target_cy:
            interval[1] *= 2
        while interval[1] - interval[0] > 0.01:
            median = 0.5 * (interval[0] + interval[1])
            median_cy = cy_from_length(median)
            if median_cy > target_cy:
                interval[1] = median
            elif median_cy < target_cy:
                interval[0] = median
        length = 0.5 * (interval[0] + interval[1])

        start_angle = np.arctan2(
            -Clothoid.x_polynomial(clot_radius, length).deriv()(length),
            Clothoid.y_polynomial(clot_radius, length).deriv()(length),
        )
        end_x = transformed_arc.center[0] + transformed_arc.radius * np.cos(start_angle)
        x_length = Clothoid.x_polynomial(clot_radius, length)(length)

        return transform.inverse() * Clothoid(
            np.array([end_x - x_length, 0]), 0, clot_radius, length
        )

class TrassHelperWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TrassHelper")

        
        self.build_gui()

    def build_gui(self):
        window_frame = tk.Frame(self)
        window_frame.pack(pady=5, padx=5)

        line_arc_label = ttk.Label(window_frame, text="Line - Clothoid - Arc")
        line_arc_label.pack(pady=5)

        line_arc_frame = tk.Frame(window_frame)
        line_arc_frame.pack(pady=5)

        line_start_label_east = ttk.Label(line_arc_frame, text="Line Start East")
        line_start_label_east.grid(row=0, column=0)
        line_start_label_north = ttk.Label(line_arc_frame, text="Line Start North")
        line_start_label_north.grid(row=0, column=1)
        line_sep_label = ttk.Label(line_arc_frame, text=" ")
        line_sep_label.grid(row=0, column=2)
        line_end_label_east = ttk.Label(line_arc_frame, text="Line End East")
        line_end_label_east.grid(row=0, column=3)
        line_end_label_north = ttk.Label(line_arc_frame, text="Line End North")
        line_end_label_north.grid(row=0, column=4)

        self.line_start_entry_east = ttk.Entry(line_arc_frame)
        self.line_start_entry_east.grid(row=1, column=0)
        self.line_start_entry_north = ttk.Entry(line_arc_frame)
        self.line_start_entry_north.grid(row=1, column=1)
        self.line_end_entry_east = ttk.Entry(line_arc_frame)
        self.line_end_entry_east.grid(row=1, column=3)
        self.line_end_entry_north = ttk.Entry(line_arc_frame)
        self.line_end_entry_north.grid(row=1, column=4)
        
        arc_center_label_east = ttk.Label(line_arc_frame, text="Arc Center East")
        arc_center_label_east.grid(row=2, column=0)
        arc_center_label_north = ttk.Label(line_arc_frame, text="Arc Center North")
        arc_center_label_north.grid(row=2, column=1)
        arc_sep_label = ttk.Label(line_arc_frame, text=" ")
        arc_sep_label.grid(row=2, column=2)
        arc_radius_label = ttk.Label(line_arc_frame, text="Arc Radius")
        arc_radius_label.grid(row=2, column=3)

        self.arc_center_entry_east = ttk.Entry(line_arc_frame)
        self.arc_center_entry_east.grid(row=3, column=0)
        self.arc_center_entry_north = ttk.Entry(line_arc_frame)
        self.arc_center_entry_north.grid(row=3, column=1)
        self.arc_radius_entry = ttk.Entry(line_arc_frame)
        self.arc_radius_entry.grid(row=3, column=3)
        
        compute_button = ttk.Button(window_frame, text="Compute", width=20, command=self.compute)
        compute_button.pack(pady=5)

    def compute(self):
        line = LineSegment(np.array([float(self.line_start_entry_east.get()), float(self.line_start_entry_north.get())]),
                           np.array([float(self.line_end_entry_east.get()), float(self.line_end_entry_north.get())]))
        arc = ArcSegment(np.array([float(self.arc_center_entry_east.get()), float(self.arc_center_entry_north.get())]),
                           float(self.arc_radius_entry.get()), 0, 360)
        clot = Clothoid.find_clothoid(line, arc)
        
        print(clot)
        
        fig, ax = plt.subplots()
        line.plot(ax)
        arc.plot(ax)
        clot.plot(ax)
        ax.set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    app = TrassHelperWindow()
    app.mainloop()
