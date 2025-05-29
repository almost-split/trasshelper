import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
import xml.etree.ElementTree as xmlet


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
        assert center.shape == (2,)
        assert radius > 0
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

    # x-polynomial of unit clothoid parametrized by length normalized with
    # tangent at origin in +x direction
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

    # y-polynomial of unit clothoid parametrized by length normalized with
    # tangent at origin in +x direction
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
            np.vstack(
                [
                    self.x_polynomial(self.radius, self.length)(ls),
                    self.y_polynomial(self.radius, self.length)(ls),
                ]
            ),
        )
        ax.plot(xys[0] + self.start[0], xys[1] + self.start[1])

    @staticmethod
    def compute_unit_clothoid_from_line(radius: float, center_y: float):
        """Computes a unit clothoid and center_x of arc segment

        The computed unit clothoid transitions from x-axis in +x direction to
        an arc segment with radius whose center has the y-coordinate center_y

        Assume both radius and center_y to be positive
        """
        assert radius > 0
        assert center_y > 0

        def tangent_at_length(length: float):
            deriv = np.array([
                Clothoid.x_polynomial(radius, length).deriv()(length),
                Clothoid.y_polynomial(radius, length).deriv()(length),
            ])
            return deriv / np.linalg.norm(deriv)

        # Find the length using nested intervals
        # First, consider the following target function which computes the
        # arc's center y-coordinate from clothoid length
        def target_function(length: float):
            # y-coordinate of clothoid end
            return Clothoid.y_polynomial(radius, length)(
                length
            # y-coordinate of segment from arc center to arc point via
            # tangent of clothoid end
            ) + radius * tangent_at_length(length)[0]
        interval = [1.0, 2.0]

        # Increase upper bound of interval until we surpass center_y
        while target_function(interval[1]) < center_y:
            interval[1] *= 2.0
        while interval[1] - interval[0] > 0.0001:
            median = 0.5 * (interval[0] + interval[1])
            median_target = target_function(median)
            if median_target > center_y:
                interval[1] = median
            elif median_target < center_y:
                interval[0] = median
        length = 0.5 * (interval[0] + interval[1])


        # Compute center_x using x-coordinate of clothoid end and then
        # via tangent
        center_x = Clothoid.x_polynomial(radius, length)(
                length
        ) - radius * tangent_at_length(length)[1]
        return Clothoid(np.array([0.0, 0.0]), 0.0, radius, length), center_x

    @staticmethod
    def compute_unit_clothoid_two_circles(radius1: float, radius2: float,
                                          distance: float):
        """Computes a unit clothoid such that the clothoid segment from radius1
        to radius2 has curvature centers with the given distance

        Also returns the two centers

        The computed unit clothoid transitions from x-axis in +x direction to
        an arc segment with radius whose center has the y-coordinate center_y

        Assumes all parameters to be positive and radius1 > radius2
        """
        assert radius1 > 0
        assert radius2 > 0
        assert distance > 0
        assert radius1 > radius2

        def compute_centers(length: float):
            x_poly = Clothoid.x_polynomial(radius2, length)
            y_poly = Clothoid.y_polynomial(radius2, length)

            length1 = length * (radius2 / radius1)
            deriv_length1 = np.array([x_poly.deriv()(length1),
                                      y_poly.deriv()(length1)])
            deriv_length1 /= np.linalg.norm(deriv_length1)
            center1 = np.array([
                x_poly(length1) - radius1 * deriv_length1[1],
                y_poly(length1) + radius1 * deriv_length1[0],
            ])
            deriv_length = np.array([x_poly.deriv()(length),
                                     y_poly.deriv()(length)])
            deriv_length /= np.linalg.norm(deriv_length)
            center2 = np.array([
                x_poly(length) - radius2 * deriv_length[1],
                y_poly(length) + radius2 * deriv_length[0],
            ])
            return center1, center2


        # Find the length using nested intervals
        # First, consider the following target function which computes the
        # requires distance
        def target_function(length: float):
            center1, center2 = compute_centers(length)
            return np.linalg.norm(center1 - center2)

        interval = [0.1, 2.0]

        # Increase upper bound of interval until we surpass center_y
        while target_function(interval[1]) < distance:
            interval[1] *= 2.0
        while interval[1] - interval[0] > 0.0001:
            median = 0.5 * (interval[0] + interval[1])
            median_target = target_function(median)
            if median_target > distance:
                interval[1] = median
            elif median_target < distance:
                interval[0] = median
        length = 0.5 * (interval[0] + interval[1])

        center1, center2 = compute_centers(length)
        clothoid = Clothoid(np.array([0.0, 0.0]), 0.0, radius2, length)
        return clothoid, center1, center2

class TrassHelperWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TrassHelper")
        self.geometry('240x120')

        self.build_gui()

    def build_gui(self):
        window_frame = tk.Frame(self)
        window_frame.pack(pady=5, padx=5)

        st2_button = ttk.Button(window_frame, text="Choose st2-file", command=self.open_st2)
        st2_button.pack()

        self.st2_status_label = ttk.Label(window_frame, text="No file loaded")
        self.st2_status_label.pack()

        input_frame = tk.Frame(window_frame)
        input_frame.pack()

        seg_a_label = ttk.Label(input_frame, text="Segment A:", justify="right")
        seg_a_label.grid(row=1, column=0)
        self.seg_a_entry = ttk.Entry(input_frame, width=3, justify="right")
        self.seg_a_entry.grid(row=1, column=1)
        seg_a_connlabel = ttk.Label(input_frame, text="Connector:", justify="right")
        seg_a_connlabel.grid(row=1, column=2)
        self.seg_a_connector = ttk.Combobox(input_frame, width=2)
        self.seg_a_connector['values'] = ['0', '1']
        self.seg_a_connector['state'] = 'readonly'
        self.seg_a_connector.grid(row=1, column=3)

        seg_b_label = ttk.Label(input_frame, text="Segment B:", justify="right")
        seg_b_label.grid(row=2, column=0)
        self.seg_b_entry = ttk.Entry(input_frame, width=3, justify="right")
        self.seg_b_entry.grid(row=2, column=1)
        seg_b_connlabel = ttk.Label(input_frame, text="Connector:", justify="right")
        seg_b_connlabel.grid(row=2, column=2)
        self.seg_b_connector = ttk.Combobox(input_frame, width=2)
        self.seg_b_connector['values'] = ['0', '1']
        self.seg_b_connector['state'] = 'readonly'
        self.seg_b_connector.grid(row=2, column=3)

        compute_button = ttk.Button(window_frame, text="Compute",
                                    command=self.compute_clothoid)
        compute_button.pack()

    def open_st2(self):
        filename = tkfd.askopenfilename()
        if len(filename) == 0:
            return
        tree = xmlet.ElementTree()
        tree.parse(filename)
        lageplan = tree.find("Gleisplan/Lageplan")
        def st2_xml_child_to_segment(child):
            if child.tag == "UTM":
                 return None
            elif child.tag == "Gerade":
                line_start = np.array([float(child[0].attrib["X"]),
                                       float(child[0].attrib["Y"])])
                line_end = np.array([float(child[1].attrib["X"]),
                                     float(child[1].attrib["Y"])])
                return LineSegment(line_start, line_end)
            elif child.tag == "Kreisbogen":
                radius = 1.0 / float(child.attrib["kr"])
                center = np.array([float(child[2].attrib["X"]),
                                   float(child[2].attrib["Y"])])
                start_angle = np.rad2deg(float(child.attrib["WinkelAnf"]))
                running_angle = np.rad2deg(float(child.attrib["WinkelLauf"]))
                return ArcSegment(center, radius, start_angle, running_angle)
            elif child.tag == "Klothoide":
                start = np.array([float(child[0].attrib["X"]),
                                  float(child[0].attrib["Y"])])
                angle = np.rad2deg(float(child[0].attrib["W"]))
                radius = float(child.attrib["R"])
                length = float(child.attrib["L"])
                return Clothoid(start, angle, radius, length)
            else:
                raise "Not implemented"
        self.segments = list(filter(lambda x: x != None,
                                    map(st2_xml_child_to_segment, lageplan)))
        self.st2_status_label['text'] = f"Loaded st2-file with {len(self.segments)} segments"

    def compute_clothoid(self):
        seg_a_index = int(self.seg_a_entry.get()) - 1
        seg_a_conn = int(self.seg_a_connector.get())
        seg_b_index = int(self.seg_b_entry.get()) - 1
        seg_b_conn = int(self.seg_b_connector.get())

        seg_a = self.segments[seg_a_index]
        seg_b = self.segments[seg_b_index]
        if seg_a_conn == 0:
            seg_a.flip()
        if seg_b_conn == 1:
            seg_b.flip()
        if isinstance(seg_a, LineSegment) and isinstance(seg_b, ArcSegment):
            transform = AffTransform.rotation_from_angle(-seg_a.angle())
            transform = AffTransform(transform.rotation, -(transform *
                                                           seg_a).target)
            transformed_arc = transform * seg_b
            if transformed_arc.center[1] > 0:
                clothoid, center_x = Clothoid.compute_unit_clothoid_from_line(
                    transformed_arc.radius, transformed_arc.center[1]
                )
            else:
                clothoid, center_x = Clothoid.compute_unit_clothoid_from_line(
                    transformed_arc.radius, -transformed_arc.center[1]
                )
                clothoid = Clothoid(np.array([0, 0]), 0, -clothoid.radius,
                                             clothoid.length)
            clothoid = AffTransform(np.array([[1, 0], [0, 1]]), np.array([
                transformed_arc.center[0] - center_x, 0
            ])) * clothoid
            clothoid = transform.inverse() * clothoid
                
        elif isinstance(seg_a, ArcSegment) and isinstance(seg_b,
                                                          ArcSegment):
            clothoid, center1, center2 = Clothoid.compute_unit_clothoid_two_circles(
                seg_a.radius, seg_b.radius, np.linalg.norm(seg_a.center -
                                                           seg_b.center)
            )
            if seg_a.running_angle < 0:
                clothoid = clothoid = Clothoid(np.array([0, 0]), 0, -clothoid.radius,
                                             clothoid.length)
                center1[1] *= -1
                center2[1] *= -1
            clothoid = AffTransform(np.array([[1, 0], [0, 1]]), -center1) * clothoid
            rot_transform = AffTransform.rotation_from_angle(
                LineSegment(seg_a.center, seg_b.center).angle() - LineSegment(
                    center1, center2).angle()
            )
            clothoid = rot_transform * clothoid
            clothoid = AffTransform(np.array([[1, 0], [0, 1]]), seg_a.center) * clothoid
        else:
            raise ValueError("Cannot connect these segments")

        print(clothoid)
        fig, ax = plt.subplots()
        seg_a.plot(ax)
        seg_b.plot(ax)
        clothoid.plot(ax)
        ax.set_aspect('equal')
        plt.show()




if __name__ == "__main__":
    app = TrassHelperWindow()
    app.mainloop()

