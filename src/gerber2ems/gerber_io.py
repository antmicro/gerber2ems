"""Module provides classes related to gerber file parsing and representation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, ClassVar
from enum import Enum
import re
import logging
from functools import partial
from math import cos, sin, radians
from copy import deepcopy
from gerber2ems.constants import BASE_UNIT, UNIT_MULTIPLIER

logger = logging.getLogger(__name__)


class PlotMode(Enum):
    """Enum represents Plotting mode, matching gerber file specification (G01,G02,G03)."""

    LINEAR = 1
    CIRCULAR_CLOCK = 2
    CIRCULAR_NCLOCK = 3


@dataclass
class NumberFormat:
    """Class contains format which is used to encode floats in gerber file."""

    int_digits: int = 4
    """Maximum number of digits used to encode integer part"""
    frac_digits: int = 6
    """Number of digits used to encode fractional part"""

    def parse(self, i: str) -> float:
        """Decode number from gerber format and convert it to float."""
        return float(i) / (10**self.frac_digits)


@dataclass
class Position:
    """Coordinates of 2D point."""

    x: float
    y: float

    def mirror(self, axis: str) -> None:
        """Mirror point using `axis` axis."""
        setattr(self, axis, -getattr(self, axis))

    def rotate(self, angle: float) -> None:
        """Rotate point around (0,0)."""
        (self.x, self.y) = (self.x * cos(angle) + self.y * sin(angle), self.x * sin(angle) + self.y * cos(angle))

    def scale(self, scale: float) -> None:
        """Scale position by `scale`."""
        (self.x, self.y) = (self.x * scale, self.y * scale)

    def move(self, offset: "Position") -> None:
        """Move point by `offset`."""
        (self.x, self.y) = (self.x + offset.x, self.y + offset.y)


@dataclass
class PadMeta:
    """Class stores metadata about pad."""

    comp_ref: str
    """Reference designator of component this pad belongs to"""
    pin_num: str
    """Pad numeration inside associated component (eg. 1, A1, ..)"""
    pin_name: str
    """Pad name/function (eg. GPIO_1)"""


@dataclass
class Pad:
    """Class describes component pad (flashed aperture in gerber file)."""

    aperture: str
    """Name/key of aperture used to create ths pad"""
    net: str
    """Net this pad is connected to"""
    pos: Position
    """Pad coordinates"""
    pin_ref: Optional[PadMeta] = None
    """Pad metadata"""
    additive: bool = True
    """If false pad draw will work like eraser"""
    mirror: str = "N"
    """Aperture is mirrored in specified axis"""
    rotation: float = 0
    """Rotation of aperture"""
    scale: float = 1
    """Scale of aperture"""


@dataclass
class TraceSegment:
    """Class represents single trace segment (one line)."""

    start: Position
    """Coordinates of point on line start"""
    stop: Position
    """Coordinates of point on line end"""

    aperture: str
    """Name/key of aperture used to create ths trace (determines width)"""
    width: float
    """Trace width (should follow aperture size)"""
    mode: PlotMode = PlotMode.LINEAR
    """Mode in which trace was drawn"""
    normal: bool = True
    """Matters only for traces near parallel to axis and when width==0;
    If true normal(pointing inward shape) aligns with positive direction of dominant axis"""

    def __post_init__(self) -> None:
        """Ensure that position is not shared with anything."""
        self.start = deepcopy(self.start)
        self.stop = deepcopy(self.stop)

    def dominant_x(self) -> bool:
        """Return True when difference between start and stop is larger on X axis than on Y axis."""
        return abs(self.start.x - self.stop.x) > abs(self.start.y - self.stop.y)

    def rotate(self, ang: float) -> None:
        """Rotate segment by `ang` angle (in degrees)."""
        if self.dominant_x():
            flipped_norm = (self.start.y < 0) != self.normal
        else:
            flipped_norm = (self.start.x < 0) != self.normal
        self.start.rotate(ang)
        self.stop.rotate(ang)
        if self.dominant_x():
            self.normal = (self.start.y < 0) ^ flipped_norm
        else:
            self.normal = (self.start.x < 0) ^ flipped_norm
        self.normal = self.normal if ang % 360 < 180 else not self.normal


@dataclass
class Trace:
    """Collection of traces of single net."""

    segments: List[TraceSegment]


class ApertureType:
    """Abstract class for describing Aperture shape."""

    hole_diameter: float
    """Diameter of a round hole. A decimal >0. If omitted the aperture is solid."""

    def _contours(self) -> List[TraceSegment]:
        """Return contours of an aperture."""
        return []

    def contours(
        self, pos: Optional[Position] = None, rot: float = 0, scale: float = 1, mirror: str = "N", post_rot: float = 0
    ) -> List[TraceSegment]:
        """Return contours of an aperture with transform applied.

        Rotation can be of two kinds `rot` that is applied before scale/move, and `post_rot` that is applied after them
        """
        cont = self._contours()
        for c in cont:
            if "X" in mirror:
                c.start.mirror("x")
                c.stop.mirror("x")
                c.normal = c.normal if c.dominant_x() else not c.normal
            if "Y" in mirror:
                c.start.mirror("y")
                c.stop.mirror("y")
                c.normal = c.normal if not c.dominant_x() else not c.normal
            c.rotate(rot)
            c.start.scale(scale)
            c.stop.scale(scale)
            c.normal = c.normal if scale > 0 else not c.normal
            pos = Position(0, 0) if pos is None else pos
            c.start.move(pos)
            c.stop.move(pos)
            c.rotate(post_rot)
        return cont


@dataclass
class ApertureCircle(ApertureType):
    """Aperture shape that is circle."""

    diameter: float

    def _contours(self) -> List[TraceSegment]:
        """Return contours of an aperture."""
        d2 = self.diameter / 2
        points = [
            Position(0, d2),
            Position(d2, 0),
            Position(-d2, 0),
            Position(0, -d2),
        ]
        seg = partial(TraceSegment, aperture="", width=0, mode=PlotMode.CIRCULAR_CLOCK)
        return [
            seg(points[0], points[1]),
            seg(points[1], points[2]),
            seg(points[2], points[3]),
            seg(points[3], points[0]),
        ]


@dataclass
class ApertureRect(ApertureType):
    """Aperture shape that is rectangle."""

    x: float
    """Width"""
    y: float
    """Height"""

    def _contours(self) -> List[TraceSegment]:
        """Return contours of an aperture."""
        x2 = self.x / 2
        y2 = self.y / 2
        points = [
            Position(x2, y2),
            Position(-x2, y2),
            Position(-x2, -y2),
            Position(x2, -y2),
        ]
        seg = partial(TraceSegment, aperture="", width=0, mode=PlotMode.LINEAR)
        return [
            seg(points[0], points[1], normal=False),
            seg(points[1], points[2], normal=True),
            seg(points[2], points[3], normal=True),
            seg(points[3], points[0], normal=False),
        ]


@dataclass
class ApertureObround(ApertureRect):
    """Aperture shape that is obround (two half circles with rect in between)."""

    def _contours(self) -> List[TraceSegment]:
        """Return contours of an aperture."""
        x2 = self.x / 2
        y2 = self.y / 2
        d = y2 - x2
        rect = [
            Position(d, y2),
            Position(-d, y2),
            Position(-d, -y2),
            Position(d, -y2),
        ]
        circ = [Position(x2, 0), Position(-x2, 0)]
        seg_c = partial(TraceSegment, aperture="", width=0, mode=PlotMode.CIRCULAR_CLOCK)
        seg_l = partial(TraceSegment, aperture="", width=0, mode=PlotMode.LINEAR)
        return [
            seg_l(rect[0], rect[1], normal=False),
            seg_l(rect[2], rect[3], normal=True),
            seg_c(rect[0], circ[0]),
            seg_c(rect[3], circ[0]),
            seg_c(rect[1], circ[1]),
            seg_c(rect[2], circ[1]),
        ]


def _points2outline(points: List[Position]) -> List[TraceSegment]:
    """Connect `points` with trace segments."""
    seg = partial(TraceSegment, aperture="", width=0, mode=PlotMode.LINEAR)
    segments = []
    for idx in range(len(points)):
        s = seg(points[idx], points[idx % len(points)])
        if s.dominant_x():
            s.normal = s.start.y < 0
        else:
            s.normal = s.start.x < 0
        segments.append(s)
    return segments


def _parse_expr(s: str) -> Callable:
    """Transform expression from aperture macro definition into python callable."""
    s = s.replace(r"x", r"*")
    s = re.sub(r"\$([0-9]*)", r"args[\1]", s)
    return lambda args: eval(s, {"__builtins__": None}, {"args": args})


@dataclass
class AperturePolygon(ApertureType):
    """Aperture shape that is regular polygon."""

    diameter: float
    """Diameter of the circle circumscribing the regular polygon, i.e. the circle through the polygon vertices"""
    vertices: int
    """Number of vertices"""
    rotation: float
    """The rotation angle, in degrees counterclockwise. A decimal.
    With rotation angle zero there is a vertex on the positive X-axis
    through the aperture center"""

    def _contours(self) -> List[TraceSegment]:
        """Return contours of an aperture."""
        points = []
        d2 = self.diameter
        for i in range(self.vertices):
            ang = radians(self.rotation + 360 * i / self.vertices)
            points.append(Position(d2 * cos(ang), d2 * sin(ang)))
        return _points2outline(points)


class ApertureMacro(ApertureType):
    """Aperture macro definition.

    After setting `args` it represents also specific aperture
    """

    args: List[float]
    """Values during aperture instantiation (macro call)"""
    name: str
    """Macro ID"""
    variables: List[Callable]
    """Expressions that calculate additional macro variables"""
    commands: List[Callable]
    """Commands that create shape of aperture"""

    def __init__(self, lines: List[str]) -> None:
        """Parse aperture macro definition."""
        lines_i = iter(lines)
        self.name = next(lines_i).removeprefix("AM")
        self.variables = []
        self.commands = []
        for line in lines_i:
            if line.startswith("0"):
                # comment line
                continue
            if line.startswith("$"):
                # variable
                self.variables.append(_parse_expr(line.partition("=")[2]))
                continue

            line_i = iter(line.split(","))
            op = next(line_i)
            sline = [_parse_expr(li) for li in line_i]
            if op == "1":
                # circle
                def f(args: List[float], sline: List[Callable] = sline) -> List[TraceSegment]:
                    param = [p(args) for p in sline]
                    ap = ApertureCircle(diameter=param[1] * FileFormat.gbr2sim)
                    rot = param[4] if len(param) > 4 else 0
                    return ap.contours(
                        pos=Position(param[2] * FileFormat.gbr2sim, param[3] * FileFormat.gbr2sim), post_rot=rot
                    )

            elif op == "20":
                # line start/stop/width
                def f(args: List[float], sline: List[Callable] = sline) -> List[TraceSegment]:
                    param = [p(args) for p in sline]
                    trace = TraceSegment(
                        start=Position(param[2] * FileFormat.gbr2sim, param[3] * FileFormat.gbr2sim),
                        stop=Position(param[4] * FileFormat.gbr2sim, param[5] * FileFormat.gbr2sim),
                        aperture="",
                        width=param[1] * FileFormat.gbr2sim,
                    )
                    trace.rotate(param[6])
                    return [trace]

            elif op == "21":
                # line center/width/length
                def f(args: List[float], sline: List[Callable] = sline) -> List[TraceSegment]:
                    param = [p(args) for p in sline]
                    len2 = FileFormat.gbr2sim * param[1] / 2
                    trace = TraceSegment(
                        start=Position(param[3] * FileFormat.gbr2sim - len2, param[4] * FileFormat.gbr2sim),
                        stop=Position(param[3] * FileFormat.gbr2sim + len2, param[4] * FileFormat.gbr2sim),
                        aperture="",
                        width=param[2] * FileFormat.gbr2sim,
                    )
                    trace.rotate(param[5])
                    return [trace]

            elif op == "4":
                # outline
                def f(args: List[float], sline: List[Callable] = sline) -> List[TraceSegment]:
                    param = [p(args) for p in sline]
                    points = []
                    for i in range(param[1] + 1):
                        points.append(
                            Position(param[2 + i * 2] * FileFormat.gbr2sim, param[3 + i * 2] * FileFormat.gbr2sim)
                        )
                    contours = _points2outline(points)
                    for seg in contours:
                        seg.rotate(param[-1])
                    return contours

            elif op == "5":
                # polygon
                def f(args: List[float], sline: List[Callable] = sline) -> List[TraceSegment]:
                    param = [p(args) for p in sline]
                    ap = AperturePolygon(vertices=param[1], diameter=param[4] * FileFormat.gbr2sim, rotation=0)
                    return ap.contours(
                        pos=Position(param[2] * FileFormat.gbr2sim, param[3] * FileFormat.gbr2sim), post_rot=param[5]
                    )

            elif op == "7":
                # Thermal relief
                # TODO
                pass
            else:
                raise Exception(f"Unknown aperture macro op: {line}")
            self.commands.append(f)

    def _contours(self) -> List[TraceSegment]:
        """Return contours of an aperture."""
        for v in self.variables:
            self.args.append(v([0.0] + self.args))
        contours = []
        for cmd in self.commands:
            contours.extend(cmd([0.0] + self.args))
        return contours


@dataclass
class Aperture:
    """Represents Aperture as specified in gerber file.

    Aperture is basic shape used to draw anything in gerber file
    """

    function: str
    """Intended use of this aperture (eg. Via, SMD-pad)"""
    data: ApertureType
    """Type/Shape of this aperture"""


@dataclass
class FileFormat:
    """Format data about file."""

    unit: str = "MM"
    """Unit in which all dimensions/coordinates are given"""
    omit_zeros: bool = True
    """Leading zeros in numbers can be omitted if True"""
    absolute: bool = True
    """Coordinates used in this file are absolute if True"""
    x_format: NumberFormat = field(default_factory=lambda: NumberFormat())
    """Format used to encode position in X-axis"""
    y_format: NumberFormat = field(default_factory=lambda: NumberFormat())
    """Format used to encode position in Y-axis"""
    gbr2sim: ClassVar[float]

    def parse_position(self, x: str, y: str) -> Position:
        """Convert position from format used in gerber file to `Position` class."""
        pos = Position(self.x_format.parse(x), self.y_format.parse(y))
        pos.x *= FileFormat.gbr2sim
        pos.y *= FileFormat.gbr2sim

        return pos

    def __setattr__(self, name: str, value: Any) -> None:
        """Setter method."""
        if name == "unit":
            mm2sim = UNIT_MULTIPLIER / BASE_UNIT / 1000
            in2sim = 25.4 * UNIT_MULTIPLIER / BASE_UNIT / 1000
            FileFormat.gbr2sim = mm2sim if value == "MM" else in2sim
        super(FileFormat, self).__setattr__(name, value)


@dataclass
class ParserState:
    """Temporary state of gerber file parser."""

    aperture: str = ""
    """Recently set aperture (will be used for upcoming trace creation/pad flashing)"""
    aperture_func: str = ""
    """Recently set aperture function (will be set for upcoming aperture declarations)"""
    net: str = "no-net"
    """Recently set net (will be used for upcoming trace creation/pad flashing)"""
    refpin: Optional[PadMeta] = None
    """Recently set pad metadata (will be used for upcoming pad flashing)"""
    unparsed_region: bool = False
    """Region of code that is currently ignored by parser"""
    pos: Position = field(default_factory=lambda: Position(x=0, y=0))
    """Current position (will be used for upcoming trace creation)"""
    plot_mode: PlotMode = PlotMode.LINEAR
    """Current plot mode (will be used for upcoming trace creation)"""
    additive: bool = True
    """If false future draws will work like eraser"""
    mirror: str = "N"
    """Aperture should be mirrored for upcoming operations"""
    rotation: float = 0
    """Rotation of aperture for upcoming operations"""
    scale: float = 1
    """Scale of aperture for upcoming operations"""
    fformat: FileFormat = field(default_factory=lambda: FileFormat())
    """Position/dimension format used currently"""
    zone: bool = False
    """Plotting zone (started by G37, ends with G36)"""
    zone_contours: List[TraceSegment] = field(default_factory=list)
    """Contours of zone that is currently being parsed"""
    ap_macro: List[str] = field(default_factory=list)
    """Body of currently parsed aperture macro"""


@dataclass(init=False)
class GerberFile:
    """Gerber file parsed representation."""

    unparsed: str
    """Parts of code that are currently not supported by parser"""
    apertures: Dict[str, Aperture]
    """All apertures used in this file"""
    traces: Dict[str, Trace]
    """Traces defined in this file, grouped by nets"""
    pads: List[Pad]
    """List of all pads defined in this file"""
    parser: ParserState
    """Temporary parser state"""
    ap_macros: Dict[str, ApertureMacro]
    """List of aperture macro definitions"""

    def __init__(self, path: Path) -> None:
        """Load and parse gerber file from specified path."""
        logger.info(f"Parsing gerber file: {path}")
        self.unparsed = ""
        self.apertures = {}
        self.traces = {}
        self.pads = []
        self.ap_macros = {}
        self.parser = ParserState()
        with open(path) as file_h:
            file = file_h.read()
            lines = re.findall(r".*[*][%]?\n", file)
        for line in lines:
            if line.startswith("%"):
                self.process_percent_line(line)
            else:
                self.process_normal_line(line)

    def process_percent_line(self, line: str) -> None:
        """Parse & process line starting with `%` sign.

        % lines carry following information:
        - pin information (component, pin number, pin name) for next pads
        - Function of following apertures
        - Net of following metal patterns
        - Scaling, polarization, rotation, mirror of aperture
        - Aperture definition

        Function updates parser state and aperture dictionary
        """
        self.unparsed = self.unparsed + line
        sline = line.strip("%*\n")
        split = sline.split(",")
        if split[0][0:2] == "AM":
            self.parser.ap_macro.append(sline)
        elif split[0] == "TO.P":
            self.parser.refpin = PadMeta(
                comp_ref=split[1], pin_num=split[2], pin_name=split[3] if len(split) > 3 else ""
            )
        elif split[0] == "TO.N":
            self.parser.net = split[1]
        elif split[0] == "TA" and split[1] == "AperFunction":
            self.parser.aperture_func = split[2]
        elif split[0] == "TD":
            self.parser.net = "no-net"
            self.parser.refpin = None
        elif split[0][0:2] == "LP":
            self.parser.additive = split[0][2] == "D"
        elif split[0][0:2] == "LR":
            self.parser.rotation = float(split[0][2:])
        elif split[0][0:2] == "LM":
            self.parser.mirror = split[0][2:].upper()
        elif split[0][0:2] == "LS":
            self.parser.scale = float(split[0][2:])

        elif split[0].startswith("AD"):
            full_id = split[0].split(",")[0].removeprefix("AD")
            aper_data_type = full_id.removeprefix("D").lstrip("0123456789")
            name = full_id.removesuffix(aper_data_type)
            aper_data = ApertureType()
            p = split[1].split("X")
            p_used = len(p)
            if aper_data_type == "C":
                aper_data = ApertureCircle(float(p[0]) * FileFormat.gbr2sim)
                p_used = 1
            elif aper_data_type == "R":
                aper_data = ApertureRect(float(p[0]) * FileFormat.gbr2sim, float(p[1]) * FileFormat.gbr2sim)
                p_used = 2
            elif aper_data_type == "O":
                aper_data = ApertureObround(float(p[0]) * FileFormat.gbr2sim, float(p[1]) * FileFormat.gbr2sim)
                p_used = 2
            elif aper_data_type == "P":
                aper_data = AperturePolygon(float(p[0]) * FileFormat.gbr2sim, int(p[1]), float(p[2]))
                p_used = 3
            elif aper_data_type in self.ap_macros:
                p_used = len(p)
                aper_data = deepcopy(self.ap_macros[aper_data_type])
                aper_data.args = [float(i) for i in p]

            if p_used < len(p):
                aper_data.hole_diameter = float(p[p_used]) * FileFormat.gbr2sim
            self.apertures[name] = Aperture(self.parser.aperture_func, aper_data)

    def process_normal_line(self, line: str) -> None:
        """Parse & process normal line (not starting with `%` sign).

        Normal line can have following functions:
        - Start/Stop Region/zone(G36-G37), block macro(AB), step&repeat(SR) definition
        - Comments (GO4)
        - Plot mode (G01-G03), Linear/Circular
        - Plotting operation (X*)
        - Set aperture (D*)
        - Body of aperture macro

        Function updates parser state and Trace/Pad collections
        """
        sline = line.strip("%*\n")
        split = sline.split(",")
        if split[0] in ["AB", "SR"]:
            self.parser.unparsed_region = not self.parser.unparsed_region

        if self.parser.unparsed_region:
            self.unparsed = self.unparsed + line
            return

        if len(self.parser.ap_macro) != 0:
            self.parser.ap_macro.append(sline)
            if line.rstrip().endswith("%"):
                apm = ApertureMacro(self.parser.ap_macro)
                self.ap_macros[apm.name] = apm
                self.parser.ap_macro = []
        elif split[0] == "G36":
            self.parser.zone_contours = []
            self.parser.zone = True
        elif split[0] == "G37":
            s_start = self.parser.zone_contours[0]
            s_end = self.parser.zone_contours[-1]
            self.parser.zone_contours.append(TraceSegment(s_start.start, s_end.stop, "", 0, PlotMode.LINEAR))
            trace = self.traces.get(self.parser.net, Trace([]))
            trace.segments.extend(self.parser.zone_contours)
            self.traces[self.parser.net] = trace
            self.parser.zone_contours = []
            self.parser.zone = False
        elif split[0][0:3] in ["G04", "G75"]:
            self.unparsed = self.unparsed + line
        elif split[0].startswith("G"):
            self.parser.plot_mode = PlotMode(int(split[0][2]))
        elif split[0].startswith("D"):
            self.parser.aperture = split[0]
        elif split[0].startswith("X"):
            self.process_drawing_line(sline)

    def process_drawing_line(self, line: str) -> None:
        """Parse & process line that describes plotting operation (starting with X).

        Updates trace/pad collections, moves cursor position
        """
        sline = line.removeprefix("X")
        [x, _, y] = sline.partition("Y")
        [y, _, op] = y.partition("D")
        [y, _, y_offset] = y.partition("J")
        [y, _, x_offset] = y.partition("I")
        pos = self.parser.fformat.parse_position(x, y)
        opi = int(op)
        if opi == 1:
            if self.parser.zone:
                self.parser.zone_contours.append(TraceSegment(self.parser.pos, pos, "", 0, self.parser.plot_mode))
            else:
                trace = self.traces.get(self.parser.net, Trace([]))
                ap_name = self.parser.aperture
                ap = self.apertures.get(ap_name, None)
                if not ap:
                    logger.error(f"Aperture `{ap_name}` used for line: `{line}` not defined!")
                    return
                if not isinstance(ap.data, ApertureCircle):
                    logger.error(f"Aperture `{ap_name}` used for line: `{line}` is not circular aperture!")
                    return

                trace.segments.append(
                    TraceSegment(self.parser.pos, pos, ap_name, ap.data.diameter, self.parser.plot_mode)
                )
                self.traces[self.parser.net] = trace
        elif opi == 2:
            self.parser.pos = pos
        elif opi == 3:
            aperture = self.apertures.get(self.parser.aperture, None)
            pin_ref = None if aperture is None or aperture.function != "ComponentPad" else self.parser.refpin
            self.pads.append(
                Pad(
                    aperture=self.parser.aperture,
                    net=self.parser.net,
                    pos=pos,
                    pin_ref=pin_ref,
                    additive=self.parser.additive,
                    mirror=self.parser.mirror,
                    rotation=self.parser.rotation,
                    scale=self.parser.scale,
                )
            )
