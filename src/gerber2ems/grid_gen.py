"""Module provides classes that enable dynamic simulation grid generation."""

from gerber2ems.gerber_io import (
    GerberFile,
    TraceSegment,
    Trace,
    Position,
    Pad,
    Aperture,
    PlotMode,
)
from gerber2ems.config import Config
from CSXCAD.CSRectGrid import CSRectGrid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import json
from math import atan2, pi, floor, log, inf
import logging
import sys
from copy import deepcopy
from scipy.optimize import fsolve
import numpy as np
from functools import partial

logger = logging.getLogger(__name__)
cfg = Config()


def dedup_grid(grid: List[float], grid_min: float, edge_grid: List[float]) -> List[float]:
    """Merge grid lines that are to close to each other."""
    to_remove = []
    grid.sort()
    for idx in range(len(grid) - 1):
        if grid[idx + 1] - grid[idx] < grid_min / 2:
            grid[idx + 1] = (grid[idx + 1] + grid[idx]) / 2
            if grid[idx] not in edge_grid:
                to_remove.append(idx)
            elif grid[idx + 1] not in edge_grid:
                to_remove.append(idx + 1)
    for idx in reversed(to_remove):
        grid.pop(idx)
    return grid


class GridGenerator:
    """Class that manages grid generation."""

    def __init__(self) -> None:
        """Create GridGenerator object."""
        pcb_width = cfg.pcb_width
        pcb_height = cfg.pcb_height
        margin = cfg.grid.margin.xy
        if pcb_width is None or pcb_height is None:
            logger.error("PCB dimensions are not set")
            sys.exit(1)
        self.x = GridGeneratorAxis("x", Region(-margin, pcb_width + margin))
        self.y = GridGeneratorAxis("y", Region(-margin, pcb_height + margin))
        self.add_pads: List[Pad] = []
        self.add_apertures: Dict[str, Aperture] = {}
        self.get_board_offset()

    def get_board_offset(self) -> None:
        """Calculate board offset in gerber files."""
        fab_dir = Path.cwd() / "fab"
        edge_cuts_path = list(fab_dir.glob("*Edge_Cuts.gbr"))
        if len(edge_cuts_path) == 0:
            logger.error(f"No EdgeCuts gerber in fab dir({fab_dir})")
            exit(1)
        edge_cuts = GerberFile(edge_cuts_path[0])
        [self.xmin, self.xmax, self.ymin, self.ymax] = [inf, -inf, inf, -inf]
        for seg in edge_cuts.traces["no-net"].segments:
            self.xmin = min(seg.start.x, seg.stop.x, self.xmin)
            self.ymin = min(seg.start.y, seg.stop.y, self.ymin)
            self.xmax = max(seg.start.x, seg.stop.x, self.xmax)
            self.ymax = max(seg.start.y, seg.stop.y, self.ymax)

    def generate_z(self) -> List[int]:
        """Generate grid in Z axis."""
        logger.info("### Grid Generator: generate Z axis ###")
        cell_ratio = cfg.grid.cell_ratio.z
        grid_min = cfg.grid.optimal / cell_ratio
        grid_max = cfg.grid.max
        margin = cfg.grid.margin.z
        z_count = cfg.grid.inter_layers
        if z_count % 2 == 1:  # Increasing by one to always have z_line at dumpbox
            z_count += 1

        z_lines_np = np.array([0])
        offset = 0
        for layer in cfg.get_substrates():
            z_lines_np = np.concatenate(
                (
                    z_lines_np,
                    np.linspace(offset - layer.thickness, offset, z_count, endpoint=False),
                )
            )
            offset -= layer.thickness
        z_lines = list(z_lines_np)
        (zmin, zmax) = min(z_lines), max(z_lines)
        z_lines.extend([cfg.grid.margin.z, offset - cfg.grid.margin.z])

        # Smooth grid
        z_lines = Region(zmin, zmax).densify_region_grid(list(z_lines), grid_max, grid_min, cell_ratio)
        z_lines = Region(zmax, margin).densify_region_grid(list(z_lines), grid_max, grid_min, cell_ratio)
        z_lines = Region(offset - margin, zmin).densify_region_grid(list(z_lines), grid_max, grid_min, cell_ratio)
        return [int(x) for x in dedup_grid(z_lines, grid_min, [])]

    def generate(self, grid: CSRectGrid) -> CSRectGrid:
        """Generate complete dynamic grid."""
        fab_dir = Path.cwd() / "fab"
        gerbers = [GerberFile(gbr_path) for gbr_path in fab_dir.glob("*_Cu.gbr")]

        logger.info("### Grid Generator: get nets of interest ###")
        n = [t.nets for t in cfg.diff_pairs] + [t.nets for t in cfg.traces]
        nets = [net for sublist in n for net in sublist]
        if len(nets) == 0:
            netinfo_path = Path.cwd() / "netinfo.json"
            try:
                netinfo = json.load(open(netinfo_path))
                nets = [net["name"] for net in netinfo["nets"]]
            except FileNotFoundError:
                logger.warning(f"File with nets under test not found! ({netinfo_path})")
                nets_s = set()
                for gbr in gerbers:
                    nets_s.update(gbr.traces.keys())
                nets = [net for net in nets_s if net not in ["GND", "gnd", "no-net"]]

        logger.info("### Grid Generator: parse gerber files ###")
        for gbr in gerbers:
            for net in nets:
                trace = gbr.traces.get(net, Trace([]))
                self.x.add_lines_from_trace(trace.segments)
                self.y.add_lines_from_trace(trace.segments)
            pads = gbr.pads + self.add_pads
            gbr.apertures.update(self.add_apertures)
            nets.append("PORT")
            self.x.add_lines_from_pads(pads, nets, gbr.apertures)
            self.y.add_lines_from_pads(pads, nets, gbr.apertures)

        logger.info("### Grid Generator: generate X axis ###")
        grid = self.x.compile_grid(grid, self.xmin)
        logger.info("### Grid Generator: generate Y axis ###")
        grid = self.y.compile_grid(grid, self.ymin)

        # process grid in Z axis

        grid.AddLine("z", self.generate_z())

        return grid


@dataclass(unsafe_hash=True)
class SubRegion:
    """Part of region that consists of 4 grid lines.

    (and 3 cells in between, with center cell to be divided to conform with grid rules)
    """

    start_idx: int
    """Index of first grid line of subregion"""
    reg_min: float
    """Minimal position of line in parent region"""
    reg_max: float
    """Maximal position of line in parent region"""
    grid_size: float
    """Target grid size"""
    grid_size_min: float
    """Optimal grid may reduce grid_size up to this value temporally"""
    grid_size_abs_min: float
    """Grid size never should go bellow this value"""
    cell_ratio: float
    """Consecutive cells optimally should differ in size by this factor"""
    edge_l: bool
    """Indicates that this is leftmost subregion of region"""
    edge_h: bool
    """Indicates that this is rightmost subregion of region"""
    lines: List[float]
    """4 grid lines that are area of interest of this subregion"""
    bounded_l: bool = True
    """Does size of cell on the left side needs to be taken into account during cell ratio check?"""
    bounded_h: bool = True
    """Does size of cell on the right side needs to be taken into account during cell ratio check?"""
    prev_size: float = 0
    """Size of previous cell (leftmost cell of subregion, between `lines[0]`&`lines[1]`)"""
    next_size: float = 0
    """Size of next cell (rightmost cell of subregion, between `lines[2]`&`lines[3]`)"""
    dist: float = 0
    """Size of cell that is being processed (center cell of subregion, between `lines[1]`&`lines[2]`)"""
    m_q_sgn: int = 1
    """Right geometric series is 1:growing; -1:shrinking"""
    n_q_sgn: int = 1
    """Left geometric series is 1:growing; -1:shrinking"""
    m_opt: int = 0
    """Number of cells created by right geom. series"""
    n_opt: int = 0
    """Number of cells created by left geom. series"""
    final_cell_ratio: float = 0
    """Calculated optimal geometric series factor"""
    k: int = 0
    """Number of cells in evenly spaced region"""

    def add_border_lines(self, grid: List[float]) -> None:
        """Add lines on subregion borders."""
        self.bounded_l = (
            abs(self.lines[1] - self.lines[0]) > 1e-6 and self.lines[1] - self.lines[0] < self.grid_size * 3
        )
        self.bounded_h = (
            abs(self.lines[2] - self.lines[3]) > 1e-6 and self.lines[3] - self.lines[2] < self.grid_size * 3
        )

        region_end_line = min(self.lines[2] - self.grid_size, self.reg_min)
        if self.edge_l and region_end_line > self.lines[1] + self.grid_size:
            # self.lines[1] is outside region -> Add line near the end of region
            self.lines[0] = self.lines[1]
            self.lines[1] = region_end_line
            grid.insert(self.start_idx + 2, region_end_line)
            self.start_idx += 1
            self.edge_l = False
        region_end_line = max(self.lines[1] + self.grid_size, self.reg_max)
        if self.edge_h and region_end_line < self.lines[2] - self.grid_size:
            # self.lines[2] is outside region -> Add line near the end of region
            self.lines[3] = self.lines[2]
            self.lines[2] = region_end_line
            grid.insert(self.start_idx + 2, region_end_line)
            self.edge_h = False
        self.bounded_l = (
            abs(self.lines[1] - self.lines[0]) > 1e-6 and self.lines[1] - self.lines[0] < self.grid_size * 3
        )
        self.bounded_h = (
            abs(self.lines[2] - self.lines[3]) > 1e-6 and self.lines[3] - self.lines[2] < self.grid_size * 3
        )

        self.prev_size = abs(self.lines[1] - self.lines[0]) if self.bounded_l else self.grid_size
        self.next_size = abs(self.lines[3] - self.lines[2]) if self.bounded_h else self.grid_size

        if (
            self.bounded_l
            and self.prev_size > self.grid_size + self.grid_size_min
            and self.reg_min < self.lines[1]
            and self.reg_min > self.lines[0]
        ):
            # self.lines[0] is outside region and far away from self.lines[1] -> Add line near the end of region
            self.lines[0] = min(self.reg_min, self.lines[1] - self.grid_size)
            grid.insert(self.start_idx + 1, self.lines[0])
            self.start_idx += 1
            self.prev_size = self.lines[1] - self.lines[0]
        if (
            self.bounded_h
            and self.next_size > self.grid_size + self.grid_size_min
            and self.reg_max > self.lines[2]
            and self.reg_max < self.lines[3]
        ):
            # self.lines[3] is outside region and far away from self.lines[2] -> Add line near the end of region
            self.lines[3] = max(self.reg_max, self.lines[2] + self.grid_size)
            grid.insert(self.start_idx + 3, self.lines[3])
            self.next_size = self.lines[3] - self.lines[2]

        self.dist = min(self.reg_max, self.lines[2]) - max(self.reg_min, self.lines[1])
        self.m_q_sgn = -1 if self.next_size > self.grid_size else 1
        self.n_q_sgn = -1 if self.prev_size > self.grid_size else 1

    def ready(self) -> bool:
        """Check if subregion needs further processing."""
        return self.dist < self.grid_size_abs_min * 2 or (
            (self.dist < (self.prev_size * self.cell_ratio * 1.1) or not self.bounded_l)
            and (self.dist < (self.next_size * self.cell_ratio * 1.1) or not self.bounded_h)
            and (self.dist < self.grid_size_min * 2)
        )

    def calc_left_dist(self, m_opt: int, n_opt: int, q: float) -> float:
        """Calculate distance left after filling subregion with two geometric series."""
        qm = q**self.m_q_sgn
        qn = q**self.n_q_sgn
        return (
            self.dist
            - self.next_size * qm * (1 - qm**m_opt) / (1 - qm)
            - self.prev_size * qn * (1 - qn**n_opt) / (1 - qn)
        )

    def try_geometric_fill(self, cell_ratio_mul: float = 1) -> bool:
        """Check if possible and calculate parameters for fill with two geometric series and evenly spaced part."""
        m_opt = floor(log(self.grid_size / self.next_size, self.cell_ratio) * self.m_q_sgn)
        m_opt = max(m_opt, 0) if self.bounded_h else 0

        n_opt = (
            max(floor(log(self.grid_size / self.prev_size, self.cell_ratio) * self.n_q_sgn), 0) if self.bounded_l else 0
        )
        break_cond = False

        for _ in range(n_opt + m_opt):

            calc_left_dist = partial(self.calc_left_dist, m_opt, n_opt)
            left_dist = calc_left_dist(self.cell_ratio)
            k = 0
            if left_dist >= -self.grid_size:
                k = max(0, floor(left_dist / self.grid_size))

                def series_sum2(q: float) -> float:
                    return calc_left_dist(q) - k * self.grid_size  # noqa: B023

                q1 = fsolve(series_sum2, self.cell_ratio)[0]
                q1norm = q1 if q1 > 1 else 1 / q1
                if 1 < q1norm < self.cell_ratio * cell_ratio_mul:
                    left_dist = calc_left_dist(q1)
                    k = max(0, floor(left_dist / (self.prev_size * (q1 ** (self.n_q_sgn * n_opt)))))
                    if k == 0:
                        # last line is the same as from `m`-series
                        if n_opt != 0:
                            n_opt -= 1
                        else:
                            m_opt = max(m_opt - 1, 0)
                    break_cond = True
                    if (
                        q1 > self.final_cell_ratio
                        and q1norm <= self.cell_ratio * cell_ratio_mul
                        or self.final_cell_ratio == 0
                    ):
                        (self.m_opt, self.n_opt, self.k, self.final_cell_ratio) = (m_opt, n_opt, k, q1)
                    continue
            sprev = self.prev_size * (self.cell_ratio ** (self.n_q_sgn * n_opt))
            snext = self.next_size * (self.cell_ratio ** (self.m_q_sgn * m_opt))
            if sprev > snext:
                n_opt = max(n_opt - 1, 0)
            else:
                m_opt = max(m_opt - 1, 0)
        return break_cond

    def try_regular_fill(self) -> bool:
        """Check if possible and calculate parameters for fill with evenly spaced lines."""
        unbounded = not self.bounded_h and not self.bounded_l
        reg_grid_possible = (1 / self.cell_ratio) <= (self.prev_size / self.next_size) <= self.cell_ratio or unbounded
        if not reg_grid_possible:
            return False
        self.m_opt = 0
        self.n_opt = 0
        if self.bounded_h and self.bounded_l:
            g = (self.prev_size + self.next_size) / 2
        elif not self.bounded_h and self.bounded_l:
            g = self.prev_size
        elif self.bounded_h and not self.bounded_l:
            g = self.next_size
        else:
            g = self.grid_size
        self.k = floor(self.dist / g)
        return True

    def find_any_fill(self) -> bool:
        """Try to match any fill pattern in subregion."""
        self.k = 0
        if not self.bounded_l:
            self.n_opt = 0
            self.m_opt = max(floor(log((1 - self.dist * (1 - self.cell_ratio) / self.next_size), self.cell_ratio)), 0)
            q1 = self.cell_ratio
        elif not self.bounded_h:
            self.m_opt = 0
            self.n_opt = max(floor(log((1 - self.dist * (1 - self.cell_ratio) / self.prev_size), self.cell_ratio)), 0)
            q1 = self.cell_ratio
        elif self.prev_size >= self.dist or self.next_size >= self.dist:
            # Subregion is too small to add any lines
            self.m_opt = 0
            self.n_opt = 0
            self.k = 0
            return True
        else:
            # solution with optimal cell ratio cannot be found, try to get best possible cell ratio
            # Try to fill subregion with single geometric series that scales from `prev_size` to `next_size` directly
            self.m_opt = 0
            self.n_q_sgn = -1 if self.prev_size > self.next_size else 1
            # dist = prev_size*(1-q^n)/(1-q) + next_size*(1-q^m)/(1-q) + grid_size*k
            # m=0 k=0 -> dist = prev_size * (1-q^n)/(1-q)
            # next_size=prev_size*q^(n-1) -> q^n=q*next_size/prev_size
            # 1-q = prev_size * (1 - q*next_size/prev_size)/dist
            # 1-q = prev_size/dist - q*next_size/dist
            q1 = (1 - self.prev_size / self.dist) / (1 - self.next_size / self.dist)
            self.n_opt = max(round(log(self.next_size / self.prev_size, q1)), 0)
            q1 = q1**self.n_q_sgn

        self.final_cell_ratio = fsolve(partial(self.calc_left_dist, self.m_opt, self.n_opt), q1)[0]
        return True

    def fill_lines(self, grid: List[float]) -> int:
        """Add lines to `grid` using previously calculated parameters (`m_opt`, `n_opt`, `k`, `final_cell_ratio`)."""
        idx = self.start_idx
        for _ in range(0, self.n_opt):
            self.lines[0] = self.lines[1]
            self.prev_size *= self.final_cell_ratio**self.n_q_sgn
            self.lines[1] += min(self.grid_size, self.prev_size)
            grid.insert(idx + 2, self.lines[1])
            idx += 1
        for _ in range(0, self.m_opt):
            self.lines[3] = self.lines[2]
            self.next_size *= self.final_cell_ratio**self.m_q_sgn
            self.lines[2] -= min(self.grid_size, self.next_size)
            grid.insert(idx + 2, self.lines[2])

        step = (self.lines[2] - self.lines[1]) / self.k if self.k != 0 else 0
        for _ in range(0, self.k - 1):
            self.lines[1] += step
            grid.insert(idx + 2, self.lines[1])
            idx += 1
        return idx + self.m_opt


@dataclass(unsafe_hash=True)
class Region:
    """Class stores Range with additional data as priority and center position."""

    min: float
    """Minimal/border value of a region"""
    max: float
    """Maximal/border value of a region"""
    prio: float = 1
    """Region's level of importance"""
    center: float = 0
    """Region's center point"""

    def distance(self, rhs: "Region") -> float:
        """Calculate distance to other Region.

        (subzero values indicate how much one of regions need to be moved to eliminate overlap)
        """
        if self.min > rhs.max:
            return self.min - rhs.max
        if self.max < rhs.min:
            return rhs.min - self.max
        return min(rhs.min - self.max, self.min - rhs.max)

    def resize(self, size: float) -> "Region":
        """Scale region to requested size (using self.center as origin)."""
        fac = abs(size / (self.max - self.min))
        self.min = self.center - fac * abs(self.center - self.min)
        self.max = self.center + fac * abs(self.center - self.max)
        return self

    def densify_region_grid(
        self, grid: List[float], grid_size: float, abs_min: float, cell_ratio: float
    ) -> List[float]:
        """Check if grid is dense enough within a region & follows optimal scaling, if not add new grid lines to fix it.

        It is assumed that calls on overlapping regions will be done with not decreasing grid_size.

        Rules for grid generation:
        1. neighboring cells should not differ too much (max factor: q)
        2. cells between lines[0]&lines[1] and lines[2]&lines[3] are not larger than target cell size
        3. Target: fill space between lines[1]&lines[2]
        4. Target: new cells should be as large as possible (but smaller than grid_size)
        5. To reach 3&4 following cell size should increase at first then decrease
        eg. |.|. . . . . . . . . . . . . . . .|.|  (q=2, grid_size>=4)
         -> |.|. .|. . . .|. . . .|. . . .|. .|.|
        Space that we want to divide can be described via dwo geometric series
        (that change cell size that was beyond subregion (`prev_size`&`next_size`) to `grid_size`)
        and rest that is evenly partitioned with each cell size close to `grid_size`,
        This is described using following formulas:
        dist = prev_size*(1-q^n)/(1-q) + next_size*(1-q^m)/(1-q) + grid_size*k
        grid_size >= prev_size*q^(n-1)
        grid_size >= next_size*q^(m-1)
        n - number of cells with increasing cell size (2 in above example)
        m - number of cells with decreasing cell size (eg. 2)
        k - number of cells with const cell size
        q - cell_ratio (how much two consecutive cells can differ)
        """
        grid_min = grid_size / cell_ratio
        grid.sort()
        grid.insert(0, grid[0])
        grid.append(grid[-1])

        idx = -1
        (edge_l, edge_h) = (True, True)
        while True:
            # There are some grid lines already present.
            # Iterate over spaces between these lines and check if each is small enough
            idx += 1
            if idx >= len(grid) - 3:
                break

            subreg = SubRegion(
                idx,
                self.min,
                self.max,
                grid_size,
                grid_min,
                abs_min,
                cell_ratio,
                edge_l,
                edge_h,
                deepcopy(grid[idx : idx + 4]),
            )

            if self.min >= subreg.lines[2] or self.max <= subreg.lines[1]:
                # lines out of region of interest
                continue

            subreg.add_border_lines(grid)
            (edge_l, edge_h) = subreg.edge_l, subreg.edge_h

            if subreg.ready():
                continue

            fill_methods: List[Callable] = [
                subreg.try_geometric_fill,
                partial(subreg.try_geometric_fill, 1.5),
                subreg.try_regular_fill,
                subreg.find_any_fill,
            ]
            for method in fill_methods:
                if method():
                    break

            idx = subreg.fill_lines(grid)

        return grid[1:-1]


@dataclass
class GridGeneratorAxis:
    """Grid generation responsible for generating grid lines in single dimmension."""

    axis: str
    """Axis on which this generator works with"""
    board: Region
    """Region that describes size of simulation space"""
    edge_cells: List[Region] = field(default_factory=list)
    """Cells (each described by 2 lines) that are containing track edge- they follow the rule of thirds"""
    parallel: List[Region] = field(default_factory=list)
    """Regions with track parallel to grid"""
    diagonal: List[Region] = field(default_factory=list)
    """Regions with tracks that are diagonal to grid axis"""
    perpendicular: List[Region] = field(default_factory=list)
    """Regions with tracks perpendicular to grid axis"""

    def add_lines_from_trace(self, segments: List[TraceSegment]) -> None:
        """Add traces that should be considered during grid generation."""
        oaxis = "x" if self.axis == "y" else "y"
        w3 = cfg.grid.optimal / 3
        for seg in segments:
            slen = Position(abs(seg.start.x - seg.stop.x), abs(seg.start.y - seg.stop.y))
            ang = atan2(slen.y, slen.x)
            deg5 = pi / 36  # angle smaller than 5 degrees

            p = [getattr(seg.start, self.axis), getattr(seg.stop, self.axis)]
            region = Region(
                min(p[0], p[1]) - seg.width / 2,
                max(p[0], p[1]) + seg.width / 2,
                getattr(slen, self.axis) + 1,
                (p[0] + p[1]) / 2,
            )
            if seg.mode != PlotMode.LINEAR:
                self.diagonal.append(region)
                continue

            if (ang < deg5 and self.axis == "x") or (ang > pi / 2 - deg5 and self.axis == "y"):
                self.perpendicular.append(region)
            elif (ang < deg5 and self.axis == "y") or (ang > pi / 2 - deg5 and self.axis == "x"):
                self.parallel.append(region)
                if seg.width != 0 or seg.normal:
                    edge_bot = Region(region.min - 2 * w3, region.min + w3, getattr(slen, oaxis) + 1, region.min)
                    self.edge_cells.append(edge_bot)
                if seg.width != 0 or not seg.normal:
                    edge_top = Region(region.max - w3, region.max + 2 * w3, getattr(slen, oaxis) + 1, region.max)
                    self.edge_cells.append(edge_top)
            else:
                self.diagonal.append(region)

    def add_lines_from_pads(self, pads: List[Pad], nets: List[str], apertures: Dict[str, Aperture]) -> None:
        """Add pads/vias that should be considered during grid generation."""
        for pad in pads:
            if pad.net not in nets:
                continue
            ap = apertures[pad.aperture]
            cont = ap.data.contours(pad.pos, pad.rotation, pad.scale, pad.mirror)
            self.add_lines_from_trace(cont)

    def resolve_edge_regions(self) -> None:
        """Shrink/Merge conflicting edge regions (try to follow the rule of thirds as closely as possible)."""
        grid_size = cfg.grid.optimal
        grid_min = grid_size / 1.8
        self.edge_cells.sort(key=lambda k: k.prio, reverse=True)  # sort by prio: high to low
        to_delete = []
        for i in range(0, len(self.edge_cells)):
            reg = self.edge_cells[i]
            for j in range(0, i):
                if j in to_delete:
                    continue
                reg2 = self.edge_cells[j]
                dist = reg.distance(reg2)
                if dist > grid_min:
                    # regions not in conflict
                    continue
                size = [reg.max - reg.min, reg2.max - reg2.min]
                shrink_potential = size[0] + size[1] - 2 * grid_min
                if dist + shrink_potential > grid_min:
                    # regions can be shrunk
                    reg = reg.resize(grid_min)
                    reg2 = reg2.resize(grid_min)
                    dist = reg.distance(reg2)
                    if dist > grid_min:
                        # regions no longer in conflict
                        continue

                # Replace regions with an averaged one
                new_reg = Region(
                    (reg.min * reg.prio + reg2.min * reg2.prio) / (reg.prio + reg2.prio),
                    (reg.max * reg.prio + reg2.max * reg2.prio) / (reg.prio + reg2.prio),
                    reg.prio + reg2.prio,
                    (reg.center * reg.prio + reg2.center * reg2.prio) / (reg.prio + reg2.prio),
                )
                to_delete.append(j)
                self.edge_cells[i] = new_reg
                self.edge_cells[j] = new_reg

        to_delete.sort(reverse=True)
        for idx in to_delete:
            self.edge_cells.pop(idx)
        self.edge_cells = list(set(self.edge_cells))

    def merge_regions(self, reg_field: str, grid_size: float) -> None:
        """Merge overlapping regions."""
        reg_list = getattr(self, reg_field)
        to_delete = []
        for i in range(0, len(reg_list)):
            reg = reg_list[i]
            for j in range(0, i):
                if j in to_delete:
                    continue
                reg2 = reg_list[j]
                if reg.distance(reg2) < grid_size:
                    (nmin, nmax) = (min(reg.min, reg2.min), max(reg.max, reg2.max))
                    new_reg = Region(nmin, nmax, reg.prio + reg2.prio, (nmin + nmax) / 2)
                    to_delete.append(j)
                    reg = new_reg
                    reg2 = deepcopy(new_reg)
                    reg_list[i] = reg
                    reg_list[j] = reg2
        to_delete.sort(reverse=True)
        for idx in to_delete:
            reg_list.pop(idx)
        setattr(self, reg_field, reg_list)

    def compile_grid(self, csgrid: CSRectGrid, offset: float) -> CSRectGrid:
        """Generate targeted grid based on added traces/pads.

        - `offset` - offset between coordinates in gerber file and metal patterns imported from images
        """
        csgrid.ClearLines(self.axis)
        grid_size = cfg.grid.optimal
        cell_ratio = cfg.grid.cell_ratio.xy
        grid_min = grid_size / cell_ratio
        grid_diag = cfg.grid.diagonal
        grid_perp = cfg.grid.perpendicular

        self.resolve_edge_regions()
        grid = []
        if cfg.grid.margin.from_trace:
            self.board.min = inf
            self.board.max = -inf
            for r in self.edge_cells + self.diagonal + self.parallel + self.perpendicular:
                self.board.min = min(self.board.min, r.min)
                self.board.max = max(self.board.max, r.max)
            margin = cfg.grid.margin.xy
            self.board.min -= margin
            self.board.max += margin
        else:
            self.board.min += offset
            self.board.max += offset

        for reg in self.edge_cells:
            grid.append(reg.min)
            grid.append(reg.max)

        edge_grid = deepcopy(grid)
        grid.append(self.board.min)
        grid.append(self.board.max)

        self.merge_regions("parallel", grid_size)
        self.merge_regions("perpendicular", grid_perp)
        self.merge_regions("diagonal", grid_diag)

        for reg in self.parallel:
            grid = reg.densify_region_grid(grid, grid_size, grid_min, cell_ratio)
        grid = dedup_grid(grid, grid_min, edge_grid)

        for reg in self.diagonal:
            grid = reg.densify_region_grid(grid, grid_diag, grid_min, cell_ratio)
        grid = dedup_grid(grid, grid_min, edge_grid)

        for reg in self.perpendicular:
            grid = reg.densify_region_grid(grid, grid_perp, grid_min, cell_ratio)
        grid = dedup_grid(grid, grid_min, edge_grid)

        grid = self.board.densify_region_grid(grid, cfg.grid.max, grid_min, cell_ratio)
        grid = dedup_grid(grid, grid_min, edge_grid)

        grid = [line - offset for line in grid]

        csgrid.AddLine(self.axis, [int(x) for x in grid])
        return csgrid
