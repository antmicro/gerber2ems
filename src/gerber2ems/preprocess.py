"""Module responsible for set up of ports v2 based on gerber files."""

from collections import defaultdict
from math import inf

from gerber2ems.common import Point, mm_to_sim
from gerber2ems.config import Config, PadProp, PortConfig
from gerber2ems.constants import FAB_DIR
from gerber2ems.gerber_io import GerberFile, Pad

cfg = Config()


def extract_ports_from_gerbers(gerbers: dict[int, GerberFile], net_name: list[str]) -> list[PortConfig]:
    """Create port configs for a net.

    For each pad of the net, find the closest ground pads of the same component
    and build a port config referencing them as reference pads.

    """

    def get_board_offset() -> tuple[float, float]:
        """Calculate board offset in gerber files."""
        fab_dir = FAB_DIR
        edge_cuts_path = list(fab_dir.glob("*Edge_Cuts.gbr"))
        if len(edge_cuts_path) == 0:
            exit(1)
        edge_cuts = GerberFile(edge_cuts_path[0])
        [xmin, xmax, ymin, ymax] = [inf, -inf, inf, -inf]
        for seg in edge_cuts.traces["no-net"].segments:
            xmin = min(seg.start.x, seg.stop.x, xmin)
            ymin = min(seg.start.y, seg.stop.y, ymin)
            xmax = max(seg.start.x, seg.stop.x, xmax)
            ymax = max(seg.start.y, seg.stop.y, ymax)
        return xmin, ymin

    offset = Point(*get_board_offset())
    offset.scale(-1)
    # Pads of the requested net, grouped by component
    net_pads_by_comp: dict[str, list[Pad]] = defaultdict(list)
    # All ground pads, grouped by component
    gnd_pads_by_comp: dict[str, list[PadProp]] = defaultdict(list)
    all_gnd = []

    for layer, gerber in gerbers.items():
        for pad in gerber.pads:
            if pad.net in net_name and pad.pin_ref is not None:
                pad.layer = layer
                net_pads_by_comp[pad.pin_ref.comp_ref].append(pad)

        for pad in gerber.pads:
            if pad.net == "GND" and pad.pin_ref is not None:
                pad.layer = layer
                p = pad.to_pad_prop()
                p.position.move(offset)
                p.size.scale(0.7)
                gnd_pads_by_comp[pad.pin_ref.comp_ref].append(p)
                all_gnd.append(p)

    ports: list[PortConfig] = []
    net_w_added_port = []

    def find_gnd(signal_pad: PadProp, gnd_pads: list[PadProp]):
        # Closest distance to any ground pad -> search radius (3x this)
        closest_gnd_distance = min((signal_pad.position.distance_to(gnd.position) for gnd in gnd_pads), default=0.0)
        radius = closest_gnd_distance * 3

        return [p.mm_pad() for p in gnd_pads if p.position.distance_to(signal_pad.position) <= radius]

    for comp_ref, pads in net_pads_by_comp.items():
        gnd_pads = gnd_pads_by_comp.get(comp_ref, [])

        for gerber_pad in pads:
            gerber_pad.pos.move(offset)
            pad = gerber_pad.to_pad_prop()
            size_sq = min(pad.size.x, pad.size.y)
            pad.size = Point(size_sq, size_sq)
            matching_port_from_pnp = next(
                iter([p for p in cfg.ports if p.position and pad.position.distance_to(p.position) < mm_to_sim(0.1)]),
                None,
            )
            print(matching_port_from_pnp)
            if not matching_port_from_pnp:
                # pad not in pnp (probably slicer has mindfully excluded it)
                continue
            if gerber_pad.layer != matching_port_from_pnp.layer:
                # for now THT pads are detected twice in gerber files, use only one that matches simulation.json
                continue

            close_gnd_pads = find_gnd(pad, gnd_pads) if gnd_pads else find_gnd(pad, all_gnd)

            if matching_port_from_pnp.length == 10000 and matching_port_from_pnp.width == 2000:
                matching_port_from_pnp.length = 0
                matching_port_from_pnp.width = 0
            matching_port_from_pnp.length = matching_port_from_pnp.length or 3600
            matching_port_from_pnp.plane = None
            matching_port_from_pnp.pad = pad.mm_pad()
            matching_port_from_pnp.pad.size.scale(0.7)
            if matching_port_from_pnp.width:
                matching_port_from_pnp.pad.size = Point(
                    matching_port_from_pnp.width, matching_port_from_pnp.width
                ).sim_to_mm()
            matching_port_from_pnp.reference_pad = close_gnd_pads
            matching_port_from_pnp.name = (
                str(len(ports) + 1)
                if matching_port_from_pnp.name in ("Unnamed", "", None)
                else matching_port_from_pnp.name
            )
            ports.append(matching_port_from_pnp)
            # ports.append(
            #     PortConfig(
            #         name=str(len(ports) + 1),
            #         length=300,
            #         layer=gerber_pad.layer or 0,
            #         pad=pad,
            #         excite=gerber_pad.net not in net_w_added_port,
            #         reference_pad=close_pads,
            #     )
            # )
            net_w_added_port.append(pad)
    from pprint import pprint

    pprint(ports)

    return sorted(ports, key=lambda p: p.name)
