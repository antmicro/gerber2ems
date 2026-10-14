"""Module responsible for set up of ports v2 based on gerber files."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from gerber2ems import importer
from gerber2ems.common import Point, mm_to_sim
from gerber2ems.config import Config, PadProp, PortConfig
from gerber2ems.constants import FAB_DIR
from gerber2ems.gerber_io import GerberFile, Pad, get_board_bbox_from_gerbers

cfg = Config()

logger = logging.getLogger(__name__)


def extract_ports_from_gerbers(gerbers: dict[int, GerberFile], net_name: list[str]) -> list[PortConfig]:
    """Create port configs for a net.

    For each pad of the net, find the closest ground pads of the same component
    and build a port config referencing them as reference pads.

    """
    offset = get_board_bbox_from_gerbers().min
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

    def find_gnd(signal_pad: PadProp, gnd_pads: list[PadProp], dist_mul: float):
        # Closest distance to any ground pad -> search radius (3x this)
        closest_gnd_distance = min((signal_pad.position.distance_to(gnd.position) for gnd in gnd_pads), default=0.0)
        radius = closest_gnd_distance * dist_mul

        return [p.mm_pad() for p in gnd_pads if p.position.distance_to(signal_pad.position) <= radius]

    for comp_ref, pads in net_pads_by_comp.items():
        gnd_pads = gnd_pads_by_comp.get(comp_ref, [])

        for gerber_pad in pads:
            gerber_pad.pos.move(offset)
            pad = gerber_pad.to_pad_prop()
            size_sq = min(pad.size.x, pad.size.y)
            pad.size = Point(size_sq, size_sq)
            port_match_from_pnp = next(
                iter([p for p in cfg.ports if p.position and pad.position.distance_to(p.position) < mm_to_sim(0.1)]),
                None,
            )
            if not port_match_from_pnp:
                # pad not in pnp (probably slicer has mindfully excluded it)
                continue
            if gerber_pad.layer != port_match_from_pnp.layer:
                # for now THT pads are detected twice in gerber files, use only one that matches simulation.json
                continue

            close_gnd_pads = find_gnd(pad, gnd_pads, 3) if gnd_pads else find_gnd(pad, all_gnd, 1.2)

            if not port_match_from_pnp.pad:
                pad.size.scale(0.7)  # smaller port than pad to fit into circular pads
                port_match_from_pnp.pad = pad.mm_pad()
                if port_match_from_pnp.length == 10000 and port_match_from_pnp.width == 2000:
                    port_match_from_pnp.length = 0
                    port_match_from_pnp.width = 0
                w = port_match_from_pnp.width
                if w:
                    port_match_from_pnp.pad.size = Point(w, w).sim_to_mm()

            port_match_from_pnp.length = port_match_from_pnp.length or 3600
            port_match_from_pnp.plane = None
            port_match_from_pnp.reference_pad = close_gnd_pads
            port_match_from_pnp.name = (
                str(len(ports) + 1) if port_match_from_pnp.name in ("Unnamed", "", None) else port_match_from_pnp.name
            )
            ports.append(port_match_from_pnp)

    return sorted(ports, key=lambda p: p.name)


def preprocess() -> None:
    """Run preprocess of simulation data.

    Currently preprocessing includes:
    * update ports in config based on PNP (with ports) and gerbers
        (this detects and save position/size if simulation & reference pads)
    """
    logger.info("Preprocessing board before simulation")
    importer.import_stackup()
    importer.import_port_positions()
    netinfo_path = Path.cwd() / "netinfo.json"
    if netinfo_path.exists():
        netinfo = json.load(open(netinfo_path))
        nn = [net["name"].replace(",", r"\u002C") for net in netinfo["nets"]]
    else:
        if not cfg.trace_under_test:
            raise ValueError("`gerber2ems --preprocess` requires `trace_under_test` in simulation.json")
        nn = (
            [cfg.trace_under_test.removesuffix("_Diff") + sx for sx in ("_P", "_N")]
            if cfg.trace_under_test.endswith("_Diff")
            else cfg.trace_under_test
        )
    cfg.ports = extract_ports_from_gerbers(
        {idx: GerberFile(next(FAB_DIR.glob("*" + lc.file + ".gbr"))) for idx, lc in enumerate(cfg.get_metals())},
        nn,
    )
    cfg.save()
