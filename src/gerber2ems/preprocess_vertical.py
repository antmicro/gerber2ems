"""Module responsible for set up of ports v2 based on gerber files."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from gerber2ems import importer
from gerber2ems.config import Config, PadProp, PortConfig
from gerber2ems.constants import FAB_DIR
from gerber2ems.gerber_io import GerberFile, Pad, get_board_bbox_from_gerbers

cfg = Config()

logger = logging.getLogger(__name__)


def _find_gnd(signal_pad: PadProp, gnd_pads: list[PadProp], dist_mul: float) -> list[PadProp]:
    """Find closest ground pad from `gnd_pads` and other pads that are at at most `dist_mult` this distance."""
    closest_gnd_distance = min((signal_pad.position.distance_to(gnd.position) for gnd in gnd_pads), default=0.0)
    radius = closest_gnd_distance * dist_mul

    return [p.mm_pad() for p in gnd_pads if p.position.distance_to(signal_pad.position) <= radius]


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
                pad.pos.move(offset)
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

    for comp_ref, pads in net_pads_by_comp.items():
        gnd_pads = gnd_pads_by_comp.get(comp_ref, [])

        for gerber_pad in pads:
            pad = gerber_pad.to_pad_prop()

            port_match_from_pnp = next((port for port in cfg.ports if port.matches_pad_pos(pad)), None)
            if not port_match_from_pnp:
                # pad not in pnp (probably slicer has mindfully excluded it)
                continue
            if gerber_pad.layer != port_match_from_pnp.layer:
                # for now THT pads are detected twice in gerber files, use only one that matches simulation.json
                continue
            if port_match_from_pnp.reference_pad:
                close_gnd_pads = []
                for ref in port_match_from_pnp.reference_pad:
                    if isinstance(ref, PadProp):
                        close_gnd_pads.append(ref)
                    else:
                        fpad = next(p.mm_pad() for p in all_gnd if p.id and p.id == ref)
                        if fpad:
                            raise ValueError(f"Pad `{ref}` could not be found in gerber files!")
                        close_gnd_pads.append(fpad)
            else:
                close_gnd_pads = _find_gnd(pad, gnd_pads, 3) if gnd_pads else _find_gnd(pad, all_gnd, 1.2)

            port_match_from_pnp.update_from_pad(pad, close_gnd_pads, str(len(ports) + 1))
            ports.append(port_match_from_pnp)

    return sorted(ports, key=lambda p: p.name)


def preprocess_vertical() -> None:
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
            raise ValueError("`gerber2ems --vertical` requires `trace_under_test` in simulation.json")
        nn = (
            [cfg.trace_under_test.removesuffix("_Diff") + sx for sx in ("_P", "_N")]
            if cfg.trace_under_test.endswith("_Diff")
            else cfg.trace_under_test
        )
    cfg.ports = extract_ports_from_gerbers(
        {idx: GerberFile(next(FAB_DIR.glob("*" + lc.file + ".gbr"))) for idx, lc in enumerate(cfg.get_metals())},
        nn,
    )
    cfg._correct_trace_config()
    cfg.save()
