"""Module responsible for set up of ports vertical based on gerber files."""

import logging
from collections import defaultdict

from gerber2ems import importer
from gerber2ems.config import Config, PadProp, PortConfig, PortType
from gerber2ems.constants import FAB_DIR
from gerber2ems.gerber_io import GerberFile, get_board_bbox_from_gerbers

cfg = Config()

logger = logging.getLogger(__name__)


def _find_gnd(signal_pad: PadProp, gnd_pads: list[PadProp], dist_mul: float) -> list[PadProp]:
    """Find closest ground pad from `gnd_pads` and other pads that are at at most `dist_mult` this distance."""
    gnd_pads_sl = [p for p in gnd_pads if p.layer == signal_pad.layer]
    closest_gnd_distance = min((signal_pad.position.distance_to(gnd.position) for gnd in gnd_pads_sl), default=0.0)
    radius = closest_gnd_distance * dist_mul

    return [p for p in gnd_pads_sl if p.position.distance_to(signal_pad.position) <= radius]


def extract_ports_from_gerbers(gerbers: dict[int, GerberFile]) -> list[PortConfig]:
    """Create port configs.

    For each pad, find the closest ground pads of the same component
    and build a port config referencing them as reference pads.

    """
    offset = get_board_bbox_from_gerbers().min
    offset.scale(-1)
    # Pads of the requested net, grouped by component
    net_pads_by_comp: dict[str, list[PadProp]] = defaultdict(list)
    # All ground pads, grouped by component
    gnd_pads_by_comp: dict[str, list[PadProp]] = defaultdict(list)
    all_gnd = []

    for layer, gerber in gerbers.items():
        for pad in gerber.pads:
            if pad.net != "GND" and pad.pin_ref is not None:
                pad.layer = layer
                pad.pos.move(offset)
                p = pad.to_pad_prop().mm_pad_port()
                net_pads_by_comp[pad.pin_ref.comp_ref].append(p)

        for pad in gerber.pads:
            if pad.net == "GND" and pad.pin_ref is not None:
                pad.layer = layer
                pad.pos.move(offset)
                p = pad.to_pad_prop().mm_pad_port()
                gnd_pads_by_comp[pad.pin_ref.comp_ref].append(p)
                all_gnd.append(p)

    ports: list[PortConfig] = []
    for port in cfg.all_ports:
        if port.type != PortType.VERTICAL:
            ports.append(port)
            continue

        for comp_ref, pads in net_pads_by_comp.items():
            gnd_pads = gnd_pads_by_comp.get(comp_ref, [])

            for pad in pads:
                if not port.matches_pad_pos(pad):
                    continue
                if pad.layer != port.layer:
                    # for now THT pads are detected twice in gerber files, use only one that matches simulation.json
                    continue
                close_gnd_pads: list[PadProp] = []
                if port.reference_pad:
                    proc_refs = set()
                    for ref in port.reference_pad:
                        if isinstance(ref, PadProp):
                            ref = ref.id
                        if ref in proc_refs:
                            continue
                        fpad = tuple(p for p in all_gnd if p.layer == pad.layer and p.id and p.id == ref)
                        proc_refs.add(ref)
                        if not fpad:
                            raise ValueError(f"Pad `{ref}` could not be found in gerber files!")
                        close_gnd_pads.extend(fpad)
                if not close_gnd_pads:
                    close_gnd_pads = _find_gnd(pad, gnd_pads, 3) if gnd_pads else _find_gnd(pad, all_gnd, 1.2)
                port.update_from_pad(pad, close_gnd_pads, str(len(ports) + 1))
                ports.append(port)

    return sorted(ports, key=lambda p: int(p.name))


def preprocess_vertical() -> None:
    """Run preprocess of simulation data.

    Currently preprocessing includes:
    * update ports in config based on PNP (with ports) and gerbers
        (this detects and save position/size of simulation & reference pads)
    """
    logger.info("Preprocessing board before simulation")
    importer.import_stackup()
    importer.import_port_positions()

    cfg.all_ports = extract_ports_from_gerbers(
        {idx: GerberFile(next(FAB_DIR.glob("*" + lc.file + ".gbr"))) for idx, lc in enumerate(cfg.get_metals())}
    )
    cfg._correct_trace_config()
    cfg.save()
