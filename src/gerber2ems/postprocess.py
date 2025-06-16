"""Module contains functions useful for postprocessing data."""

from typing import Union, Tuple, Optional
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import skrf

from gerber2ems.config import Config
from gerber2ems.constants import RESULTS_DIR, PLOT_STYLE

logger = logging.getLogger(__name__)
cfg = Config()


class Postprocesor:
    """Class used to postprocess and display simulation data."""

    def __init__(self, frequencies: np.ndarray, port_count: int) -> None:
        """Initialize postprocessor."""
        self.frequencies = frequencies  # Frequency list for whitch parameters are calculated
        self.count = port_count  # Number of ports

        self.incident = np.empty(
            [self.count, self.count, len(self.frequencies)], np.complex128
        )  # Incident wave phasor table ([measured_port][excited_port][frequency])
        self.incident[:] = np.nan
        self.reflected = np.empty(
            [self.count, self.count, len(self.frequencies)], np.complex128
        )  # Reflected wave phasors table ([measured_port][excited_port][frequency])
        self.reflected[:] = np.nan
        self.reference_zs = np.empty([self.count], np.complex128)
        self.reference_zs[:] = np.nan  # Reference impedances of ports

        self.s_params = np.empty(
            [self.count, self.count, len(self.frequencies)], np.complex128
        )  # S-parameter table ([output_port][input_port][frequency])
        self.s_params[:] = np.nan
        self.impedances = np.empty([self.count, len(self.frequencies)], np.complex128)
        self.impedances[:] = np.nan
        self.delays = np.empty(
            [self.count, self.count, len(self.frequencies)], np.float64
        )  # Group delay table ([output_port][input_port][frequency])
        self.delays_valid = False

    def add_port_data(
        self,
        port: int,
        excited_port: int,
        incident: np.ndarray,
        reflected: np.ndarray,
    ) -> None:
        """Add port data to postprocessor.

        Data consists of incident and reflected phasor data in relation to frequency
        """
        if self.is_valid(self.incident[port][excited_port]):
            logger.warning("This port data has already been supplied, overwriting")
        self.incident[port][excited_port] = incident
        self.reflected[port][excited_port] = reflected

    def add_impedances(self, impedances: np.ndarray) -> None:
        """Add port reference impedances."""
        self.reference_zs = impedances

    def calculate_sparams(self) -> None:
        """Calculate all needed parameters for further processing. Should be called after all ports are added."""
        logger.info("Processing all data from simulation. Calculating S-parameters and impedance")
        for i, _ in enumerate(self.incident):
            if self.is_valid(self.incident[i][i]):
                for j, _ in enumerate(self.incident):
                    if self.is_valid(self.reflected[j][i]):
                        self.s_params[j][i] = self.reflected[j][i] / self.incident[i][i]

    def process_data(self) -> None:
        """Calculate all needed parameters for further processing. Should be called after all ports are added."""
        logger.info("Processing all data from simulation. Calculating Delay & Impedance")

        for i, reference_z in enumerate(self.reference_zs):
            s_param = self.s_params[i][i]
            if not np.isnan(reference_z) and self.is_valid(s_param):
                self.impedances[i] = reference_z * (1 + s_param) / (1 - s_param)

        for i in range(self.count):
            if self.is_valid(self.incident[i][i]):
                for j in range(self.count):
                    if self.is_valid(self.s_params[j][i]):
                        phase = np.unwrap(np.angle(self.s_params[j][i]))
                        group_delay = -(
                            np.convolve(phase, [1, -1], mode="valid")
                            / np.convolve(self.frequencies, [1, -1], mode="valid")
                            / 2
                            / np.pi
                        )
                        group_delay = np.append(group_delay, group_delay[-1])
                        self.delays[j][i] = group_delay
        self.delays_valid = True

    def get_impedance(self, port: int) -> Union[np.ndarray, None]:
        """Return specified port impedance."""
        if port >= self.count:
            logger.error("Port no. %d doesn't exist", port)
            return None
        if self.is_valid(self.impedances[port]):
            logger.error("Impedance for port %d wasn't calculated", port)
            return None
        return self.impedances[port]

    def get_s_param(self, output_port: int, input_port: int) -> Optional[np.ndarray]:
        """Return specified S parameter."""
        if output_port >= self.count:
            logger.error("Port no. %d doesn't exist", output_port)
            return None
        if input_port >= self.count:
            logger.error("Port no. %d doesn't exist", output_port)
            return None
        s_param = self.s_params[output_port][input_port]
        if self.is_valid(s_param):
            return s_param
        logger.error("S%d%d wasn't calculated", output_port, input_port)
        return None

    def render_s_params(self) -> None:
        """Render all S parameter plots to files."""
        logger.info("Rendering S-parameter plots")
        plt.style.use(PLOT_STYLE)
        for i in range(self.count):
            if self.is_valid(self.s_params[i][i]):
                fig, axes = plt.subplots()
                for j in range(self.count):
                    s_param = self.s_params[j][i]
                    if self.is_valid(s_param):
                        axes.plot(
                            self.frequencies / 1e9,
                            20 * np.log10(np.abs(s_param)),
                            label="$S_{" + f"{j+1}{i+1}" + "}$",
                        )
                axes.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
                axes.set_xlabel("Frequency, f [GHz]")
                axes.set_ylabel("Magnitude, [dB]")
                axes.grid(True)
                bottom, top = axes.get_ylim()
                axes.set_ylim([min(bottom, -60), max(top, 5)])
                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, f"S_x{i+1}.png"),
                    bbox_inches="tight",
                    transparent=cfg.arguments.transparent,
                )

    def render_diff_pair_s_params(self) -> None:
        """Render differential pair S parameter plots to files."""
        logger.info("Rendering differential pair S-parameter plots")
        plt.style.use(PLOT_STYLE)
        for pair in cfg.diff_pairs:
            if (
                pair.correct
                and self.is_valid(self.s_params[pair.start_p][pair.start_p])
                and self.is_valid(self.s_params[pair.start_n][pair.start_n])
            ):
                fig, axes = plt.subplots()
                s_param = 0.5 * (
                    self.s_params[pair.start_p][pair.start_p]
                    - self.s_params[pair.start_n][pair.start_p]
                    - self.s_params[pair.start_p][pair.start_n]
                    + self.s_params[pair.start_n][pair.start_n]
                )
                if self.is_valid(s_param):
                    axes.plot(
                        self.frequencies / 1e9,
                        20 * np.log10(np.abs(s_param)),
                        label="$SDD_{11}$",
                    )
                s_param = 0.5 * (
                    self.s_params[pair.stop_p][pair.start_p]
                    - self.s_params[pair.stop_p][pair.start_n]
                    - self.s_params[pair.stop_n][pair.start_p]
                    + self.s_params[pair.stop_n][pair.start_n]
                )
                if self.is_valid(s_param):
                    axes.plot(
                        self.frequencies / 1e9,
                        20 * np.log10(np.abs(s_param)),
                        label="$SDD_{21}$",
                    )
                axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                axes.set_xlabel("Frequency, f [GHz]")
                axes.set_ylabel("Magnitude, [dB]")
                axes.grid(True)
                bottom, top = axes.get_ylim()
                axes.set_ylim([min(bottom, -60), max(top, 5)])
                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, "SDD_Diff"),
                    bbox_inches="tight",
                    transparent=cfg.arguments.transparent,
                )

    def render_diff_impedance(self) -> None:
        """Render differential pair impedance plots to files."""
        logger.info("Rendering differential pair impedance plots")
        plt.style.use(PLOT_STYLE)
        for pair in cfg.diff_pairs:
            if (
                pair.correct
                and self.is_valid(self.s_params[pair.start_p][pair.start_p])
                and self.is_valid(self.s_params[pair.start_n][pair.start_n])
            ):
                fig, axes = plt.subplots()
                s11 = self.s_params[pair.start_p][pair.start_p]
                s21 = self.s_params[pair.start_n][pair.start_p]
                s12 = self.s_params[pair.start_p][pair.start_n]
                s22 = self.s_params[pair.start_n][pair.start_n]
                gamma = ((2 * s11 - s21) * (1 - s22 - s12) + (1 - s11 - s21) * (1 + s22 - 2 * s12)) / (
                    (2 - s21) * (1 - s22 - s12) + (1 - s11 - s21) * (1 + s22)
                )
                if (
                    self.reference_zs[pair.start_p]
                    == self.reference_zs[pair.start_n]
                    == self.reference_zs[pair.stop_p]
                    == self.reference_zs[pair.stop_n]
                ):
                    z0 = self.reference_zs[pair.start_p]
                    impedance = z0 * (1 + gamma) / (1 - gamma)

                    fig, axs = plt.subplots(2)
                    axs[0].plot(self.frequencies / 1e9, np.abs(impedance))
                    axs[1].plot(
                        self.frequencies / 1e9,
                        np.angle(impedance, deg=True),
                        linestyle="dashed",
                        color="orange",
                    )

                    axs[0].set_ylabel("Magnitude, $|Z_{diff}| [\Omega]$")
                    axs[1].set_ylabel("Angle, $arg(Z_{diff}) [^\circ]$")
                    axs[1].set_xlabel("Frequency, f [GHz]")
                    axs[0].grid(True)
                    axs[1].grid(True)

                    bottom, top = axs[0].get_ylim()
                    axs[0].set_ylim([min(bottom, 0), max(top, 200)])
                    bottom, top = axs[1].get_ylim()
                    axs[1].set_ylim([min(bottom, -90), max(top, 90)])

                    fig.savefig(
                        os.path.join(os.getcwd(), RESULTS_DIR, "Z_diff.png"),
                        bbox_inches="tight",
                        transparent=cfg.arguments.transparent,
                    )
                else:
                    logger.error(
                        f"Reference impedances for ports in differential pair {pair.name} are not all equal. Cannot calculate impedance"  # noqa: E501
                    )

    def calculate_min_max_impedance(self, s11_margin: np.ndarray, z0: float) -> Tuple[float, float]:
        """Calculate aproximated min-max values for impedance (it assumes phase is 0)."""
        angles = [0, np.pi]
        reflection_coeffs = 10 ** (-s11_margin / 20) * (np.cos(angles) + 1j * np.sin(angles))
        impedances = z0 * (1 + reflection_coeffs) / (1 - reflection_coeffs)
        return (abs(impedances[0]), abs(impedances[1]))

    def render_impedance(self, include_margins: bool = False) -> None:
        """Render all ports impedance plots to files."""
        logger.info("Rendering impedance plots")
        plt.style.use(PLOT_STYLE)
        for port, impedance in enumerate(self.impedances):
            if self.is_valid(impedance):
                fig, axs = plt.subplots(2)
                axs[0].plot(self.frequencies / 1e9, np.abs(impedance))
                axs[1].plot(
                    self.frequencies / 1e9,
                    np.angle(impedance, deg=True),
                    linestyle="dashed",
                    color="orange",
                )

                axs[0].set_ylabel("Magnitude, $|Z_{" + str(port) + r"}| [\Omega]$")
                axs[1].set_ylabel("Angle, $arg(Z_{" + str(port) + r"}) [^\circ]$")
                axs[1].set_xlabel("Frequency, f [GHz]")
                axs[0].grid(True)
                axs[1].grid(True)

                if include_margins:
                    s11_margin = cfg.ports[port].dB_margin
                    z0 = cfg.ports[port].impedance
                    min_z, max_z = self.calculate_min_max_impedance(s11_margin, z0)

                    axs[0].axhline(np.real(min_z), color="red")
                    axs[0].axhline(np.real(max_z), color="red")

                bottom, top = axs[0].get_ylim()
                axs[0].set_ylim([min(bottom, 0), max(top, 100)])
                bottom, top = axs[1].get_ylim()
                axs[1].set_ylim([min(bottom, -90), max(top, 90)])

                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, f"Z_{port+1}.png"),
                    bbox_inches="tight",
                    transparent=cfg.arguments.transparent,
                )

    def render_smith(self) -> None:
        """Render port reflection smithcharts to files."""
        logger.info("Rendering smith charts")
        plt.style.use(PLOT_STYLE)
        net = skrf.Network(frequency=self.frequencies / 1e9, s=self.s_params.transpose(2, 0, 1))
        for port in range(self.count):
            if self.is_valid(self.s_params[port][port]):
                fig, axes = plt.subplots()
                s11_margin = cfg.ports[port].dB_margin
                vswr_margin = (10 ** (s11_margin / 20) + 1) / (10 ** (s11_margin / 20) - 1)
                net.plot_s_smith(
                    m=port,
                    n=port,
                    ax=axes,
                    draw_labels=False,
                    show_legend=True,
                    draw_vswr=[vswr_margin],
                )
                axes.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, f"S_{port+1}{port+1}_smith.png"),
                    bbox_inches="tight",
                    transparent=cfg.arguments.transparent,
                )

    def render_trace_delays(self) -> None:
        """Render all trace delay plots to files."""
        logger.info("Rendering trace delay plots")
        plt.style.use(PLOT_STYLE)
        for trace in cfg.traces:
            if trace.correct and self.is_valid(self.delays[trace.stop][trace.start]):
                fig, axes = plt.subplots()
                axes.plot(
                    self.frequencies / 1e9,
                    self.delays[trace.stop][trace.start] * 1e9,
                    label=f"{trace.name} delay",
                )
                axes.legend(loc="center left", bbox_to_anchor=(0.5, 1))
                axes.set_xlabel("Frequency, f [GHz]")
                axes.set_ylabel("Trace delay, [ns]")
                axes.grid(True)
                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, f"{trace.name}_delay.png"),
                    bbox_inches="tight",
                    transparent=cfg.arguments.transparent,
                )

        for pair in cfg.diff_pairs:
            if (
                pair.correct
                and self.is_valid(self.delays[pair.stop_p][pair.start_n])
                and self.is_valid(self.delays[pair.stop_n][pair.start_p])
            ):
                fig, axes = plt.subplots()
                axes.plot(
                    self.frequencies / 1e9,
                    self.delays[pair.stop_p][pair.start_n] * 1e9,
                    label="N trace delay",
                )
                axes.plot(
                    self.frequencies / 1e9,
                    self.delays[pair.stop_n][pair.start_p] * 1e9,
                    label="P trace delay",
                )
                axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                axes.set_xlabel("Frequency, f [GHz]")
                axes.set_ylabel("Trace delay, [ns]")
                axes.grid(True)
                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, "diff_delay.png"),
                    bbox_inches="tight",
                    transparent=cfg.arguments.transparent,
                )

    def save_to_file(self) -> None:
        """Save all parameters to files."""
        for i, _ in enumerate(self.s_params):
            if self.is_valid(self.s_params[i][i]):
                self.save_port_to_file(i, RESULTS_DIR)

    def save_port_to_file(self, port_number: int, path: str) -> None:
        """Save all parameters from single excitation."""
        frequencies = np.transpose([self.frequencies])
        s_params = np.transpose(self.s_params[:, port_number, :], (1, 0))

        header: str = "Frequency [Hz], "
        header += "".join([f"|S{i}-{port_number}| [-], " for i, _ in enumerate(self.s_params[port_number])])
        header += "".join([f"Arg(S{i}-{port_number}) [-], " for i, _ in enumerate(self.s_params[port_number])])

        if self.delays_valid:
            delays = np.transpose(self.delays[:, port_number, :], (1, 0))
            impedances = np.transpose([self.impedances[port_number]])
            header += "".join([f"Delay {port_number}>{i} [s], " for i, _ in enumerate(self.delays[port_number])])
            header += f"|Z{port_number}| [Ohm] , "
            header += f"Arg(Z{port_number}) [-]"
            file_path = f"Port_{port_number}_data.csv"
            add_data = [delays, np.abs(impedances), np.angle(impedances)]
        else:
            file_path = f"Port_Sx-{port_number}.csv"
            add_data = []

        output = np.hstack(
            [
                frequencies,
                np.abs(s_params),
                np.angle(s_params),
                *add_data,
            ]
        )
        logger.debug("Saving port no. %d parameters to file: %s", port_number, file_path)
        np.savetxt(os.path.join(path, file_path), output, fmt="%e", delimiter=", ", header=header, comments="")

    def load_data(self) -> None:
        """TODO"""
        pass

    @staticmethod
    def is_valid(array: np.ndarray) -> bool:
        """Check if array doesn't have any NaN's."""
        return not np.any(np.isnan(array))
