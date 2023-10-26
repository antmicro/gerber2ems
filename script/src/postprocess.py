"""Module contains functions usefull for postprocessing data."""
from typing import Union
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import skrf
from config import Config


from constants import RESULTS_DIR, PLOT_STYLE

logger = logging.getLogger(__name__)


class Postprocesor:
    """Class used to postprocess and display simulation data."""

    def __init__(self, frequencies: np.ndarray, port_count: int) -> None:
        """Initialize postprocessor."""
        self.frequencies = (
            frequencies  # Frequency list for whitch parameters are calculated
        )
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

    def add_port_data(
        self,
        port: int,
        excited_port: int,
        incident: np.ndarray,
        reflected: np.ndarray,
    ):
        """Add port data to postprocessor.

        Data consists of incident and reflected phasor data in relation to frequency
        """
        if self.is_valid(self.incident[port][excited_port]):
            logger.warning("This port data has already been supplied, overwriting")
        self.incident[port][excited_port] = incident
        self.reflected[port][excited_port] = reflected

    def add_impedances(self, impedances: np.ndarray):
        """Add port reference impedances."""
        self.reference_zs = impedances

    def process_data(self):
        """Calculate all needed parameters for further processing. Should be called after all ports are added."""
        logger.info(
            "Processing all data from simulation. Calculating S-parameters and impedance"
        )
        for i, _ in enumerate(self.incident):
            if self.is_valid(self.incident[i][i]):
                for j, _ in enumerate(self.incident):
                    if self.is_valid(self.reflected[j][i]):
                        self.s_params[j][i] = self.reflected[j][i] / self.incident[i][i]

        for i, reference_z in enumerate(self.reference_zs):
            s_param = self.s_params[i][i]
            if self.is_valid(reference_z) and self.is_valid(s_param):
                self.impedances[i] = reference_z * (1 - s_param) / (1 + s_param)

    def get_impedance(self, port: int) -> Union[np.ndarray, None]:
        """Return specified port impedance."""
        if port >= self.count:
            logger.error("Port no. %d doesn't exist", port)
            return None
        if self.is_valid(self.impedances[port]):
            logger.error("Impedance for port %d wasn't calculated", port)
            return None
        return self.impedances[port]

    def get_s_param(self, output_port, input_port):
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

    def render_s_params(self):
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
                axes.legend()
                axes.set_xlabel("Frequency, f [GHz]")
                axes.set_ylabel("Magnitude, [dB]")
                axes.grid(True)
                fig.savefig(os.path.join(os.getcwd(), RESULTS_DIR, f"S_x{i+1}.png"))

    def calculate_min_max_impedance(self, s11_margin, z0):
        """Calculate aproximated min-max values for impedance (it assumes phase is 0)."""
        angles = [0, np.pi]
        reflection_coeffs = 10 ** (-s11_margin / 20) * (
            np.cos(angles) + 1j * np.sin(angles)
        )
        impedances = z0 * (1 + reflection_coeffs) / (1 - reflection_coeffs)
        return (abs(impedances[0]), abs(impedances[1]))

    def render_impedance(self, include_margins=False):
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
                    s11_margin = Config.get().ports[port].dB_margin
                    z0 = Config.get().ports[port].impedance
                    min_z, max_z = self.calculate_min_max_impedance(s11_margin, z0)

                    axs[0].axhline(np.real(min_z), color="red")
                    axs[0].axhline(np.real(max_z), color="red")

                fig.savefig(
                    os.path.join(os.getcwd(), RESULTS_DIR, f"Z_{port+1}.png"),
                    bbox_inches="tight",
                )

    def render_smith(self):
        """Render port reflection smithcharts to files."""
        logger.info("Rendering smith charts")
        plt.style.use(PLOT_STYLE)
        net = skrf.Network(
            frequency=self.frequencies / 1e9, s=self.s_params.transpose(2, 0, 1)
        )
        for port in range(self.count):
            if self.is_valid(self.s_params[port][port]):
                fig, axes = plt.subplots()
                s11_margin = Config.get().ports[port].dB_margin
                vswr_margin = (10 ** (s11_margin / 20) + 1) / (
                    10 ** (s11_margin / 20) - 1
                )
                net.plot_s_smith(
                    m=port,
                    n=port,
                    ax=axes,
                    draw_labels=False,
                    show_legend=True,
                    draw_vswr=[vswr_margin],
                )
                fig.savefig(
                    os.path.join(
                        os.getcwd(), RESULTS_DIR, f"S_{port+1}{port+1}_smith.png"
                    ),
                    bbox_inches="tight",
                )

    def save_to_file(self) -> None:
        """Save S parameters to files."""
        logger.info("Saving S parameters")
        s_param_path = os.path.join(RESULTS_DIR, "S-parameters")
        os.mkdir(s_param_path)
        for i, _ in enumerate(self.s_params):
            if self.is_valid(self.s_params[i][i]):
                self.save_port_to_file(i, s_param_path)

    def save_port_to_file(self, port_number: int, path) -> None:
        """Save S parameters from single excitation."""
        header: str = "Frequency, ," + "".join(
            [
                f"|S_{i}{port_number}|, arg(S_{i}{port_number}), "
                for i, _ in enumerate(self.s_params[port_number])
            ]
        )
        header += f"|Z_{port_number}|, arg(Z_{port_number})"
        s_params = np.transpose(self.s_params[:, port_number, :], (1, 0))
        frequencies = np.transpose([self.frequencies], (1, 0))
        output_values = np.concatenate((frequencies, s_params), axis=1)

        magnitude = np.abs(output_values)
        angle = np.angle(output_values)

        output = np.empty(
            (magnitude.shape[0], (magnitude.shape[1] + angle.shape[1])),
            dtype=magnitude.dtype,
        )
        output[:, 0::2] = magnitude
        output[:, 1::2] = angle
        output = np.hstack([output, np.array(np.transpose(np.abs(self.impedances)))])
        output = np.hstack([output, np.array(np.transpose(np.angle(self.impedances)))])

        file_path = f"S_x{port_number}.csv"
        logger.debug(
            "Saving S_x%d parameters and impedances to file: %s", port_number, file_path
        )
        np.savetxt(
            os.path.join(path, file_path),
            output,
            fmt="%e",
            delimiter=",",
            header=header,
        )

    @staticmethod
    def is_valid(array: np.ndarray):
        """Check if array doesn't have any NaN's."""
        return not np.any(np.isnan(array))
