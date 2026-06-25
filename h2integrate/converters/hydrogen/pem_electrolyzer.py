import math

import numpy as np
import jax
import jax.numpy as jnp
from attrs import field, define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.validators import gt_zero, contains
from h2integrate.core.model_baseclasses import ResizeablePerformanceModelBaseConfig
from h2integrate.converters.hydrogen.utilities import size_electrolyzer_for_hydrogen_demand
from h2integrate.converters.hydrogen.pem_model.run_h2_PEM import run_h2_PEM
from h2integrate.converters.hydrogen.pem_model.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters as _PEMClusters,
)
from h2integrate.converters.hydrogen.electrolyzer_baseclass import ElectrolyzerPerformanceBaseClass

jax.config.update("jax_enable_x64", True)


def _make_jax_h2_per_stack_fn(curve_coeff):
    """JAX-differentiable H2 production for one 1 MW stack.

    Returns a function: power_per_stack_kw (scalar) -> h2_kg_hr (scalar).

    Implements calc_current -> faradaic_efficiency -> h2_production_rate from
    PEM_H2_Clusters. Uses V_deg=0 (BOL), which introduces <1% Jacobian error
    for simulation windows short relative to stack lifetime (80 000 h).
    Intended for always-on operation (no discrete on/off switching).
    """
    p = jnp.array(curve_coeff[:5], dtype=jnp.float64)

    N_cells = 135
    F = 96485.34        # C/mol
    moles_per_g = 0.49606  # mol/g H2
    dt = 3600.0         # s/hr
    cell_area = 1949.0  # cm^2

    def h2_per_stack_fn(power_per_stack_kw):
        # calc_current: cubic polynomial fit (p6 unused in original)
        pwr = power_per_stack_kw
        I = (p[0]*pwr**3 + p[1]*pwr**2 + p[2]*pwr
             + p[3]*jnp.sqrt(jnp.maximum(pwr, 0.0)) + p[4])
        I = jnp.maximum(I, 0.0)
        # faradaic_efficiency
        i_mA_cm2 = (I * 1000.0) / cell_area
        n_F = (i_mA_cm2**2 / (250.0 + i_mA_cm2**2)) * 0.9909
        # h2_production_rate (Faraday's law)
        h2_mol_s = n_F * (N_cells * I / (2.0 * F))
        return h2_mol_s / moles_per_g * (dt / 1000.0)  # kg/hr

    return h2_per_stack_fn


@define(kw_only=True)
class ECOElectrolyzerPerformanceModelConfig(ResizeablePerformanceModelBaseConfig):
    """
    Configuration class for the ECOElectrolyzerPerformanceModel.

    Args:
        size_mode (str): The mode in which the component is sized. Options:
            - "normal": The component size is taken from the tech_config.
            - "resize_by_max_feedstock": Resize based on maximum feedstock availability.
            - "resize_by_max_commodity": Resize based on maximum commodity demand.
        flow_used_for_sizing (str | None): The feedstock/commodity flow used for sizing.
            Required when size_mode is not "normal".
        max_feedstock_ratio (float): Ratio for sizing in "resize_by_max_feedstock" mode.
            Defaults to 1.0.
        max_commodity_ratio (float): Ratio for sizing in "resize_by_max_commodity" mode.
            Defaults to 1.0.
        n_clusters (int): number of electrolyzer clusters within the system.
        location (str): The location of the electrolyzer; options include "onshore" or "offshore".
        cluster_rating_MW (float): The rating of the clusters that the electrolyzer is grouped
            into, in MW.
        eol_eff_percent_loss (float): End-of-life (EOL) defined as a percent change in efficiency
            from beginning-of-life (BOL).
        uptime_hours_until_eol (int): Number of "on" hours until the electrolyzer reaches EOL.
        include_degradation_penalty (bool): Flag to include degradation of the electrolyzer due to
            operational hours, ramping, and on/off power cycles.
        turndown_ratio (float): The ratio at which the electrolyzer will shut down.
        electrolyzer_capex (int): $/kW overnight installed capital costs for a 1 MW system in
            2022 USD/kW (DOE hydrogen program record 24005 Clean Hydrogen Production Cost Scenarios
            with PEM Electrolyzer Technology 05/20/24) #TODO: convert to refs
            (https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/24005-clean-hydrogen-production-cost-pem-electrolyzer.pdf?sfvrsn=8cb10889_1)
    """

    n_clusters: int = field(validator=gt_zero)
    location: str = field(validator=contains(["onshore", "offshore"]))
    cluster_rating_MW: float = field(validator=gt_zero)
    eol_eff_percent_loss: float = field(validator=gt_zero)
    uptime_hours_until_eol: int = field(validator=gt_zero)
    include_degradation_penalty: bool = field()
    turndown_ratio: float = field(validator=gt_zero)
    electrolyzer_capex: int = field()


class ECOElectrolyzerPerformanceModel(ElectrolyzerPerformanceBaseClass):
    """
    An OpenMDAO component that wraps the PEM electrolyzer model.
    Takes electricity input and outputs hydrogen and oxygen generation rates.
    """

    def setup(self):
        self.config = ECOElectrolyzerPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()
        self.add_output(
            "efficiency",
            val=0.0,
            units="unitless",
            desc="Average efficiency of the electrolyzer",
        )

        self.add_output(
            "time_until_replacement", val=80000.0, units="h", desc="Time until replacement"
        )

        self.add_input(
            "n_clusters",
            val=self.config.n_clusters,
            units="unitless",
            desc="number of electrolyzer clusters in the system",
        )

        self.add_output(
            "electrolyzer_size_mw",
            val=0.0,
            units="MW",
            desc="Size of the electrolyzer in MW",
        )
        self.add_input("cluster_size", val=-1.0, units="MW")
        self.add_input("max_hydrogen_capacity", val=1000.0, units="kg/h")
        # TODO: add feedstock inputs and consumption outputs

        # Initialise cached quantities used by compute_partials
        self._h2_per_kw = 1.0 / (39.4 * 1000.0 / 0.67)  # fallback: ~67% HHV efficiency
        self._turndown_power_kw = self.config.turndown_ratio * self.config.cluster_rating_MW * 1e3
        self._h2_out_scaled = np.zeros(0)
        self._ann_h2_scaled = np.zeros(0)
        self._cf_scaled = np.zeros(0)

        # JAX electrochemical function (curve_coeff depends only on turndown_ratio)
        _pem_tmp = _PEMClusters(
            1,
            1,
            eol_eff_percent_loss=self.config.eol_eff_percent_loss,
            uptime_hours_until_eol=self.config.uptime_hours_until_eol,
            include_degradation_penalty=False,
            turndown_ratio=self.config.turndown_ratio,
        )
        self._jax_h2_fn = _make_jax_h2_per_stack_fn(_pem_tmp.curve_coeff)
        self._n_total_stacks = 1
        self._scale = 1.0

    def setup_partials(self):
        n_ts = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        arange = np.arange(n_ts)
        zeros = np.zeros(n_ts, int)

        # hydrogen_out is (n_ts,) → diagonal Jacobian
        self.declare_partials("hydrogen_out", "electricity_in", rows=arange, cols=arange)

        # Scalar aggregate outputs → dense row (1 × n_ts)
        for out in ("total_hydrogen_produced", "annual_hydrogen_produced", "capacity_factor"):
            self.declare_partials(out, "electricity_in", rows=zeros, cols=arange)

        self._n_ts = n_ts

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # --- electricity_in: JAX diagonal Jacobian ---
        # d(hydrogen_out[t])/d(electricity_in[t]) = scale * grad(h2_per_stack)(P/n_total_stacks)
        # n_total_stacks cancels in the chain rule (see derivation in NOTES.md).
        # Zero out off-timesteps: when electricity_in[t] < cluster min power, the
        # electrolyzer is off and d(hydrogen_out)/d(electricity_in) = 0.  FD gives 0
        # at those timesteps (relative step × 0 = 0), so analytic must match.
        n = float(self._n_total_stacks)
        elec = inputs["electricity_in"]
        cluster_min_kw = self.config.turndown_ratio * self.config.cluster_rating_MW * 1e3
        power_per_stack = jnp.array(elec / n, dtype=jnp.float64)
        on_mask = (np.array(power_per_stack) >= cluster_min_kw).astype(float)
        jac_diag = np.array(jax.vmap(jax.grad(self._jax_h2_fn))(power_per_stack)) * self._scale * on_mask

        partials["hydrogen_out", "electricity_in"] = jac_diag

        n_ts = self._n_ts
        rated_h2 = getattr(self, "_rated_hydrogen_production", 1.0)

        partials["total_hydrogen_produced", "electricity_in"] = jac_diag
        partials["annual_hydrogen_produced", "electricity_in"] = jac_diag
        cf_denom = rated_h2 * n_ts if rated_h2 > 0 else 1.0
        partials["capacity_factor", "electricity_in"] = jac_diag / cf_denom

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        plant_life = self.options["plant_config"]["plant"]["plant_life"]
        electrolyzer_size_mw = inputs["n_clusters"][0] * self.config.cluster_rating_MW
        electrolyzer_capex_kw = self.config.electrolyzer_capex

        hydrogen_production_capacity_required_kgphr = []
        grid_connection_scenario = "off-grid"
        energy_to_electrolyzer_kw = inputs["electricity_in"]

        # Resize if necessary based on sizing mode
        size_mode = discrete_inputs["size_mode"]
        # Make changes to computation based on sizing_mode:
        if size_mode != "normal":
            size_flow = discrete_inputs["flow_used_for_sizing"]
        if size_mode == "resize_by_max_feedstock":
            # In this sizing mode, electrolyzer size comes from feedstock
            feed_ratio = inputs["max_feedstock_ratio"][0]
            # Make sure COBLYA doesn't cause any shenanigans trying to set feed_ratio <= 0
            if feed_ratio <= 1e-6:
                feed_ratio = 1.0e-6
            if size_flow == "electricity":
                electrolyzer_size_mw = np.max(inputs["electricity_in"]) / 1000.0 * feed_ratio
            else:
                raise ValueError(f"Cannot resize for '{size_flow}' feedstock")
        elif size_mode == "resize_by_max_commodity":
            # In this sizing mode, electrolyzer size comes from a connected tech's capacity
            # to take in one of the electrolyzer's commodities
            comm_ratio = inputs["max_commodity_ratio"]
            # Make sure COBLYA doesn't cause any shenanigans trying to set comm_ratio <= 0
            if comm_ratio <= 1e-6:
                comm_ratio = 1e-6
            if size_flow == "hydrogen":
                h2_kgphr = inputs["max_hydrogen_capacity"]
                electrolyzer_size_mw = size_electrolyzer_for_hydrogen_demand(h2_kgphr * comm_ratio)
            else:
                raise ValueError(f"Cannot resize for '{size_flow}' commodity")
        elif size_mode != "normal":
            raise NotImplementedError("Sizing mode '%s' not implemented".format())

        n_pem_clusters = round(electrolyzer_size_mw / self.config.cluster_rating_MW)

        electrolyzer_actual_capacity_MW = n_pem_clusters * self.config.cluster_rating_MW
        self._n_total_stacks = n_pem_clusters * int(round(self.config.cluster_rating_MW))
        pem_param_dict = {
            "eol_eff_percent_loss": self.config.eol_eff_percent_loss,
            "uptime_hours_until_eol": self.config.uptime_hours_until_eol,
            "include_degradation_penalty": self.config.include_degradation_penalty,
            "turndown_ratio": self.config.turndown_ratio,
        }

        energy_to_electrolyzer_kw = inputs["electricity_in"]
        H2_Results, h2_ts, h2_tot, power_to_electrolyzer_kw = run_h2_PEM(
            electrical_generation_timeseries=energy_to_electrolyzer_kw,
            electrolyzer_size=electrolyzer_actual_capacity_MW,  # rounded, so scale factor is the only FD signal
            useful_life=plant_life,
            n_pem_clusters=n_pem_clusters,
            electrolyzer_direct_cost_kw=electrolyzer_capex_kw,
            user_defined_pem_param_dictionary=pem_param_dict,
            grid_connection_scenario=grid_connection_scenario,  # if not offgrid, assumes steady h2 demand in kgphr for full year  # noqa: E501
            hydrogen_production_capacity_required_kgphr=hydrogen_production_capacity_required_kgphr,
            debug_mode=False,
            verbose=False,
        )

        # Assuming `h2_results` includes hydrogen and oxygen rates per timestep
        h2_out_raw = H2_Results["Hydrogen Hourly Production [kg/hr]"]
        # Scale by n_clusters/n_pem_clusters so hydrogen outputs vary continuously with
        # n_clusters. At integer n_clusters the scale is exactly 1.0; FD steps now produce
        # a non-zero response that matches the analytic Jacobian.
        scale = electrolyzer_size_mw / electrolyzer_actual_capacity_MW
        self._scale = scale
        outputs["hydrogen_out"] = h2_out_raw * scale
        outputs["total_hydrogen_produced"] = outputs["hydrogen_out"].sum()

        # Cache _h2_per_kw from unscaled PEM result (for electricity_in partials only)
        elec = inputs["electricity_in"]
        total_elec = float(np.sum(elec))
        total_h2_raw = float(np.sum(h2_out_raw))
        self._h2_per_kw = total_h2_raw / total_elec if total_elec > 0 else self._h2_per_kw
        self._turndown_power_kw = (
            self.config.turndown_ratio * electrolyzer_actual_capacity_MW * 1e3
        )
        self._h2_out_scaled = outputs["hydrogen_out"]
        outputs["efficiency"] = H2_Results["Sim: Average Efficiency [%-HHV]"]
        refurb_schedule = np.zeros(self.plant_life)
        if np.isnan(H2_Results["Time Until Replacement [hrs]"]):
            refurb_period = int(80000 / (24 * 365))
        else:
            refurb_period = int(round(float(H2_Results["Time Until Replacement [hrs]"]) / (24 * 365)))
        refurb_schedule[refurb_period : self.plant_life : refurb_period] = 1

        # The replacement_schedule is the fraction of the total capacity that is replaced per year
        # The replacement_schedule may be used in the finance model if the replacement_cost_percent
        # is specified in the tech_config under
        # ['model_inputs']['finance_parameters']['capital_items']['replacement_cost_percent']
        outputs["replacement_schedule"] = refurb_schedule
        # NOTE: could replace above with line with below:
        # outputs["replacement_schedule"] = (H2_Results["Performance Schedules"]
        # ['Refurbishment Schedule [MW replaced/year]'].values
        # /electrolyzer_actual_capacity_MW
        # )

        # TODO: remove time_until_replacement as output after finance model(s) have been updated to not use it
        outputs["time_until_replacement"] = H2_Results["Time Until Replacement [hrs]"]

        outputs["rated_hydrogen_production"] = H2_Results["Rated BOL: H2 Production [kg/hr]"]
        self._rated_hydrogen_production = float(outputs["rated_hydrogen_production"].flat[0])
        # Use the continuous n_clusters * cluster_rating_MW so the gradient chain is unbroken.
        # electrolyzer_actual_capacity_MW (rounded) is used only for the PEM simulation above.
        outputs["electrolyzer_size_mw"] = electrolyzer_size_mw
        outputs["capacity_factor"] = H2_Results["Performance Schedules"][
            "Capacity Factor [-]"
        ].values * scale
        outputs["annual_hydrogen_produced"] = H2_Results["Life: Annual H2 production [kg/year]"] * scale
        self._ann_h2_scaled = outputs["annual_hydrogen_produced"]
        self._cf_scaled = outputs["capacity_factor"]

        # TODO: replace above line w below
        # outputs["annual_hydrogen_produced"] = H2_Results["Performance Schedules"][
        #     "Annual H2 Production [kg/year]"
        # ].values
