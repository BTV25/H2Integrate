import math

import numpy as np
import jax
import jax.numpy as jnp
from attrs import field, define
from scipy.interpolate import PchipInterpolator

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
        self._rated_capacity_kw = self.config.cluster_rating_MW * 1e3  # updated in compute()
        self._h2_out_scaled = np.zeros(0)
        self._size_mode = "normal"
        self._n_pem_clusters = self.config.n_clusters

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
        plant_life = self.options["plant_config"]["plant"]["plant_life"]
        arange = np.arange(n_ts)
        zeros = np.zeros(n_ts, int)

        # hydrogen_out is (n_ts,) → diagonal Jacobian
        self.declare_partials("hydrogen_out", "electricity_in", rows=arange, cols=arange)

        # total_hydrogen_produced is scalar → dense row
        self.declare_partials("total_hydrogen_produced", "electricity_in", rows=zeros, cols=arange)

        # annual_hydrogen_produced and capacity_factor are (plant_life,) → one dense row per year
        ann_rows = np.repeat(np.arange(plant_life), n_ts)
        ann_cols = np.tile(arange, plant_life)
        self.declare_partials("annual_hydrogen_produced", "electricity_in", rows=ann_rows, cols=ann_cols)
        self.declare_partials("capacity_factor", "electricity_in", rows=ann_rows, cols=ann_cols)

        # Sizing derivatives (n_clusters is the continuous relaxation of the electrolyzer's
        # discrete cluster count -- see compute_partials for the derivation).
        self.declare_partials("electrolyzer_size_mw", "n_clusters")
        self.declare_partials("hydrogen_out", "n_clusters", rows=arange, cols=zeros)
        self.declare_partials("total_hydrogen_produced", "n_clusters")
        self.declare_partials(
            "annual_hydrogen_produced", "n_clusters",
            rows=np.arange(plant_life), cols=np.zeros(plant_life, int),
        )
        self.declare_partials(
            "capacity_factor", "n_clusters",
            rows=np.arange(plant_life), cols=np.zeros(plant_life, int),
        )

        self._n_ts = n_ts
        self._plant_life = plant_life

    # Hardcoded per-stack power rating in PEM_H2_Clusters.external_power_supply, which
    # clips input power to n_total_stacks * _STACK_RATING_KW before it ever reaches the
    # electrochemical model. Must mirror that cap here so the analytic Jacobian goes to
    # zero at saturation like the true (clipped) forward model does.
    _STACK_RATING_KW = 1000.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # --- electricity_in: JAX diagonal Jacobian ---
        # d(hydrogen_out[t])/d(electricity_in[t]) = scale * grad(h2_per_stack)(P/n_total_stacks)
        # n_total_stacks cancels in the chain rule (see derivation in NOTES.md).
        # Zero out off-timesteps: when electricity_in[t] < cluster min power, the
        # electrolyzer is off and d(hydrogen_out)/d(electricity_in) = 0.  FD gives 0
        # at those timesteps (relative step × 0 = 0), so analytic must match.
        # NOTE: this Jacobian uses the *nearest* integer cluster count's stack count as a
        # representative dispatch configuration (self._n_total_stacks, set in compute()) and
        # does not propagate through the n_clusters PCHIP blend across neighboring cluster
        # counts, nor through the per-timestep multi-cluster dispatch split -- both are Phase 2
        # scope (dispatch-side floor/staircase), not this phase (sizing bin-crossing).
        elec = inputs["electricity_in"]
        if self._n_total_stacks <= 0:
            # No electrolyzer at this size: hydrogen_out is identically zero regardless of
            # electricity_in, so the Jacobian is zero.
            jac_diag = np.zeros(len(elec))
        else:
            n = float(self._n_total_stacks)
            power_per_stack = jnp.array(elec / n, dtype=jnp.float64)
            # Use forward-pass output to mask off-timesteps: if run_h2_PEM produced no
            # hydrogen at timestep t, FD also gives 0, so analytic must match.
            on_mask = (self._h2_out_scaled > 0).astype(float) if len(self._h2_out_scaled) == len(elec) else np.ones(len(elec))
            # Zero out saturated timesteps: external_power_supply() clips power_per_stack at
            # _STACK_RATING_KW, so beyond that point hydrogen_out no longer responds to
            # electricity_in and FD correctly gives 0 there too.
            sat_mask = (np.asarray(power_per_stack) < self._STACK_RATING_KW).astype(float)
            jac_diag = (
                np.array(jax.vmap(jax.grad(self._jax_h2_fn))(power_per_stack))
                * self._scale * on_mask * sat_mask
            )

        partials["hydrogen_out", "electricity_in"] = jac_diag
        partials["total_hydrogen_produced", "electricity_in"] = jac_diag

        rated_h2 = getattr(self, "_rated_hydrogen_production", 1.0)
        cf_denom = rated_h2 * self._n_ts if rated_h2 > 0 else 1.0
        partials["annual_hydrogen_produced", "electricity_in"] = np.tile(jac_diag, self._plant_life)
        partials["capacity_factor", "electricity_in"] = np.tile(jac_diag / cf_denom, self._plant_life)

        # --- n_clusters: sizing derivative ---
        # electrolyzer_size_mw = n_clusters * cluster_rating_MW is exact and continuous in
        # "normal" mode. hydrogen_out/total_hydrogen_produced/annual_hydrogen_produced/
        # capacity_factor are now each a monotone (PCHIP) spline over the discrete simulated
        # cluster counts bracketing n_continuous = electrolyzer_size_mw/cluster_rating_MW (see
        # compute()), so d(output)/d(n_clusters) = spline.derivative()(n_continuous) *
        # d(n_continuous)/d(n_clusters); the second factor is 1 in "normal" mode (0 otherwise).
        partials["electrolyzer_size_mw", "n_clusters"] = (
            self.config.cluster_rating_MW if self._size_mode == "normal" else 0.0
        )
        if self._size_mode == "normal":
            partials["hydrogen_out", "n_clusters"] = self._pchip_hydrogen_deriv(self._n_continuous)
            partials["total_hydrogen_produced", "n_clusters"] = float(
                self._pchip_total_deriv(self._n_continuous)
            )
            # annual_hydrogen_produced is a scalar broadcast to all plant_life years (see
            # compute()), so its derivative broadcasts the same way.
            partials["annual_hydrogen_produced", "n_clusters"] = np.full(
                self._plant_life, float(self._pchip_annual_deriv(self._n_continuous))
            )
            partials["capacity_factor", "n_clusters"] = self._pchip_cf_deriv(self._n_continuous)
        else:
            partials["hydrogen_out", "n_clusters"] = np.zeros(self._n_ts)
            partials["total_hydrogen_produced", "n_clusters"] = 0.0
            partials["annual_hydrogen_produced", "n_clusters"] = np.zeros(self._plant_life)
            partials["capacity_factor", "n_clusters"] = np.zeros(self._plant_life)

    def _run_pem_at_n_pem(
        self,
        n_pem,
        energy_to_electrolyzer_kw,
        plant_life,
        electrolyzer_capex_kw,
        pem_param_dict,
        grid_connection_scenario,
        hydrogen_production_capacity_required_kgphr,
    ):
        """Run the PEM simulation at an exact integer cluster count (or return exact zeros
        for n_pem<=0 -- below one cluster there is no electrolyzer to run, and run_h2_PEM
        divides by n_pem_clusters internally so it can't be called with n_pem<=0)."""
        n_ts = self._n_ts
        if n_pem <= 0:
            return {
                "hydrogen_out": np.zeros(n_ts),
                "total_hydrogen_produced": 0.0,
                "annual_hydrogen_produced": 0.0,  # scalar: broadcast to plant_life in compute()
                "capacity_factor": np.zeros(self.plant_life),
                "efficiency": 0.0,
                "replacement_schedule": np.zeros(self.plant_life),
                "time_until_replacement": 80000.0,
                "rated_hydrogen_production": 0.0,
                "n_total_stacks": 0,
            }

        capacity_mw = n_pem * self.config.cluster_rating_MW
        H2_Results, h2_ts, h2_tot, power_to_electrolyzer_kw = run_h2_PEM(
            electrical_generation_timeseries=energy_to_electrolyzer_kw,
            electrolyzer_size=capacity_mw,
            useful_life=plant_life,
            n_pem_clusters=n_pem,
            electrolyzer_direct_cost_kw=electrolyzer_capex_kw,
            user_defined_pem_param_dictionary=pem_param_dict,
            grid_connection_scenario=grid_connection_scenario,
            hydrogen_production_capacity_required_kgphr=hydrogen_production_capacity_required_kgphr,
            debug_mode=False,
            verbose=False,
        )
        hydrogen_out = np.asarray(H2_Results["Hydrogen Hourly Production [kg/hr]"])

        refurb_schedule = np.zeros(self.plant_life)
        if np.isnan(H2_Results["Time Until Replacement [hrs]"]):
            refurb_period = int(80000 / (24 * 365))
        else:
            refurb_period = int(round(float(H2_Results["Time Until Replacement [hrs]"]) / (24 * 365)))
        refurb_schedule[refurb_period : self.plant_life : refurb_period] = 1

        return {
            "hydrogen_out": hydrogen_out,
            "total_hydrogen_produced": float(hydrogen_out.sum()),
            # Scalar (same value broadcast across all years) -- see original code's
            # `* scale` on a bare float, and the "annual_hydrogen_produced[1:] ==
            # annual_hydrogen_produced[0]" regression test.
            "annual_hydrogen_produced": float(
                np.asarray(H2_Results["Life: Annual H2 production [kg/year]"])
            ),
            "capacity_factor": np.asarray(
                H2_Results["Performance Schedules"]["Capacity Factor [-]"].values
            ),
            "efficiency": H2_Results["Sim: Average Efficiency [%-HHV]"],
            "replacement_schedule": refurb_schedule,
            "time_until_replacement": H2_Results["Time Until Replacement [hrs]"],
            "rated_hydrogen_production": float(
                np.asarray(H2_Results["Rated BOL: H2 Production [kg/hr]"]).flat[0]
            ),
            "n_total_stacks": n_pem * int(round(self.config.cluster_rating_MW)),
        }

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

        self._size_mode = size_mode

        # n_continuous is the continuous analog of "how many clusters", regardless of which
        # sizing mode drove electrolyzer_size_mw. Interpolate a monotone (PCHIP) spline through
        # the true discrete PEM simulations at the integer cluster counts bracketing it, instead
        # of rounding to one integer -- this removes both the level-jump and the gradient-reset
        # at cluster-count boundaries (see plan doc / outputs/2026-08-18_electrolyzer_sizing_*
        # for the comparison against a naive blend, which failed two different ways first).
        n_continuous = max(electrolyzer_size_mw / self.config.cluster_rating_MW, 0.0)
        n_lo = int(np.floor(n_continuous))
        n_hi = n_lo + 1
        neighbor_ns = sorted({n for n in (n_lo - 1, n_lo, n_hi, n_hi + 1) if n >= 0})
        self._n_continuous = n_continuous
        self._n_pem_clusters = int(round(n_continuous))

        pem_param_dict = {
            "eol_eff_percent_loss": self.config.eol_eff_percent_loss,
            "uptime_hours_until_eol": self.config.uptime_hours_until_eol,
            "include_degradation_penalty": self.config.include_degradation_penalty,
            "turndown_ratio": self.config.turndown_ratio,
        }
        energy_to_electrolyzer_kw = inputs["electricity_in"]

        sims = {
            n_pem: self._run_pem_at_n_pem(
                n_pem,
                energy_to_electrolyzer_kw,
                plant_life,
                electrolyzer_capex_kw,
                pem_param_dict,
                grid_connection_scenario,
                hydrogen_production_capacity_required_kgphr,
            )
            for n_pem in neighbor_ns
        }

        nodes = np.array(neighbor_ns, dtype=float)
        hydrogen_stack = np.stack([sims[n]["hydrogen_out"] for n in neighbor_ns], axis=0)
        total_stack = np.array([sims[n]["total_hydrogen_produced"] for n in neighbor_ns])
        # annual_hydrogen_produced is a scalar per node (same value broadcast across all
        # years), not a (plant_life,) array -- see _run_pem_at_n_pem.
        annual_scalar_stack = np.array(
            [sims[n]["annual_hydrogen_produced"] for n in neighbor_ns]
        )
        cf_stack = np.stack([sims[n]["capacity_factor"] for n in neighbor_ns], axis=0)

        pchip_hydrogen = PchipInterpolator(nodes, hydrogen_stack, axis=0)
        pchip_total = PchipInterpolator(nodes, total_stack)
        pchip_annual = PchipInterpolator(nodes, annual_scalar_stack)
        pchip_cf = PchipInterpolator(nodes, cf_stack, axis=0)

        outputs["hydrogen_out"] = pchip_hydrogen(n_continuous)
        outputs["total_hydrogen_produced"] = float(pchip_total(n_continuous))
        outputs["annual_hydrogen_produced"] = float(pchip_annual(n_continuous))
        outputs["capacity_factor"] = pchip_cf(n_continuous)
        # Use the continuous n_clusters * cluster_rating_MW so the gradient chain is unbroken.
        outputs["electrolyzer_size_mw"] = electrolyzer_size_mw

        self._pchip_hydrogen_deriv = pchip_hydrogen.derivative()
        self._pchip_total_deriv = pchip_total.derivative()
        self._pchip_annual_deriv = pchip_annual.derivative()
        self._pchip_cf_deriv = pchip_cf.derivative()
        self._h2_out_scaled = outputs["hydrogen_out"]

        # Informational-only outputs (no declared n_clusters partials): report the nearest
        # integer configuration's simulation rather than interpolating them.
        nearest = sims[int(np.clip(round(n_continuous), neighbor_ns[0], neighbor_ns[-1]))]
        outputs["efficiency"] = nearest["efficiency"]
        outputs["replacement_schedule"] = nearest["replacement_schedule"]
        outputs["time_until_replacement"] = nearest["time_until_replacement"]
        outputs["rated_hydrogen_production"] = nearest["rated_hydrogen_production"]
        self._rated_hydrogen_production = nearest["rated_hydrogen_production"]

        # Context for the electricity_in Jacobian (see compute_partials): the nearest integer
        # configuration's stack count, used as a representative dispatch context. self._scale
        # is neutral now that hydrogen_out is the spline value directly, not a raw*scale product.
        self._n_total_stacks = nearest["n_total_stacks"]
        self._scale = 1.0
        self._rated_capacity_kw = self._n_pem_clusters * self.config.cluster_rating_MW * 1e3
        elec = inputs["electricity_in"]
        total_elec = float(np.sum(elec))
        total_h2_nearest = float(np.sum(nearest["hydrogen_out"]))
        self._h2_per_kw = total_h2_nearest / total_elec if total_elec > 0 else self._h2_per_kw
        self._turndown_power_kw = (
            self.config.turndown_ratio * self._n_pem_clusters * self.config.cluster_rating_MW * 1e3
        )
