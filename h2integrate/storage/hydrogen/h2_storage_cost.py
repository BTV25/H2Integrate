import numpy as np
import jax
import jax.numpy as jnp
from attrs import field, define
from openmdao.utils import units

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import contains, gte_zero, range_val
from h2integrate.core.model_baseclasses import CostModelBaseClass
from h2integrate.storage.hydrogen.h2_transport.h2_compression import Compressor

jax.config.update("jax_enable_x64", True)


def _make_jax_lrc_cost(a, b, c_coef, stages, n_compressors, labor_rate,
                       insurance, property_taxes, licensing_permits,
                       comp_om, facility_om, cepci_cavern, p_outlet_bar):
    """JAX-differentiable LRC cost function. stages and n_compressors are pre-computed integers.

    Mirrors exactly the computation in LinedRockCavernStorageCostModel.compute() and
    Compressor.compressor_costs() / compressor_power() from h2_compression.py.
    """
    def compute(x):
        h2_kg, flow_kg_d = x[0], x[1]

        # ── Storage capex (from Papadias 2021) ─────────────────────────────────
        log_x = jnp.log(h2_kg / 1000.0)
        capex_per_kg = jnp.exp(a * log_x**2 - b * log_x + c_coef) * cepci_cavern
        storage_capex = capex_per_kg * h2_kg

        # ── Compressor: power (mirrors Compressor.compressor_power) ────────────
        Z_p = jnp.array([1.0, 10.0, 50.0, 100.0, 300.0, 500.0, 1000.0])
        Z_z = jnp.array([1.0006, 1.0059, 1.0297, 1.0601, 1.1879, 1.3197, 1.6454])
        p_in = 20.0
        Z = (jnp.interp(p_in, Z_p, Z_z) + jnp.interp(p_outlet_bar, Z_p, Z_z)) / 2.0

        flow_per_comp_kg_mols_s = flow_kg_d / n_compressors / 24.0 / 3600.0 / 2.0158
        p_ratio = p_outlet_bar / p_in
        cpcv = 1.41
        theoretical_power = (
            Z * flow_per_comp_kg_mols_s * 8.314 * (25.0 + 273.15) * stages
            * (cpcv / (cpcv - 1.0))
            * (p_ratio ** ((cpcv - 1.0) / (stages * cpcv)) - 1.0)
        )
        actual_power = theoretical_power / 0.88
        log_ap = jnp.log(actual_power)
        motor_eff = (0.00008 * log_ap**4 - 0.0015 * log_ap**3
                     + 0.0061 * log_ap**2 + 0.0311 * log_ap + 0.7617)
        motor_rating = 1.1 * actual_power / motor_eff

        # ── Compressor: costs in 2016 USD (mirrors Compressor.compressor_costs) ─
        cepci_2007_to_2016 = 1.29 / 1.1
        cost_per_unit_2016 = 1962.2 * motor_rating**0.8225 * 0.55 * cepci_2007_to_2016
        cost_per_unit_2016 = jnp.where(
            stages > 2,
            cost_per_unit_2016 * (1.0 + 0.2 * (stages - 2)),
            cost_per_unit_2016,
        )
        n_total = n_compressors + 1
        direct_capex_2016 = cost_per_unit_2016 * n_total * 2.0
        land = 10000.0 * 12.35
        other_pcts = 0.05 + 0.1 + 0.1 + 0.0 + 0.03 + 0.12   # = 0.40
        other_capital_2016 = other_pcts * direct_capex_2016 + land
        comp_capex_2016 = direct_capex_2016 + other_capital_2016

        # h2_storage_cost.py converts from $2016 to $2018
        cepci_2016_to_2018 = 1.36 / 1.29
        comp_capex = comp_capex_2016 * cepci_2016_to_2018

        # ── CapEx: storage cavern only (comp_capex NOT included — mirrors compute()) ─
        CapEx = storage_capex

        # ── OpEx (mirrors LinedRockCavernStorageCostModel.compute) ──────────────
        annual_hours = 8760.0 * (flow_kg_d / 100000.0) ** 0.25
        overhead = 0.5
        labor = annual_hours * labor_rate * (1.0 + overhead)
        insurance_c = insurance * storage_capex
        prop_tax = property_taxes * storage_capex
        lic_permit = licensing_permits * storage_capex
        comp_op = comp_om * comp_capex
        facility_op = facility_om * (storage_capex - comp_capex)
        total_om = labor + insurance_c + prop_tax + lic_permit + comp_op + facility_op

        return jnp.stack([CapEx, total_om])

    return compute


@define(kw_only=True)
class HydrogenStorageBaseCostModelConfig(BaseConfig):
    """Base config class for HydrogenStorageBaseCostModel.

    Fields include `max_capacity`, `max_charge_rate`, `sizing_mode`, `commodity_name`,
    `commodity_units`, `cost_year`, `labor_rate`, `insurance`, `property_taxes`,
    `licensing_permits`, `compressor_om`, and `facility_om`.
    """

    max_capacity: float | None = field(default=None)
    max_charge_rate: float | None = field(default=None)
    sizing_mode: str = field(
        default="set", converter=(str.strip, str.lower), validator=contains(["auto", "set"])
    )

    commodity_name: str = field(default="hydrogen")
    commodity_units: str = field(default="kg/h", validator=contains(["kg/h", "g/h", "t/h"]))

    cost_year: int = field(default=2018, converter=int, validator=contains([2018]))
    labor_rate: float = field(default=37.39817, validator=gte_zero)
    insurance: float = field(default=0.01, validator=range_val(0, 1))
    property_taxes: float = field(default=0.01, validator=range_val(0, 1))
    licensing_permits: float = field(default=0.001, validator=range_val(0, 1))
    compressor_om: float = field(default=0.04, validator=range_val(0, 1))
    facility_om: float = field(default=0.01, validator=range_val(0, 1))

    def __attrs_post_init__(self):
        undefined_capacities = self.max_capacity is None or self.max_charge_rate is None
        if undefined_capacities and self.sizing_mode == "set":
            msg = (
                "Missing storage attribute(s): max_capacity and/or max_charge_rate, "
                "for the cost_parameters. These attributes are required if `sizing_mode` "
                "is 'set'. If storage will be auto-sized by the performance model, set the "
                "`sizing_mode` cost parameter to 'auto'."
            )
            raise ValueError(msg)
        if not undefined_capacities and self.sizing_mode == "auto":
            msg = (
                "Extra storage attribute(s) found: max_capacity and/or max_charge_rate, "
                "for the cost_parameters. These attributes should not be defined if `sizing_mode` "
                "is 'auto'. If storage will be auto-sized by the performance model, set the "
                "`sizing_mode` cost parameter to 'auto' and do not include max_capacity or "
                "max_charge_rate and a cost parameter. Set `sizing_mode` to 'set' if the storage "
                "capacity is fixed."
            )
            raise ValueError(msg)

        if undefined_capacities and self.sizing_mode == "auto":
            # set to zero for initialization in setup().
            self.max_capacity = 0.0
            self.max_charge_rate = 0.0

    def make_model_dict(self):
        params = self.as_dict()
        h2i_params = [
            "max_capacity",
            "max_charge_rate",
            "commodity_name",
            "commodity_units",
            "cost_year",
        ]
        lrc_dict = {k: v for k, v in params.items() if k not in h2i_params}
        return lrc_dict


class HydrogenStorageBaseCostModel(CostModelBaseClass):
    def initialize(self):
        super().initialize()

    def setup(self):
        self.config = HydrogenStorageBaseCostModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )

        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        super().setup()

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=f"{self.config.commodity_units}",
            desc="Hydrogen storage charge rate",
        )

        self.add_input(
            "max_capacity",
            val=self.config.max_capacity,
            units=f"{self.config.commodity_units}*h",
            desc="Hydrogen storage capacity",
        )

        self.add_input(
            "hydrogen_in",
            val=0.0,
            shape=n_timesteps,
            units=f"{self.config.commodity_units}",
            desc="Hydrogen input timeseries for average flow rate calculation",
        )

    def make_storage_input_dict(self, inputs):
        storage_input = {}

        storage_input = self.config.make_model_dict()

        # convert capacity to kg
        max_capacity_kg = units.convert_units(
            inputs["max_capacity"], f"({self.config.commodity_units})*h", "kg"
        )

        storage_input["h2_storage_kg"] = max_capacity_kg[0]

        # system_flow_rate must be in kg/day.
        # Per HDSAM (Papadias 2021), system_flow_rate is the average flow rate,
        # not the maximum fill rate.
        avg_hydrogen_in = np.mean(inputs["hydrogen_in"])
        system_flow_rate = units.convert_units(
            avg_hydrogen_in, f"{self.config.commodity_units}", "kg/d"
        )
        storage_input["system_flow_rate"] = system_flow_rate  # kg/day

        return storage_input

    def compute(self, inputs, outputs):
        # storage_input = self.make_storage_input_dict(inputs)

        raise NotImplementedError("This method should be implemented in a subclass.")


class LinedRockCavernStorageCostModel(HydrogenStorageBaseCostModel):
    """Capital and operational cost model for lined rock cavern hydrogen storage.

    Costs are in 2018 USD. Operational dynamics are not yet included.

    References:
        [1] Papadias 2021: https://www.sciencedirect.com/science/article/pii/S0360319921030834?via%3Dihub
        [2] Papadias 2021: Bulk Hydrogen as Function of Capacity.docx documentation at
            hydrogen_storage.md in the docs
        [3] HDSAM V4.0 Gaseous H2 Geologic Storage sheet
    """

    _LRC_A = 0.095803
    _LRC_B = 1.5868
    _LRC_C = 10.332
    _P_OUTLET = 200.0   # bar, lined rock cavern max outlet pressure

    def setup_partials(self):
        from math import ceil, log10
        n_ts = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        # Pre-compute fixed Compressor constants (depend only on pressures and nominal flow)
        stages = ceil((log10(self._P_OUTLET) - log10(20.0)) / log10(2.1))
        nominal_flow_kg_d = units.convert_units(
            self.config.max_charge_rate, self.config.commodity_units, "kg/d"
        )
        comp = Compressor(self._P_OUTLET, nominal_flow_kg_d, n_compressors=2)
        comp.compressor_power()
        motor_rating, _ = comp.compressor_system_power()
        n_compressors = 3 if motor_rating > 1600 else 2

        cfg = self.config
        _fn = _make_jax_lrc_cost(
            a=self._LRC_A, b=self._LRC_B, c_coef=self._LRC_C,
            stages=stages, n_compressors=n_compressors,
            labor_rate=cfg.labor_rate, insurance=cfg.insurance,
            property_taxes=cfg.property_taxes, licensing_permits=cfg.licensing_permits,
            comp_om=cfg.compressor_om, facility_om=cfg.facility_om,
            cepci_cavern=1.29 / 1.30,
            p_outlet_bar=self._P_OUTLET,
        )
        self._jax_jac = jax.jit(jax.jacobian(_fn))
        self._n_ts = n_ts

        # CapEx = storage cavern cost (depends only on max_capacity, not hydrogen_in)
        self.declare_partials("CapEx", "max_capacity")
        # OpEx depends on both (insurance/taxes on storage_capex → max_capacity;
        # comp_op via comp_capex → hydrogen_in through system_flow_rate)
        self.declare_partials("OpEx", "max_capacity")
        self.declare_partials("OpEx", "hydrogen_in",
                              rows=np.zeros(n_ts, int), cols=np.arange(n_ts))

    def compute_partials(self, inputs, partials):
        # h2_storage_kg: (kg/h)*h → kg, conversion factor = 1
        h2_kg = float(inputs["max_capacity"][0])
        # system_flow_rate: mean(kg/h) → kg/d, multiply by 24
        flow_kg_d = float(np.mean(inputs["hydrogen_in"])) * 24.0
        jac = self._jax_jac(jnp.array([h2_kg, flow_kg_d]))  # shape (2, 2)

        # d(CapEx)/d(max_capacity): CapEx = storage_capex only, d(h2_kg)/d(max_capacity) = 1
        partials["CapEx", "max_capacity"] = float(jac[0, 0])
        # d(OpEx)/d(max_capacity): via insurance/taxes on storage_capex
        partials["OpEx", "max_capacity"] = float(jac[1, 0])
        # d(OpEx)/d(hydrogen_in[t]): via comp_capex → system_flow_rate → mean(h2_in)
        # chain rule: d(flow_kg_d)/d(hydrogen_in[t]) = 24/n_ts
        d_opex_d_h = float(jac[1, 1]) * 24.0 / self._n_ts
        partials["OpEx", "hydrogen_in"] = np.full(self._n_ts, d_opex_d_h)

    def compute(self, inputs, outputs):
        """Calculate installed capital and O&M costs for lined rock cavern hydrogen storage.

        Args:
            inputs: OpenMDAO inputs containing ``max_capacity`` (total capacity [kg]),
                ``max_charge_rate`` (charge rate [kg/h]), and ``hydrogen_in``
                (timeseries [kg/h]).
            outputs: OpenMDAO outputs dict.

        Sets:
            outputs["CapEx"]: Installed capital cost in 2018 USD (including compressor).
            outputs["OpEx"]: Annual fixed O&M in 2018 USD/yr (excluding electricity).

        Notes:
            Additional parameters from ``storage_input``:

            - h2_storage_kg (float): Total capacity of hydrogen storage [kg].
            - system_flow_rate (float): Average flow rate [kg/day].
            - labor_rate (float): Labor rate, default 37.40 [$2018/hr].
            - insurance (float): Fraction of total investment, default 1%.
            - property_taxes (float): Fraction of total investment, default 1%.
            - licensing_permits (float): Fraction of total investment, default 0.1%.
            - compressor_om (float): Fraction of compressor investment, default 4%.
            - facility_om (float): Fraction of facility investment minus compressor, default 1%.
        """
        storage_input = self.make_storage_input_dict(inputs)

        # Extract input parameters
        h2_storage_kg = storage_input["h2_storage_kg"]  # [kg]
        system_flow_rate = storage_input["system_flow_rate"]  # [kg/day]
        labor_rate = storage_input.get("labor_rate", 37.39817)  # $(2018)/hr
        insurance = storage_input.get("insurance", 1 / 100)  # % of total capital investment
        property_taxes = storage_input.get(
            "property_taxes", 1 / 100
        )  # % of total capital investment
        licensing_permits = storage_input.get(
            "licensing_permits", 0.1 / 100
        )  # % of total capital investment
        comp_om = storage_input.get("compressor_om", 4 / 100)  # % of compressor capital investment
        facility_om = storage_input.get(
            "facility_om", 1 / 100
        )  # % of facility capital investment minus compressor capital investment

        # ============================================================================
        # Calculate CAPEX
        # ============================================================================
        # Installed capital cost per kg from Papadias [2]
        # Coefficients for lined rock cavern storage cost equation
        a = 0.095803
        b = 1.5868
        c = 10.332
        # Calculate installed capital cost per kg using exponential fit
        lined_rock_cavern_storage_capex_per_kg = np.exp(
            a * (np.log(h2_storage_kg / 1000)) ** 2 - b * np.log(h2_storage_kg / 1000) + c
        )  # 2019 [USD] from Papadias [2]
        installed_capex = lined_rock_cavern_storage_capex_per_kg * h2_storage_kg
        cepci_overall = 1.29 / 1.30  # Convert from $2019 to $2018
        installed_capex = cepci_overall * installed_capex

        # ============================================================================
        # Calculate compressor costs
        # ============================================================================
        outlet_pressure = 200  # Max outlet pressure of lined rock cavern in [1] [bar]
        n_compressors = 2
        storage_compressor = Compressor(
            outlet_pressure, system_flow_rate, n_compressors=n_compressors
        )
        storage_compressor.compressor_power()
        motor_rating, power = storage_compressor.compressor_system_power()
        # Check if motor rating exceeds maximum, add additional compressor if needed
        if motor_rating > 1600:
            n_compressors += 1
            storage_compressor = Compressor(
                outlet_pressure, system_flow_rate, n_compressors=n_compressors
            )
            storage_compressor.compressor_power()
            motor_rating, power = storage_compressor.compressor_system_power()
        comp_capex, comp_OM = storage_compressor.compressor_costs()
        cepci = 1.36 / 1.29  # convert from $2016 to $2018
        comp_capex = comp_capex * cepci

        # ============================================================================
        # Calculate OPEX
        # ============================================================================
        # Operations and Maintenance costs [3]
        # Labor
        # Base case is 1 operator, 24 hours a day, 7 days a week for a 100,000 kg/day
        # average capacity facility. Scaling factor of 0.25 is used for other sized facilities
        annual_hours = 8760 * (system_flow_rate / 100000) ** 0.25
        overhead = 0.5
        labor = (annual_hours * labor_rate) * (1 + overhead)  # Burdened labor cost
        insurance_cost = insurance * installed_capex
        property_taxes_cost = property_taxes * installed_capex
        licensing_permits_cost = licensing_permits * installed_capex
        comp_op_maint = comp_om * comp_capex
        facility_op_maint = facility_om * (installed_capex - comp_capex)

        # O&M excludes electricity requirements
        total_om = (
            labor
            + insurance_cost
            + licensing_permits_cost
            + property_taxes_cost
            + comp_op_maint
            + facility_op_maint
        )

        outputs["CapEx"] = installed_capex
        outputs["OpEx"] = total_om


class SaltCavernStorageCostModel(HydrogenStorageBaseCostModel):
    """Capital and operational cost model for salt cavern hydrogen storage.

    Costs are in 2018 USD. Operational dynamics are not yet included.

    References:
        [1] Papadias 2021: https://www.sciencedirect.com/science/article/pii/S0360319921030834?via%3Dihub
        [2] Papadias 2021: Bulk Hydrogen as Function of Capacity.docx documentation at
            hydrogen_storage.md in the docs
        [3] HDSAM V4.0 Gaseous H2 Geologic Storage sheet
    """

    def compute(self, inputs, outputs):
        """Calculate installed capital and O&M costs for salt cavern hydrogen storage.

        Args:
            inputs: OpenMDAO inputs containing ``max_capacity`` (total capacity [kg]),
                ``max_charge_rate`` (charge rate [kg/h]), and ``hydrogen_in``
                (timeseries [kg/h]).
            outputs: OpenMDAO outputs dict.
            discrete_inputs: OpenMDAO discrete inputs dict.
            discrete_outputs: OpenMDAO discrete outputs dict.

        Sets:
            outputs["CapEx"]: Installed capital cost in 2018 USD (including compressor).
            outputs["OpEx"]: Annual fixed O&M in 2018 USD/yr (excluding electricity).

        Notes:
            Additional parameters from ``storage_input``:

            - h2_storage_kg (float): Total capacity of hydrogen storage [kg].
            - system_flow_rate (float): Average flow rate [kg/day].
            - labor_rate (float): Labor rate, default 37.40 [$2018/hr].
            - insurance (float): Fraction of total investment, default 1%.
            - property_taxes (float): Fraction of total investment, default 1%.
            - licensing_permits (float): Fraction of total investment, default 0.1%.
            - compressor_om (float): Fraction of compressor investment, default 4%.
            - facility_om (float): Fraction of facility investment minus compressor, default 1%.
        """
        storage_input = self.make_storage_input_dict(inputs)

        # Extract input parameters
        h2_storage_kg = storage_input["h2_storage_kg"]  # [kg]
        system_flow_rate = storage_input["system_flow_rate"]  # [kg/day]
        labor_rate = storage_input.get("labor_rate", 37.39817)  # $(2018)/hr
        insurance = storage_input.get("insurance", 1 / 100)  # % of total capital investment
        property_taxes = storage_input.get(
            "property_taxes", 1 / 100
        )  # % of total capital investment
        licensing_permits = storage_input.get(
            "licensing_permits", 0.1 / 100
        )  # % of total capital investment
        comp_om = storage_input.get("compressor_om", 4 / 100)  # % of compressor capital investment
        facility_om = storage_input.get(
            "facility_om", 1 / 100
        )  # % of facility capital investment minus compressor capital investment

        # ============================================================================
        # Calculate CAPEX
        # ============================================================================
        # Installed capital cost per kg from Papadias [2]
        # Coefficients for salt cavern storage cost equation
        a = 0.092548
        b = 1.6432
        c = 10.161
        # Calculate installed capital cost per kg using exponential fit
        salt_cavern_storage_capex_per_kg = np.exp(
            a * (np.log(h2_storage_kg / 1000)) ** 2 - b * np.log(h2_storage_kg / 1000) + c
        )  # 2019 [USD] from Papadias [2]
        installed_capex = salt_cavern_storage_capex_per_kg * h2_storage_kg
        cepci_overall = 1.29 / 1.30  # Convert from $2019 to $2018
        installed_capex = cepci_overall * installed_capex

        # ============================================================================
        # Calculate compressor costs
        # ============================================================================
        outlet_pressure = 120  # Max outlet pressure of salt cavern in [1] [bar]
        n_compressors = 2
        storage_compressor = Compressor(
            outlet_pressure, system_flow_rate, n_compressors=n_compressors
        )
        storage_compressor.compressor_power()
        motor_rating, power = storage_compressor.compressor_system_power()
        # Check if motor rating exceeds maximum, add additional compressor if needed
        if motor_rating > 1600:
            n_compressors += 1
            storage_compressor = Compressor(
                outlet_pressure, system_flow_rate, n_compressors=n_compressors
            )
            storage_compressor.compressor_power()
            motor_rating, power = storage_compressor.compressor_system_power()
        comp_capex, comp_OM = storage_compressor.compressor_costs()
        cepci = 1.36 / 1.29  # convert from $2016 to $2018
        comp_capex = comp_capex * cepci

        # ============================================================================
        # Calculate OPEX
        # ============================================================================
        # Operations and Maintenance costs [3]
        # Labor
        # Base case is 1 operator, 24 hours a day, 7 days a week for a 100,000 kg/day
        # average capacity facility. Scaling factor of 0.25 is used for other sized facilities
        annual_hours = 8760 * (system_flow_rate / 100000) ** 0.25
        overhead = 0.5
        labor = (annual_hours * labor_rate) * (1 + overhead)  # Burdened labor cost
        insurance_cost = insurance * installed_capex
        property_taxes_cost = property_taxes * installed_capex
        licensing_permits_cost = licensing_permits * installed_capex
        comp_op_maint = comp_om * comp_capex
        facility_op_maint = facility_om * (installed_capex - comp_capex)

        # O&M excludes electricity requirements
        total_om = (
            labor
            + insurance_cost
            + licensing_permits_cost
            + property_taxes_cost
            + comp_op_maint
            + facility_op_maint
        )

        outputs["CapEx"] = installed_capex
        outputs["OpEx"] = total_om


class PipeStorageCostModel(HydrogenStorageBaseCostModel):
    """Capital and operational cost model for underground pipeline hydrogen storage.

    Costs are in 2018 USD. Operational dynamics and physical size (footprint and
    mass) are not yet included.

    Notes:
        - Oversize pipe: pipe OD = 24" schedule 60 [1].
        - Max pressure: 100 bar.

    References:
        [1] Papadias 2021: https://www.sciencedirect.com/science/article/pii/S0360319921030834?via%3Dihub
        [2] Papadias 2021: Bulk Hydrogen as Function of Capacity.docx documentation at
            hydrogen_storage.md in the docs
        [3] HDSAM V4.0 Gaseous H2 Geologic Storage sheet
    """

    def compute(self, inputs, outputs):
        """Calculate installed capital and O&M costs for underground pipe hydrogen storage.

        Args:
            inputs: OpenMDAO inputs containing ``max_capacity`` (total capacity [kg]),
                ``max_charge_rate`` (charge rate [kg/h]), and ``hydrogen_in``
                (timeseries [kg/h]).
            outputs: OpenMDAO outputs dict.
            discrete_inputs: OpenMDAO discrete inputs dict.
            discrete_outputs: OpenMDAO discrete outputs dict.

        Sets:
            outputs["CapEx"]: Installed capital cost in 2018 USD (including compressor).
            outputs["OpEx"]: Annual fixed O&M in 2018 USD/yr (excluding electricity).

        Notes:
            - Oversize pipe: pipe OD = 24" schedule 60.
            - Max pressure: 100 bar.
            - ``compressor_output_pressure`` must be 100 bar for underground pipe storage.

            Additional parameters from ``storage_input``:

            - h2_storage_kg (float): Total capacity of hydrogen storage [kg].
            - system_flow_rate (float): Average flow rate [kg/day].
            - labor_rate (float): Labor rate, default 37.40 [$2018/hr].
            - insurance (float): Fraction of total investment, default 1%.
            - property_taxes (float): Fraction of total investment, default 1%.
            - licensing_permits (float): Fraction of total investment, default 0.1%.
            - compressor_om (float): Fraction of compressor investment, default 4%.
            - facility_om (float): Fraction of facility investment minus compressor, default 1%.
        """
        storage_input = self.make_storage_input_dict(inputs)

        # Extract input parameters
        h2_storage_kg = storage_input["h2_storage_kg"]  # [kg]
        system_flow_rate = storage_input["system_flow_rate"]  # [kg/day]
        labor_rate = storage_input.get("labor_rate", 37.39817)  # $(2018)/hr
        insurance = storage_input.get("insurance", 1 / 100)  # % of total capital investment
        property_taxes = storage_input.get(
            "property_taxes", 1 / 100
        )  # % of total capital investment
        licensing_permits = storage_input.get(
            "licensing_permits", 0.1 / 100
        )  # % of total capital investment
        comp_om = storage_input.get("compressor_om", 4 / 100)  # % of compressor capital investment
        facility_om = storage_input.get(
            "facility_om", 1 / 100
        )  # % of facility capital investment minus compressor capital investment

        # compressor_output_pressure must be 100 bar for underground pipe storage
        compressor_output_pressure = 100  # [bar]

        # ============================================================================
        # Calculate CAPEX
        # ============================================================================
        # Installed capital cost per kg from Papadias [2]
        # Coefficients for underground pipe storage cost equation
        a = 0.0041617
        b = 0.060369
        c = 6.4581
        # Calculate installed capital cost per kg using exponential fit
        pipe_storage_capex_per_kg = np.exp(
            a * (np.log(h2_storage_kg / 1000)) ** 2 - b * np.log(h2_storage_kg / 1000) + c
        )  # 2019 [USD] from Papadias [2]
        installed_capex = pipe_storage_capex_per_kg * h2_storage_kg
        cepci_overall = 1.29 / 1.30  # Convert from $2019 to $2018
        installed_capex = cepci_overall * installed_capex

        # ============================================================================
        # Calculate compressor costs
        # ============================================================================
        outlet_pressure = (
            compressor_output_pressure  # Max outlet pressure of underground pipe storage [1] [bar]
        )
        n_compressors = 2
        storage_compressor = Compressor(
            outlet_pressure, system_flow_rate, n_compressors=n_compressors
        )
        storage_compressor.compressor_power()
        motor_rating, power = storage_compressor.compressor_system_power()
        # Check if motor rating exceeds maximum, add additional compressor if needed
        if motor_rating > 1600:
            n_compressors += 1
            storage_compressor = Compressor(
                outlet_pressure, system_flow_rate, n_compressors=n_compressors
            )
            storage_compressor.compressor_power()
            motor_rating, power = storage_compressor.compressor_system_power()
        comp_capex, comp_OM = storage_compressor.compressor_costs()
        cepci = 1.36 / 1.29  # convert from $2016 to $2018
        comp_capex = comp_capex * cepci

        # ============================================================================
        # Calculate OPEX
        # ============================================================================
        # Operations and Maintenance costs [3]
        # Labor
        # Base case is 1 operator, 24 hours a day, 7 days a week for a 100,000 kg/day
        # average capacity facility. Scaling factor of 0.25 is used for other sized facilities
        annual_hours = 8760 * (system_flow_rate / 100000) ** 0.25
        overhead = 0.5
        labor = (annual_hours * labor_rate) * (1 + overhead)  # Burdened labor cost
        insurance_cost = insurance * installed_capex
        property_taxes_cost = property_taxes * installed_capex
        licensing_permits_cost = licensing_permits * installed_capex
        comp_op_maint = comp_om * comp_capex
        facility_op_maint = facility_om * (installed_capex - comp_capex)

        # O&M excludes electricity requirements
        total_om = (
            labor
            + insurance_cost
            + licensing_permits_cost
            + property_taxes_cost
            + comp_op_maint
            + facility_op_maint
        )

        outputs["CapEx"] = installed_capex
        outputs["OpEx"] = total_om
