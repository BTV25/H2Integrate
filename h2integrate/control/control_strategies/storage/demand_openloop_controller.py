from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from attrs import field, define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.validators import gte_zero, range_val, range_val_or_none
from h2integrate.control.control_strategies.demand_openloop_controller import (
    DemandOpenLoopControlBase,
    DemandOpenLoopControlBaseConfig,
)


def _make_jax_compute(
    max_charge_fraction,
    min_charge_fraction,
    init_charge_fraction,
    charge_efficiency,
    discharge_efficiency,
    charge_equals_discharge,
    fixed_max_discharge_rate,
):
    """Returns a pure JAX function implementing the storage controller time-step scan.

    When charge_equals_discharge=True, max_discharge_rate tracks max_charge_rate
    (so gradients flow through it). Otherwise fixed_max_discharge_rate is a constant.
    """

    def jax_compute(commodity_in, demand, max_charge_rate, max_capacity):
        mdr = (
            max_charge_rate
            if charge_equals_discharge
            else jnp.array(fixed_max_discharge_rate, dtype=max_charge_rate.dtype)
        )

        def step(soc, x):
            demand_t, input_flow = x[0], x[1]

            avail_discharge = jnp.maximum(0.0, (soc - min_charge_fraction) * max_capacity)
            avail_charge = jnp.maximum(0.0, (max_charge_fraction - soc) * max_capacity)

            discharge = jnp.minimum(
                jnp.minimum(
                    jnp.maximum(0.0, demand_t - input_flow) / discharge_efficiency,
                    avail_discharge,
                ),
                mdr / discharge_efficiency,
            )

            unused_input = jnp.maximum(0.0, input_flow - demand_t)
            charge = (
                jnp.maximum(
                    0.0,
                    jnp.minimum(
                        jnp.minimum(unused_input, avail_charge / charge_efficiency),
                        max_charge_rate,
                    ),
                )
                * charge_efficiency
            )

            discharging = demand_t > input_flow
            act_discharge = jnp.where(discharging, discharge, 0.0)
            act_charge = jnp.where(discharging, 0.0, charge)
            act_unused = jnp.where(discharging, 0.0, unused_input)

            new_soc = jnp.clip(
                soc + (act_charge - act_discharge) / max_capacity,
                min_charge_fraction,
                max_charge_fraction,
            )
            output = jnp.where(
                discharging,
                input_flow + act_discharge * discharge_efficiency,
                demand_t,
            )
            unused_commodity = jnp.maximum(0.0, act_unused - act_charge / charge_efficiency)
            unmet = jnp.maximum(0.0, demand_t - output)

            return new_soc, (output, new_soc, unused_commodity, unmet)

        init_soc = jnp.array(init_charge_fraction, dtype=commodity_in.dtype)
        _, (set_point, soc_arr, unused, unmet) = jax.lax.scan(
            step, init_soc, jnp.stack([demand, commodity_in], axis=-1)
        )
        return set_point, soc_arr, unused, unmet, jnp.sum(unmet)

    return jax_compute


@define(kw_only=True)
class DemandOpenLoopStorageControllerConfig(DemandOpenLoopControlBaseConfig):
    """
    Configuration class for the DemandOpenLoopStorageController.

    This class defines the parameters required to configure the `DemandOpenLoopStorageController`.

    Attributes:
        max_capacity (float): Maximum storage capacity of the commodity (in non-rate units,
            e.g., "kg" if `commodity_rate_units` is "kg/h").
        max_charge_fraction (float): Maximum allowable state of charge (SOC) as a fraction
            of `max_capacity`, between 0 and 1.
        min_charge_fraction (float): Minimum allowable SOC as a fraction of `max_capacity`,
            between 0 and 1.
        init_charge_fraction (float): Initial SOC as a fraction of `max_capacity`,
            between 0 and 1.
        max_charge_rate (float): Maximum rate at which the commodity can be charged (in units
            per time step, e.g., "kg/time step"). This rate does not include the charge_efficiency.
        charge_equals_discharge (bool, optional): If True, set the max_discharge_rate equal to the
            max_charge_rate. If False, specify the max_discharge_rate as a value different than
            the max_charge_rate. Defaults to True.
        max_discharge_rate (float | None, optional): Maximum rate at which the commodity can be
            discharged (in units per time step, e.g., "kg/time step"). This rate does not include
            the discharge_efficiency. Only required if `charge_equals_discharge` is False.
        charge_efficiency (float | None, optional): Efficiency of charging the storage, represented
            as a decimal between 0 and 1 (e.g., 0.9 for 90% efficiency). Optional if
            `round_trip_efficiency` is provided.
        discharge_efficiency (float | None, optional): Efficiency of discharging the storage,
            represented as a decimal between 0 and 1 (e.g., 0.9 for 90% efficiency). Optional if
            `round_trip_efficiency` is provided.
        round_trip_efficiency (float | None, optional): Combined efficiency of charging and
            discharging the storage, represented as a decimal between 0 and 1 (e.g., 0.81 for
            81% efficiency). Optional if `charge_efficiency` and `discharge_efficiency` are
            provided.
    """

    max_capacity: float = field()
    max_charge_fraction: float = field(validator=range_val(0, 1))
    min_charge_fraction: float = field(validator=range_val(0, 1))
    init_charge_fraction: float = field(validator=range_val(0, 1))
    max_charge_rate: float = field(validator=gte_zero)
    charge_equals_discharge: bool = field(default=True)
    max_discharge_rate: float | None = field(default=None)
    charge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    discharge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    round_trip_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))

    def __attrs_post_init__(self):
        """
        Post-initialization logic to validate and calculate efficiencies.

        Ensures that either `charge_efficiency` and `discharge_efficiency` are provided,
        or `round_trip_efficiency` is provided. If `round_trip_efficiency` is provided,
        it calculates `charge_efficiency` and `discharge_efficiency` as the square root
        of `round_trip_efficiency`.
        """
        if self.round_trip_efficiency is not None:
            if self.charge_efficiency is not None or self.discharge_efficiency is not None:
                raise ValueError(
                    "Provide either `round_trip_efficiency` or both `charge_efficiency` "
                    "and `discharge_efficiency`, but not both."
                )
            # Calculate charge and discharge efficiencies from round-trip efficiency
            self.charge_efficiency = np.sqrt(self.round_trip_efficiency)
            self.discharge_efficiency = np.sqrt(self.round_trip_efficiency)
        elif self.charge_efficiency is not None and self.discharge_efficiency is not None:
            # Ensure both charge and discharge efficiencies are provided
            pass
        else:
            raise ValueError(
                "You must provide either `round_trip_efficiency` or both "
                "`charge_efficiency` and `discharge_efficiency`."
            )

        if self.charge_equals_discharge:
            if (
                self.max_discharge_rate is not None
                and self.max_discharge_rate != self.max_charge_rate
            ):
                msg = (
                    "Max discharge rate does not equal max charge rate but charge_equals_discharge "
                    f"is True. Discharge rate is {self.max_discharge_rate} and charge rate "
                    f"is {self.max_charge_rate}."
                )
                raise ValueError(msg)

            self.max_discharge_rate = self.max_charge_rate



class DemandOpenLoopStorageController(DemandOpenLoopControlBase):
    """
    A controller that manages commodity flow based on demand and storage constraints.

    The `DemandOpenLoopStorageController` computes the state of charge (SOC), output flow,
    curtailment, and missed load for a commodity storage system. It uses a demand profile
    and storage parameters to determine how much of the commodity to charge, discharge,
    or curtail at each time step.

    Note: the units of the outputs are the same as the commodity units, which is typically a rate
    in H2Integrate (e.g. kg/h)

    Attributes:
        config (DemandOpenLoopStorageControllerConfig): Configuration object containing parameters
            such as commodity name, units, time steps, storage capacity, charge/discharge rates,
            efficiencies, and demand profile.

    Inputs:
        {commodity}_in (float): Input commodity flow timeseries (e.g., hydrogen production).
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").

    Outputs:
        {commodity}_out (float): Output commodity flow timeseries after storage to meet demand.
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").
            - Note: the may include commodity from commodity_in that was never used to charge the
                    storage system but was directly dispatched to meet demand.
        {commodity}_soc (float): State of charge (SOC) timeseries for the storage system.
            - Units: "unitless" (percentage of maximum capacity given as a ratio between 0 and 1).
        {commodity}_unused_commodity (float): Curtailment timeseries for unused
            input commodity.
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").
            - Note: curtailment in this case does not reduce what the converter produces, but
                rather the system just does not use it (throws it away) because this controller is
                specific to the storage technology and has no influence on other technologies in
                the system.
        {commodity}_unmet_demand (float): Unmet demand timeseries when demand exceeds supply.
            Same meaning as missed load.
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").
        total_{commodity}_unmet_demand (float): Total unmet demand over the simulation period.
            - Units: Defined in `commodity_rate_units` (e.g., "kg").
    """

    def setup(self):
        """
        Set up inputs, outputs, and configuration for the open-loop storage controller.

        This method initializes the controller configuration from the technology
        configuration, establishes the number of simulation time steps, adds inputs
        related to storage constraints (e.g., maximum charge rate and storage capacity),
        and defines outputs such as the commodity state-of-charge (SOC) timeseries
        and the estimated storage duration.

        Inputs defined:
            * ``max_charge_rate``: Maximum rate at which storage can charge/discharge.
            * ``max_capacity``: Maximum total storage capacity.

        Outputs defined:
            * ``<commodity>_soc``: Timeseries of storage state of charge.
            * ``storage_duration``: Estimated duration (hours) the storage can
                discharge at its maximum rate.

        Returns:
            None
        """
        self.config = DemandOpenLoopStorageControllerConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "control"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        commodity = self.config.commodity

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=self.config.commodity_rate_units,
            desc="Storage charge/discharge rate",
        )

        self.add_input(
            "max_capacity",
            val=self.config.max_capacity,
            units=self.config.commodity_rate_units + "*h",
            desc="Maximum storage capacity",
        )

        self.add_output(
            f"{commodity}_soc",
            copy_shape=f"{commodity}_in",
            units="unitless",
            desc=f"{commodity} state of charge timeseries for storage",
        )

        self.add_output(
            "storage_duration",
            units="h",
            desc="Estimated storage duration based on max capacity and discharge rate",
        )

    def compute(self, inputs, outputs):
        """
        Compute storage state of charge (SOC), delivered output, curtailment, and unmet
        demand over the simulation horizon.

        This method applies an open-loop storage control strategy to balance the
        commodity demand and input flow. When input exceeds demand, excess commodity
        is used to charge storage (subject to rate, efficiency, and SOC limits). When
        demand exceeds input, storage is discharged to meet the deficit (also subject
        to constraints). SOC is updated at each time step, ensuring it remains within
        allowable bounds.

        Expected input keys:
            * ``<commodity>_in``: Timeseries of commodity available at each time step.
            * ``<commodity>_demand``: Timeseries demand profile.
            * ``max_charge_rate``: Maximum charge rate permitted.
            * ``max_capacity``: Maximum total storage capacity.

        Outputs populated:
            * ``<commodity>_soc``: State-of-charge timeseries (unitless).
            * ``<commodity>_out``: Output delivered to meet demand.
            * ``<commodity>_unused_commodity``: Curtailment timeseries.
            * ``<commodity>_unmet_demand``: Missed load timeseries.
            * ``total_<commodity>_unmet_demand``: Aggregated unmet demand.
            * ``storage_duration``: Estimated discharge duration at maximum rate (hours).

        Control logic includes:
            * Enforcing SOC limits (min, max, and initial conditions).
            * Applying charge and discharge efficiencies.
            * Observing charge/discharge rate limits.
            * Tracking energy shortfalls and excesses at each time step.

        Raises:
            UserWarning: If the demand profile is entirely zero.
            UserWarning: If ``max_charge_rate`` or ``max_capacity`` is negative.

        Returns:
            None
        """
        commodity = self.config.commodity
        if np.all(inputs[f"{commodity}_demand"] == 0.0):
            msg = "Demand profile is zero, check that demand profile is input"
            raise UserWarning(msg)
        if inputs["max_charge_rate"][0] < 0:
            msg = (
                f"max_charge_rate cannot be less than zero and has value of "
                f"{inputs['max_charge_rate']}"
            )
            raise UserWarning(msg)
        if inputs["max_capacity"][0] < 0:
            msg = (
                f"max_capacity cannot be less than zero and has value of "
                f"{inputs['max_capacity']}"
            )
            raise UserWarning(msg)

        max_capacity = inputs["max_capacity"].item()
        max_charge_fraction = self.config.max_charge_fraction
        min_charge_fraction = self.config.min_charge_fraction
        init_charge_fraction = self.config.init_charge_fraction
        max_charge_rate = inputs["max_charge_rate"].item()
        if self.config.charge_equals_discharge:
            max_discharge_rate = inputs["max_charge_rate"].item()
        else:
            max_discharge_rate = float(self.config.max_discharge_rate)
        charge_efficiency = float(self.config.charge_efficiency)
        discharge_efficiency = float(self.config.discharge_efficiency)

        # Initialize time-step state of charge prior to loop so the loop starts with
        # the previous time step's value
        soc = deepcopy(init_charge_fraction)

        demand_profile = inputs[f"{commodity}_demand"]

        # initialize outputs
        soc_array = outputs[f"{commodity}_soc"]
        unused_commodity_array = outputs[f"{commodity}_unused_commodity"]
        output_array = outputs[f"{commodity}_set_point"]
        unmet_demand_array = outputs[f"{commodity}_unmet_demand"]

        # Loop through each time step
        for t, demand_t in enumerate(demand_profile):
            # Get the input flow at the current time step
            input_flow = inputs[f"{commodity}_in"][t]

            # Calculate the available charge/discharge capacity
            available_charge = float((max_charge_fraction - soc) * max_capacity)
            available_discharge = float((soc - min_charge_fraction) * max_capacity)

            # Initialize persistent variables for curtailment and missed load
            unused_input = 0.0
            charge = 0.0

            # Determine the output flow based on demand_t and SOC
            if demand_t > input_flow:
                # Discharge storage to meet demand.
                # `discharge_needed` is as seen by the storage
                discharge_needed = (demand_t - input_flow) / discharge_efficiency
                # `discharge` is as seen by the storage, but `max_discharge_rate` is as observed
                # outside the storage
                discharge = min(
                    discharge_needed, available_discharge, max_discharge_rate / discharge_efficiency
                )

                soc -= discharge / max_capacity  # soc is a ratio with value between 0 and 1
                # output is as observed outside the storage, so we need to adjust `discharge` by
                # applying `discharge_efficiency`.
                output_array[t] = input_flow + discharge * discharge_efficiency
            else:
                # Charge storage with unused input
                # `unused_input` is as seen outside the storage
                unused_input = input_flow - demand_t
                unused_input = unused_input.item()
                # `charge` is as seen by the storage, but the things being compared should all be as
                # seen outside the storage so we need to adjust `available_charge` outside the
                # storage view and the final result back into the storage view.
                charge = (
                    min(unused_input, available_charge / charge_efficiency, max_charge_rate)
                    * charge_efficiency
                )
                soc += charge / max_capacity  # soc is a ratio with value between 0 and 1
                output_array[t] = demand_t

            # Ensure SOC stays within bounds
            soc = max(min_charge_fraction, min(max_charge_fraction, soc))

            # Record the SOC for the current time step
            soc_array[t] = deepcopy(soc)

            # Record the curtailment at the current time step. Adjust `charge` from storage view to
            # outside view for curtailment
            unused_commodity_array[t] = max(0.0, unused_input - charge / charge_efficiency)

            # Record the missed load at the current time step
            unmet_demand_array[t] = max(0.0, (demand_t - output_array[t]))

        outputs[f"{commodity}_set_point"] = output_array

        # Return the SOC
        outputs[f"{commodity}_soc"] = soc_array

        # Return the unused commodity
        outputs[f"{commodity}_unused_commodity"] = unused_commodity_array

        # Return the unmet load demand
        outputs[f"{commodity}_unmet_demand"] = unmet_demand_array

        # Calculate and return the total unmet demand over the simulation period
        outputs[f"total_{commodity}_unmet_demand"] = np.sum(unmet_demand_array)

        # Output the storage duration in hours
        outputs["storage_duration"] = max_capacity / max_discharge_rate

    def setup_partials(self):
        commodity = self.config.commodity
        cfg = self.config

        ts_out = [
            f"{commodity}_set_point",
            f"{commodity}_soc",
            f"{commodity}_unused_commodity",
            f"{commodity}_unmet_demand",
        ]
        all_out = ts_out + [f"total_{commodity}_unmet_demand"]
        all_in = [f"{commodity}_in", f"{commodity}_demand", "max_charge_rate", "max_capacity"]

        for out in all_out:
            for inp in all_in:
                self.declare_partials(out, inp)
        self.declare_partials("storage_duration", ["max_charge_rate", "max_capacity"])

        fixed_mdr = None if cfg.charge_equals_discharge else float(cfg.max_discharge_rate)
        # jacfwd, not jacobian/jacrev: outputs (4x8760+1 rows) outnumber inputs
        # (2x8760+2 cols), so forward mode halves the batch width through the scan
        # and avoids the scan-transpose residual traffic (measured 3.2x faster,
        # identical Jacobian; see research notes 2026-07-14 Session 7)
        self._jax_jac = jax.jit(
            jax.jacfwd(
                _make_jax_compute(
                    cfg.max_charge_fraction,
                    cfg.min_charge_fraction,
                    cfg.init_charge_fraction,
                    float(cfg.charge_efficiency),
                    float(cfg.discharge_efficiency),
                    cfg.charge_equals_discharge,
                    fixed_mdr,
                ),
                argnums=(0, 1, 2, 3),
            )
        )

    def compute_partials(self, inputs, partials):
        commodity = self.config.commodity

        ci = jnp.array(inputs[f"{commodity}_in"])
        demand = jnp.array(inputs[f"{commodity}_demand"])
        max_charge_rate = jnp.array(inputs["max_charge_rate"][0])
        max_capacity = jnp.array(inputs["max_capacity"][0])

        # jac[i_out][i_in]: Jacobian of output i_out wrt arg i_in
        # out order: (set_point, soc, unused, unmet, total_unmet)
        # arg order: (ci, demand, max_charge_rate, max_capacity)
        jac = self._jax_jac(ci, demand, max_charge_rate, max_capacity)

        out_names = [
            f"{commodity}_set_point",
            f"{commodity}_soc",
            f"{commodity}_unused_commodity",
            f"{commodity}_unmet_demand",
            f"total_{commodity}_unmet_demand",
        ]
        in_names = [f"{commodity}_in", f"{commodity}_demand", "max_charge_rate", "max_capacity"]

        for i_out, out in enumerate(out_names):
            for i_in, inp in enumerate(in_names):
                partials[out, inp] = np.asarray(jac[i_out][i_in])

        # storage_duration = max_capacity / max_discharge_rate (analytic)
        mc = float(inputs["max_capacity"][0])
        mcr = float(inputs["max_charge_rate"][0])
        mdr = mcr if self.config.charge_equals_discharge else float(self.config.max_discharge_rate)
        partials["storage_duration", "max_capacity"] = 1.0 / mdr
        partials["storage_duration", "max_charge_rate"] = (
            -mc / mdr**2 if self.config.charge_equals_discharge else 0.0
        )
