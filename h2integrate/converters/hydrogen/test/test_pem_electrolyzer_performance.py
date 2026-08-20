import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.converters.hydrogen.pem_electrolyzer import ECOElectrolyzerPerformanceModel


@fixture
def plant_config():
    plant_config = {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "n_timesteps": 8760,
                "dt": 3600,
            },
        },
    }
    return plant_config


@fixture
def tech_config():
    config = {
        "model_inputs": {
            "performance_parameters": {
                "n_clusters": 4.0,
                "location": "onshore",
                "cluster_rating_MW": 10,
                "eol_eff_percent_loss": 10.0,
                "uptime_hours_until_eol": 8000,
                "include_degradation_penalty": True,
                "turndown_ratio": 0.1,
                "activation_frac": 0.1,
                "electrolyzer_capex": 10.0,
            }
        }
    }
    return config


@pytest.mark.unit
def test_electrolyzer_outputs(tech_config, plant_config, subtests):
    plant_life = int(plant_config["plant"]["plant_life"])
    n_timesteps = int(plant_config["plant"]["simulation"]["n_timesteps"])

    prob = om.Problem()
    comp = ECOElectrolyzerPerformanceModel(
        plant_config=plant_config, tech_config=tech_config, driver_config={}
    )
    prob.model.add_subsystem("comp", comp, promotes=["*"])
    prob.setup()
    power_profile = np.ones(n_timesteps) * 32.0
    prob.set_val("comp.electricity_in", power_profile, units="MW")

    prob.run_model()

    commodity = "hydrogen"
    commodity_amount_units = "kg"
    commodity_rate_units = "kg/h"

    # Check that replacement schedule is between 0 and 1
    with subtests.test("0 <= replacement_schedule <=1"):
        assert np.all(prob.get_val("comp.replacement_schedule", units="unitless") >= 0)
        assert np.all(prob.get_val("comp.replacement_schedule", units="unitless") <= 1)

    with subtests.test("replacement_schedule length"):
        assert len(prob.get_val("comp.replacement_schedule", units="unitless")) == plant_life

    # Check that capacity factor is between 0 and 1 with units of "unitless"
    with subtests.test("0 <= capacity_factor (unitless) <=1"):
        assert np.all(prob.get_val("comp.capacity_factor", units="unitless") >= 0)
        assert np.all(prob.get_val("comp.capacity_factor", units="unitless") <= 1)

    # Check that capacity factor is between 1 and 100 with units of "percent"
    with subtests.test("1 <= capacity_factor (percent) <=1"):
        assert np.all(prob.get_val("comp.capacity_factor", units="percent") >= 1)
        assert np.all(prob.get_val("comp.capacity_factor", units="percent") <= 100)

    with subtests.test("capacity_factor length"):
        assert len(prob.get_val("comp.capacity_factor", units="unitless")) == plant_life

    # Test that rated commodity production is greater than zero
    with subtests.test(f"rated_{commodity}_production > 0"):
        assert np.all(
            prob.get_val(f"comp.rated_{commodity}_production", units=commodity_rate_units) > 0
        )

    with subtests.test(f"rated_{commodity}_production length"):
        assert (
            len(prob.get_val(f"comp.rated_{commodity}_production", units=commodity_rate_units)) == 1
        )

    # Test that total commodity production is greater than zero
    with subtests.test(f"total_{commodity}_produced > 0"):
        assert np.all(
            prob.get_val(f"comp.total_{commodity}_produced", units=commodity_amount_units) > 0
        )
    with subtests.test(f"total_{commodity}_produced length"):
        assert (
            len(prob.get_val(f"comp.total_{commodity}_produced", units=commodity_amount_units)) == 1
        )

    # Test that annual commodity production is greater than zero
    with subtests.test(f"annual_{commodity}_produced > 0"):
        assert np.all(
            prob.get_val(f"comp.annual_{commodity}_produced", units=f"{commodity_amount_units}/yr")
            > 0
        )

    with subtests.test(f"annual_{commodity}_produced[1:] == annual_{commodity}_produced[0]"):
        annual_production = prob.get_val(
            f"comp.annual_{commodity}_produced", units=f"{commodity_amount_units}/yr"
        )
        assert np.all(annual_production[1:] == annual_production[0])

    with subtests.test(f"annual_{commodity}_produced length"):
        assert len(annual_production) == plant_life

    # Test that commodity output has some values greater than zero
    with subtests.test(f"Some of {commodity}_out > 0"):
        assert np.any(prob.get_val(f"comp.{commodity}_out", units=commodity_rate_units) > 0)

    with subtests.test(f"{commodity}_out length"):
        assert len(prob.get_val(f"comp.{commodity}_out", units=commodity_rate_units)) == n_timesteps

    # Test default values
    with subtests.test("operational_life default value"):
        assert prob.get_val("comp.operational_life", units="yr") == plant_life
    with subtests.test("replacement_schedule value"):
        assert np.any(prob.get_val("comp.replacement_schedule", units="unitless") == 0)


@pytest.mark.unit
def test_n_clusters_sizing_gradient_across_bin_boundary(tech_config, plant_config, subtests):
    """d(total_hydrogen_produced)/d(n_clusters) must match FD on both sides of, and exactly
    at, a cluster-count bin boundary -- regression test for the PCHIP sizing smoothing
    (previously this partial was only correct within one round()-based bin)."""
    n_timesteps = int(plant_config["plant"]["simulation"]["n_timesteps"])
    power_profile = np.ones(n_timesteps) * 32.0
    step = 1e-4

    def total_h2_and_size(n_clusters_val):
        prob = om.Problem()
        comp = ECOElectrolyzerPerformanceModel(
            plant_config=plant_config, tech_config=tech_config, driver_config={}
        )
        prob.model.add_subsystem("comp", comp, promotes=["*"])
        prob.setup()
        prob.set_val("comp.electricity_in", power_profile, units="MW")
        prob.set_val("comp.n_clusters", n_clusters_val)
        prob.run_model()
        return (
            float(prob.get_val("comp.total_hydrogen_produced")[0]),
            float(prob.get_val("comp.electrolyzer_size_mw")[0]),
            comp,
        )

    for n_clusters_val in [2.1, 2.5, 2.9]:
        with subtests.test(f"n_clusters={n_clusters_val}"):
            h2_0, size_0, comp = total_h2_and_size(n_clusters_val)
            h2_1, size_1, _ = total_h2_and_size(n_clusters_val + step)

            fd_h2 = (h2_1 - h2_0) / step
            fd_size = (size_1 - size_0) / step
            analytic_h2 = float(comp._pchip_total_deriv(comp._n_continuous))
            analytic_size = float(comp.config.cluster_rating_MW)

            assert np.isclose(fd_h2, analytic_h2, rtol=1e-2)
            assert np.isclose(fd_size, analytic_size, rtol=1e-6)
