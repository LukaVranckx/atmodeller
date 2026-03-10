#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Core classes and functions for thermochemical and critical data"""

import importlib.resources
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from jaxmod.constants import GAS_CONSTANT
from jaxmod.units import unit_conversion
from jaxmod.utils import as_j64, to_native_floats
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller.constants import TEMPERATURE_REFERENCE

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory"""
THERMODYNAMIC_DATA_SOURCE: Path = Path("nasa_glenn_coefficients.txt")
"""Source of the thermodynamic data"""
CRITICAL_DATA_SOURCE: Path = Path("critical_data.txt")
"""Source of the critical data"""


class CondensateActivity(eqx.Module):
    """Activity of a stable condensate

    Args:
        activity: Activity. Defaults to ``1``.
    """

    activity: Array = eqx.field(converter=as_j64, default=1)
    """Activity"""

    def active(self) -> Bool[Array, "..."]:
        """Active activity constraint

        Condensate activity is imposed in the reaction network and therefore is never part of an
        active constraint in the residual.

        Returns:
            Always ``False`` because it does not require solution.
        """
        return jnp.full_like(self.activity, False, dtype=jnp.bool_)

    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> Float[Array, "..."]:
        """Log activity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log activity, which is dimensionless
        """
        del temperature
        del pressure

        return jnp.log(self.activity)

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Float[Array, "..."]:
        return self.log_activity(temperature, pressure)


class ThermodynamicCoefficients(eqx.Module):
    """NASA Glenn coefficients for the thermodynamic properties of an individual species

    Coefficients are available at https://ntrs.nasa.gov/citations/20020085330

    Args:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in K in the range
        T_max: Maximum temperature(s) in K in the range
    """

    b1: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Enthalpy constant(s) of integration"""
    b2: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Entropy constant(s) of integration"""
    cp_coeffs: tuple[tuple[float, ...], ...] = eqx.field(converter=to_native_floats)
    """Heat capacity coefficients"""
    T_min: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Minimum temperature(s) in K in the range"""
    T_max: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Maximum temperature(s) in K in the range"""

    def _get_index(self, temperature: ArrayLike) -> Integer[Array, " T"]:
        """Gets the index of the temperature range for the given temperature

        This assumes the temperature is within one of the ranges and will produce unexpected output
        if the temperature is outside the ranges.

        Args:
            temperature: Temperature in K

        Returns:
            Index of the temperature range
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        T_max: Array = as_j64(self.T_max)
        T_min: Array = as_j64(self.T_min)

        # Reshape for broadcasting
        bool_mask: Bool[Array, "N T"] = (T_min[:, None] <= temperature[None, :]) & (
            temperature[None, :] <= T_max[:, None]
        )
        index: Integer[Array, " T"] = jnp.argmax(bool_mask, axis=0)

        return index

    def _cp_over_R(
        self, cp_coefficients: Float[Array, "T 7"], temperature: ArrayLike
    ) -> Float[Array, " T"]:
        """Heat capacity relative to :const:`~atmodeller.constants.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            temperature: Temperature in K

        Returns:
            Heat capacity relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        temperature_terms: Float[Array, "T 7"] = jnp.stack(
            [
                jnp.power(temperature, -2),
                jnp.power(temperature, -1),
                jnp.ones_like(temperature),
                temperature,
                jnp.power(temperature, 2),
                jnp.power(temperature, 3),
                jnp.power(temperature, 4),
            ],
            axis=-1,
        )

        heat_capacity: Float[Array, " T"] = jnp.einsum(
            "ti,ti->t", cp_coefficients, temperature_terms
        )

        return heat_capacity

    def _S_over_R(
        self, cp_coefficients: Float[Array, "T 7"], b2: ArrayLike, temperature: ArrayLike
    ) -> Float[Array, " T"]:
        """Entropy relative to :const:`~atmodeller.constants.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            b2: Entropy integration constant
            temperature: Temperature in K

        Returns:
            Entropy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        temperature_terms: Float[Array, "T 7"] = jnp.stack(
            [
                -jnp.power(temperature, -2) / 2,
                -jnp.power(temperature, -1),
                jnp.log(temperature),
                temperature,
                jnp.power(temperature, 2) / 2,
                jnp.power(temperature, 3) / 3,
                jnp.power(temperature, 4) / 4,
            ],
            axis=-1,
        )

        entropy: Float[Array, " T"] = (
            jnp.einsum("ti,ti->t", cp_coefficients, temperature_terms) + b2
        )

        return entropy

    def _H_over_RT(
        self, cp_coefficients: Float[Array, "T 7"], b1: ArrayLike, temperature: ArrayLike
    ) -> Float[Array, " T"]:
        r"""Enthalpy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            temperature: Temperature in K

        Returns:
            Enthalpy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        temperature_terms: Float[Array, "T 7"] = jnp.stack(
            [
                -jnp.power(temperature, -2),
                jnp.log(temperature) / temperature,
                jnp.ones_like(temperature),
                temperature / 2,
                jnp.power(temperature, 2) / 3,
                jnp.power(temperature, 3) / 4,
                jnp.power(temperature, 4) / 5,
            ],
            axis=-1,
        )

        enthalpy: Float[Array, " T"] = (
            jnp.einsum("ti,ti->t", cp_coefficients, temperature_terms) + b1 / temperature
        )

        return enthalpy

    def _G_over_RT(
        self,
        cp_coefficients: Float[Array, "T 7"],
        b1: ArrayLike,
        b2: ArrayLike,
        temperature: ArrayLike,
    ) -> Float[Array, " T"]:
        r"""Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            b2: Entropy integration constant
            temperature: Temperature in K

        Returns:
            Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        enthalpy: Float[Array, " T"] = self._H_over_RT(cp_coefficients, b1, temperature)
        # jax.debug.print("enthalpy = {out}", out=enthalpy)
        entropy: Float[Array, " T"] = self._S_over_R(cp_coefficients, b2, temperature)
        # jax.debug.print("entropy = {out}", out=entropy)
        # No temperature multiplication is correct since the return is Gibbs energy relative to RT
        gibbs: Float[Array, " T"] = enthalpy - entropy

        return gibbs

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets Gibbs energy to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index)
        b1_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        b2_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b2), index)
        # jax.debug.print("b2_for_index = {out}", out=b2_for_index)
        gibbs_for_index: Float[Array, " T"] = self._G_over_RT(
            cp_coeffs_for_index, b1_for_index, b2_for_index, temperature
        )

        return gibbs_for_index

    def cp(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index.shape)
        cp: Float[Array, " T"] = self._cp_over_R(cp_coeffs_for_index, temperature) * GAS_CONSTANT

        return cp

    def enthalpy(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        b1_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b1), index)
        enthalpy: Float[Array, " T"] = (
            self._H_over_RT(cp_coeffs_for_index, b1_for_index, temperature)
            * GAS_CONSTANT
            * temperature
        )

        return enthalpy

    def reference_enthalpy(self) -> Float[Array, ""]:
        r"""Gets reference enthalpy.

        This is :math:`H^{\circ}(T_r)` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Reference enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        index: Integer[Array, ""] = self._get_index(TEMPERATURE_REFERENCE)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Float[Array, "7"] = jnp.take(jnp.array(self.cp_coeffs), index, axis=0)
        b1_for_index: Float[Array, ""] = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        reference_enthalpy: Float[Array, ""] = (
            self._H_over_RT(cp_coeffs_for_index, b1_for_index, TEMPERATURE_REFERENCE)
            * GAS_CONSTANT
            * TEMPERATURE_REFERENCE
        )

        return reference_enthalpy

    def enthalpy_function(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets enthalpy function/increment.

        This is :math:`H-H^{\circ}(T_r)` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy increment in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        enthalpy: Float[Array, " T"] = self.enthalpy(temperature)
        reference_enthalpy: Float[Array, ""] = self.reference_enthalpy()

        return enthalpy - reference_enthalpy

    def entropy(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets entropy

        This is :math:`S^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        b2_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b2), index)
        entropy: Float[Array, " T"] = (
            self._S_over_R(cp_coeffs_for_index, b2_for_index, temperature) * GAS_CONSTANT
        )

        return entropy

    def gibbs_function(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets Gibbs energy function.

        This is :math:`-[G^\circ-H^{\circ}(T_r)]/T` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy function in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        gibbs: Float[Array, " T"] = (
            self.get_gibbs_over_RT(temperature) * GAS_CONSTANT * temperature
        )
        gibbs_function: Float[Array, " T"] = -(gibbs - self.reference_enthalpy()) / temperature

        return gibbs_function


@dataclass
class ThermodynamicDataSource:
    """Thermodynamic data source for all species"""

    data: pd.DataFrame
    """Thermodynamic data for all species"""

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            self.data = pd.read_csv(datapath, comment="#")

    @property
    def formula_column(self) -> str:
        """Name of the column that refers to the hill formula"""
        return "hill_formula"

    @property
    def state_column(self) -> str:
        """Name of the column that refers to the state of aggregation"""
        return "state"

    def available_species(self) -> tuple[str, ...]:
        """Available species

        Returns:
            Available species
        """
        df: pd.DataFrame = cast(
            pd.DataFrame, self.data[[self.formula_column, self.state_column]].drop_duplicates()
        )
        available_species: tuple[str, ...] = tuple(
            f"{getattr(row, self.formula_column)}_{getattr(row, self.state_column)}"
            for row in df.itertuples(index=False)
        )

        return available_species

    def create_dictionary(self) -> dict[str, ThermodynamicCoefficients]:
        """Dictionary of thermodynamic coefficients for all species

        Returns:
            Dictionary of thermodynamic coefficients for all species
        """
        unique_combinations: pd.DataFrame = cast(
            pd.DataFrame, self.data[[self.formula_column, self.state_column]].drop_duplicates()
        )
        coefficient_dict: dict[str, ThermodynamicCoefficients] = {}

        for row in unique_combinations.itertuples(index=False):
            hill_formula: str = str(getattr(row, self.formula_column))
            state: str = str(getattr(row, self.state_column))
            name: str = f"{hill_formula}_{state}"

            # Find all data across all temperature ranges
            df: pd.DataFrame = cast(
                pd.DataFrame,
                self.data[
                    (self.data[self.formula_column] == hill_formula)
                    & (self.data[self.state_column] == state)
                ],
            )
            cp_coeffs: pd.DataFrame | pd.Series = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]]
            coefficient_dict[name] = ThermodynamicCoefficients(
                df["b1"], df["b2"], cp_coeffs, df["T_min"], df["T_max"]
            )

        return coefficient_dict


class CriticalData(eqx.Module):
    """Critical temperature and pressure of a gas species

    Args:
        temperature: Critical temperature in K
        pressure: Critical pressure in bar
    """

    temperature: float = eqx.field(converter=float, default=1)
    """Critical temperature in K"""
    pressure: float = eqx.field(converter=float, default=1)
    """Critical pressure in bar"""


@dataclass
class CriticalDataSource:
    """Critical data source for all species"""

    data: pd.DataFrame
    """Critical data for all species"""

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(CRITICAL_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            self.data = pd.read_csv(datapath, comment="#")

    @property
    def name_column(self) -> str:
        """Name of the column that refers to the hill formula and an optional suffix"""
        return "name"

    @property
    def critical_temperature_column(self) -> str:
        """Name of the column that refers to the critical temperature in K"""
        return "Tc"

    @property
    def critical_pressure_column(self) -> str:
        """Name of the column that refers to the critical pressure"""
        return "Pc"

    def create_dictionary(self) -> dict[str, CriticalData]:
        """Dictionary of critical data for all species

        Returns:
            Dictionary of critical data for all species
        """
        critical_dict: dict[str, CriticalData] = {}

        for row in self.data.itertuples(index=False):
            name: str = str(getattr(row, self.name_column))
            critical_dict[name] = CriticalData(
                temperature=float(getattr(row, self.critical_temperature_column)),
                pressure=float(getattr(row, self.critical_pressure_column)),
            )

        return critical_dict


# Create dictionaries of instantiated data (JAX-compliant Pytrees) that we can use for lookup.
# It should also be net faster to create these data once and then access (potentially many times).
# These are also set to private to avoid sphinx (autodoc) from printing long strings.
thermodynamic_data_source: ThermodynamicDataSource = ThermodynamicDataSource()
"""Thermodynamic data source

:meta private:
"""
thermodynamic_coefficients_dictionary: dict[str, ThermodynamicCoefficients] = (
    thermodynamic_data_source.create_dictionary()
)
"""Thermodynamic coefficients dictionary

:meta private:
"""
critical_data_source: CriticalDataSource = CriticalDataSource()
"""Critical data source

:meta private:
"""
critical_data_dictionary: dict[str, CriticalData] = critical_data_source.create_dictionary()
"""Critical data dictionary

:meta private:
"""


class ChemicalSpeciesData(eqx.Module):
    """Individual species data

    Args:
        formula: Formula
        state: State of aggregation as defined by JANAF
    """

    formula: str
    """Formula"""
    state: str
    """State of aggregation"""
    thermo: ThermodynamicCoefficients
    """Thermodynamic coefficient and methods"""
    composition: ImmutableMap[str, tuple[int, float, float]]
    """Composition"""
    hill_formula: str
    """Hill formula"""
    molar_mass: float = eqx.field(converter=float)
    """Molar mass"""
    miscibility: bool = False
    """Mix of H2-H2O"""
    mole_frac_H2 : Optional[float] = None

    def __init__(self, formula: str, state: str, miscibility: bool,mole_frac_H2: Optional[float] = None):
        self.formula = formula
        self.state = state
        mformula: Formula = Formula(self.formula)
        self.composition = ImmutableMap(mformula.composition().asdict())
        self.hill_formula = mformula.formula
        self.molar_mass = mformula.mass * unit_conversion.g_to_kg
        self.miscibility = miscibility
        self.mole_frac_H2 = mole_frac_H2
        if not miscibility:
            try:
                self.thermo = thermodynamic_coefficients_dictionary[self.name]
            except KeyError:
                raise KeyError(
                    f"{self.name} not available. "
                    f"Available species are {thermodynamic_data_source.available_species()}"
                )
        else:
            # The Gibbs values will be overwritten in get_gibbs_over_RT()
            self.thermo = ChemicalSpeciesData("HO",'g',miscibility=False).thermo # Coincidence that NIST JANAF has real species 'HO'
            
            ## Changing composition by taking into account mole fracions of constituent gasses (H2, H2O)
            base_comp = dict(ChemicalSpeciesData("HO",'g',miscibility=False).composition) # copy of self.composition
            
            # jax.debug.print("{}", base_comp['H'])
            ## PROBLEM: at second iteration, no H2 and H2O anymore
            mole_frac_H2O = 1 - mole_frac_H2  # type: ignore
            # print('mole_frac_H2', mole_frac_H2)
            x_H = 2 * mole_frac_H2 + 2 * mole_frac_H2O  # type: ignore
            x_O = mole_frac_H2O # type: ignore
            ## Hard to derive mole fraction H2,H2O from mass_H, mass_O because also have O2, choice very arbitrarily
            # EARTH_MASS = 5.972e24 # 1 Earth mass (kg)
            # planet_mass = 5 * EARTH_MASS 
            # x_H = mass_constraints["H"]/planet_mass # type: ignore
            # x_O = mass_constraints["O"]/planet_mass # type: ignore
            # x_H = 4
            # x_O = 1
            n, a, b = base_comp["H"]
            base_comp["H"] = (n * x_H, a * x_H, b * x_H) # type: ignore
            n, a, b = base_comp["O"]
            base_comp["O"] = (n * x_O, a * x_O, b * x_O) # type: ignore

            self.composition = ImmutableMap(base_comp) # replace self.composition
            print(self.composition)

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and state of aggregation"""
        return f"{self.hill_formula}_{self.state}"

    def get_gibbs_over_RT(self, temperature: ArrayLike, pressure: Optional[ArrayLike], mole_frac_H2: Optional[Float] = None) -> Array:
        """Gets Gibbs energy over RT. For the miscible phase of H and H2O, the Gibbs energy of 
        mixing is calculated according to Gupta et al. 2025 A.3 {Equation A3}

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy over RT
        """
        gibbs_over_RT = self.thermo.get_gibbs_over_RT(temperature)

        if self.miscibility:
            # if self.formula == 'H4O':
            #     # Asumptions: Equal ammount of moles: 50% H2, 50% H2O
            #     # Pressure estimate comes from no miscibilty case
            #     LAMBDA = 2.62 + (-0.68)/(temperature/1000) 
            #     X = LAMBDA / (1+ LAMBDA) # critical composition
            #     # jax.debug.print("x = {}, temperature {}", X, temperature)
            #     # Y = X / (X + LAMBDA*(1-X)) # y in Gupta et al. 2025
            #     Y = X # Using X in equations below (bcs Y always defined as 0.5)
            #     # jax.debug.print("Y = {}, temperature {}", Y, temperature)
            #     pressure = 36 # (GPa) output of atmodelle for O-H system in same conditions (WITH miscibility, ideal gasses)
            #     # pressure = 72 # (GPa) output of atmodelle for O-H system in same conditions (WITH miscibility, real gasses)
                
            #     print('INFO | Calculating Gibbs free energy of mixing between H2 and H2O')
            #     gibbs_over_RT_pure: Float[Array, " T"] = ChemicalSpeciesData('H2', 'g',False).thermo.get_gibbs_over_RT(temperature) * Y + ChemicalSpeciesData('H2O', 'g',False).thermo.get_gibbs_over_RT(temperature) * (1-Y)
            #     # jax.debug.print("gibbs pure = {}, temperature {}", gibbs_over_RT_pure, temperature)
                
            #     gibbs_idealmix: Float[Array, " T"] = jnp.array(Y*jnp.log(Y)+(1-Y)*jnp.log(1-Y), float)
            #     # jax.debug.print("gibbs_idealmix = {}, temperature {}", gibbs_idealmix, temperature)
            #     W_H = -599.08
            #     W_S = -16.08 
            #     W_V = -26.12 + 981.78/(temperature/1000)**2
            #     # jax.debug.print("W_H={}, T*W_S={}, P*W_V={}", W_H, temperature*W_S, pressure*W_V)
            #     W = W_H - temperature*W_S + pressure*W_V
            #     gibbs_excess: Float[Array, " T"] = jnp.array(W*Y*(1-Y), float)
            #     jax.debug.print('Gibbs energy of mixing is {}, temperature {}', gibbs_idealmix + gibbs_excess/(GAS_CONSTANT * temperature),temperature)
            #     gibbs_over_RT: Float[Array, " T"] = gibbs_over_RT_pure + gibbs_idealmix + gibbs_excess/(GAS_CONSTANT * temperature) # unit: (J/mol/K) / (J/mol/K) so unitless
            #     # jax.debug.print('Gibbs_over_RT H4O {out}', out=gibbs_over_RT)

            #     # Printing Gibbs energy of reacion H2 + H2O -> H4O
            #     G_H4O = gibbs_over_RT
            #     G_H2 = ChemicalSpeciesData('H2', 'g',False).thermo.get_gibbs_over_RT(temperature)
            #     G_H2O = ChemicalSpeciesData('H2O', 'g',False).thermo.get_gibbs_over_RT(temperature)
            #     # jax.debug.print("G_H2 = {}, temperature {}",G_H2, temperature)
            #     # jax.debug.print("G_H2O = {}, temperature {}",G_H2O, temperature)
            #     # jax.debug.print("G_H4O = {}, temperature {}",G_H4O, temperature)
            #     jax.debug.print("gibbs_over_RT reaction = {}, temperature {}, y {}",G_H4O-G_H2-G_H2O, temperature, Y)
            #     jax.debug.print("gibbs_over_RT reaction Y corrected = {}, temperature {}, y {}",G_H4O-Y*G_H2-(1-Y)*G_H2O, temperature, Y)
            if self.formula == 'HO':
                print('INFO | Calculating Gibbs energy of mixing')
                # jax.debug.print('pressure is {}', pressure) 
                pressure_GPa = pressure/1e4 # type: ignore
                x = 0.99 #TODO from user input 'mole_fractions’
                # x = mole_frac_H2 # type: ignore
                # jax.debug.print('mole fraction H2 is {}', x)
                G_H2 = ChemicalSpeciesData('H2', 'g',False).thermo.get_gibbs_over_RT(temperature)
                G_H2O = ChemicalSpeciesData('H2O', 'g',False).thermo.get_gibbs_over_RT(temperature)
                gibbs_over_RT_pure = x*G_H2 + (1-x)*G_H2O # type: ignore

                LAMBDA = 2.62 + (-0.68)/(temperature/1000) 
                Y = x / (x + LAMBDA*(1-x)) # type: ignore
                gibbs_idealmix: Float[Array, " T"] = jnp.array(Y*jnp.log(Y)+(1-Y)*jnp.log(1-Y), float)
                # jax.debug.print("gibbs_idealmix = {}, temperature {}", gibbs_idealmix, temperature)
                W_H = -599.08
                W_S = -16.08 
                W_V = -26.12 + 981.78/(temperature/1000)**2
                # jax.debug.print("W_H={}, T*W_S={}, P*W_V={}", W_H, temperature*W_S, pressure*W_V)
                W = W_H - temperature*W_S + pressure_GPa*W_V
                gibbs_excess: Float[Array, " T"] = jnp.array(W*Y*(1-Y), float)
                # jax.debug.print('Gibbs energy of mixing is {}, temperature {}', gibbs_idealmix + gibbs_excess/(GAS_CONSTANT * temperature),temperature)
                
                # gibbs_over_RT: Float[Array, " T"] = gibbs_over_RT_pure + gibbs_excess/(GAS_CONSTANT * temperature)
                gibbs_over_RT: Float[Array, " T"] = gibbs_over_RT_pure + gibbs_idealmix + gibbs_excess/(GAS_CONSTANT * temperature)

                
        return gibbs_over_RT
