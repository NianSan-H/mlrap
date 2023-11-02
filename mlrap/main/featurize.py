import os
import re
import warnings

import pandas as pd
import numpy as np
from CBFV import composition as cbfvComposition

from matminer.featurizers.utils.stats import PropertyStats
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.site import (AverageBondAngle, AverageBondLength,
                                       CoordinationNumber)
from matminer.featurizers.structure import (ChemicalOrdering,
                                            MaximumPackingEfficiency,
                                            SiteStatsFingerprint,
                                            StructuralHeterogeneity)
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Poscar
from xenonpy.descriptor import Compositions as xenonpyCompositions


def xenonpy_elem_feat(df, col_id="formula", elem_prop=None, **kwargs):
    """
    xenonpy element descriptor
    """
    composition = df[col_id].to_frame()

    # transform formula to XenonPy embedded information.
    def formula_to_element_count(formula):
        pattern = r"([A-Z][a-z]*)(\d*)"
        matches = re.findall(pattern, formula)
        element_count = {}
        for match in matches:
            element = match[0]
            count = match[1] if match[1] else "1"
            count = int(count)
            if element in element_count:
                element_count[element] += count
            else:
                element_count[element] = count
        return element_count
    composition[col_id] = composition[col_id].apply(formula_to_element_count)

    xenonpy_compositions = xenonpyCompositions(**kwargs)
    descriptors = xenonpy_compositions.transform(composition, target_col=col_id)
    
    return pd.concat([df, descriptors], axis=1)


def cbfv_feature(df, col_id="formula", target_id="target", elem_prop="oliynyk", **kwargs):
    """
    CBFV element featurize
    """
    df_old = df.copy().drop([col_id, target_id], axis=1)
    df.rename(columns = {col_id: "formula", target_id: "target"}, inplace=True)
    df, y, formula, _ = cbfvComposition.generate_features(
                                    df, elem_prop=elem_prop, **kwargs)
    df = pd.concat([formula, y, df, df_old], axis=1)
    df.rename(columns = {"formula": col_id, "target": target_id}, inplace=True)

    return df


def matminer_elem_feat(df, formula_col_id="formula", pbar=True, elem_prop=None,
                     preset_name="magpie", stats=["mean"]):
    """
    matminer element descriptor
    """
    compound_to_str = StrToComposition()
    df = compound_to_str.featurize_dataframe(df, col_id=formula_col_id, pbar=pbar)
    ele_prop = ElementProperty.from_preset(preset_name=preset_name)
    ele_prop.stats = stats
    ele_prop.featurize_dataframe(df, inplace=True, col_id='composition',
                        ignore_errors=True, pbar=pbar)

    return df.drop('composition', axis=1)


def matminer_struct_feat(df, col_id="structure", descriptors=None, elem_prop=None,
                         pbar=True, ignore_errors=True, **kwargs):
    """
    matminer structural descriptor
    """
    kwargs["inplace"] = False
    kwargs["pbar"] = pbar
    kwargs["ignore_errors"] = ignore_errors

    descriptors_dict = {
        "structural_heterogeneity": structural_heterogeneity, 
        "chemical_ordering": chemical_ordering, 
        "maximum_packing_efficiency": maximum_packing_efficiency,
        "coordination_number": coordination_number, 
        "average_bond_length": average_bond_length, 
        "average_bond_angle": average_bond_angle
    }
    if descriptors is None: descriptors = list(descriptors_dict.keys())
    if not all(item in descriptors_dict for item in descriptors):
        raise ValueError(f"Descriptors must be a subset of the list {list(descriptors_dict.keys())}")
    
    for key in descriptors:
        df = descriptors_dict[key](df, col_id=col_id, **kwargs)
        
    return df


def structural_heterogeneity(df, col_id="structure", weight="area", 
                stats=["minimum", "maximum", "range", "mean", "avg_dev"], **kwargs):
    """
    Structural Heterogeneity featurize
    """
    struct_hetero = StructuralHeterogeneity(weight=weight, stats=stats)
    df = struct_hetero.featurize_dataframe(df=df, col_id=col_id, **kwargs)

    if "inplace" in kwargs and kwargs["inplace"]:
        return None
    else: return df


def chemical_ordering(df, col_id="structure", shells=(1, 2, 3), weight="area", **kwargs):
    """
    Chemical Ordering featurize
    """
    chem_order = ChemicalOrdering(shells=shells, weight=weight)
    df = chem_order.featurize_dataframe(df=df, col_id=col_id, **kwargs)

    if "inplace" in kwargs and kwargs["inplace"]:
        return None
    else: return df


def maximum_packing_efficiency(df, col_id="structure", **kwargs):
    """
    Maximum Packing Efficiency featurize
    """
    max_pack_effic = MaximumPackingEfficiency()
    df = max_pack_effic.featurize_dataframe(df=df, col_id=col_id, **kwargs)

    if "inplace" in kwargs and kwargs["inplace"]:
        return None
    else: return df


def average_bond_length(df, col_id="structure", algo_name="VoronoiNN", **kwargs):
    """
    Average Bond Length featurize
    """
    env_algo = init_bond(df, col_id, algo_name)
    bond_length = AverageBondLength(env_algo)
    sspbl = SiteStatsFingerprint(site_featurizer=bond_length)
    df = sspbl.featurize_dataframe(df, col_id=col_id, **kwargs)

    if "inplace" in kwargs and kwargs["inplace"]:
        return None
    else: return df


def average_bond_angle(df, col_id="structure", algo_name="VoronoiNN", **kwargs):
    """
    Average Bond Angle featurize
    """
    
    env_algo = init_bond(df, col_id, algo_name)
    bond_angle = NewAverageBondAngle(env_algo)
    sspba = SiteStatsFingerprint(site_featurizer=bond_angle)
    df = sspba.featurize_dataframe(df, col_id=col_id, **kwargs)

    if "inplace" in kwargs and kwargs["inplace"]:
        return None
    else: return df


def init_bond(df, col_id, algo_name):
    env_algo = VoronoiNN() if algo_name=="VoronoiNN" else CrystalNN()
    for i in df.index:
        df.loc[i, col_id].add_oxidation_state_by_guess()
    return env_algo


def coordination_number(df, col_id="structure", preset="VoronoiNN",
        stats=["minimum", "maximum", "range", "mean", "avg_dev"], **kwargs):
    """
    Coordination Number featurize
    """
    coor_num = CoordinationNumber.from_preset(preset=preset)
    sspbnum = SiteStatsFingerprint(site_featurizer=coor_num, stats=stats)
    df = sspbnum.featurize_dataframe(df, col_id=col_id, **kwargs)
    if "inplace" in kwargs and kwargs["inplace"]:
        return None
    else: return df


def to_structure(df, file_path, structure_col="structure", 
                 material_col="material_id", file_type="cif"):
    """
    structure file to cif str
    """
    # check file type
    if file_type not in ["cif", "vasp"]:
        raise ValueError(
            f"The method does not support the file type you have selected --{file_type}"
        )

    structure_list = []
    structure_miss = []

    structure_files = [os.path.join(file_path, f"{material_id}.{file_type}") 
                       for material_id in df[material_col]]

    for structure_file in structure_files:
        try:
            structure_object = Structure.from_file(structure_file)
            structure_list.append(structure_object)
        except Exception as e:
            structure_miss.append(os.path.basename(structure_file))
            structure_list.append(None)
            warnings.warn(
                f"Error reading structure file {structure_file}: {str(e)}", UserWarning
            )

    df[structure_col] = structure_list

    if structure_miss:
        warning_message = (
            f"\nDue to the lack of structural files provided, the following "
            f"materials did not generate structural objects:\n"
        )
        warning_message += "\n".join([f"\t- {material}" for material in structure_miss])
        warnings.warn(warning_message, UserWarning)

    print(f"Your structure file has been added to column {structure_col}.")
    return df


def to_structfile(df, structure_col="structure", 
                  material_col="material_id", file_path=None, file_type="cif"):
    """
    pymatgen object to structure file
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'structure')
    os.makedirs(file_path, exist_ok=True)

    if file_type not in ["cif", "vasp"]:
        warnings.warn(
            f"\nThe {file_type} is not a supported file type, "
            "it will be automatically saved as a CIF file."
        )
        file_type = "cif"
        
    for _, row in df.iterrows():
        material_id = row[material_col]
        structure = row[structure_col]
        struct_file = os.path.join(file_path, f"{material_id}.{file_type}")

        writer = Poscar(structure) if file_type == "vasp" else CifWriter(structure)
        writer.write_file(struct_file)

    print(f"Your structure files have been saved in {file_path}")

    return pd.DataFrame([])


class NewAverageBondAngle(AverageBondAngle):
    def featurize(self, strc, idx):
        # Compute nearest neighbors of the indexed site
        np.seterr(invalid="ignore")
        nns = self.method.get_nn_info(strc, idx)
        if len(nns) == 0:
            raise IndexError("Input structure has no bonds.")
        center = strc[idx].coords

        sites = [i["site"].coords for i in nns]

        # Calculate bond angles for each neighbor
        bond_angles = np.empty((len(sites), len(sites)))
        bond_angles.fill(np.nan)
        for a, a_site in enumerate(sites):
            for b, b_site in enumerate(sites):
                if b == a:
                    continue
                dot = np.dot(a_site - center, b_site - center) / (
                    np.linalg.norm(a_site - center) * np.linalg.norm(b_site - center)
                )
                if np.isnan(np.arccos(dot)):
                    bond_angles[a, b] = bond_angles[b, a] = np.arccos(round(dot, 5))
                else:
                    bond_angles[a, b] = bond_angles[b, a] = np.arccos(dot)
        # Take the minimum bond angle of each neighbor
        minimum_bond_angles = np.nanmin(bond_angles, axis=1)

        return [PropertyStats.mean(minimum_bond_angles)]