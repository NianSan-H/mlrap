def copy_config(source_file, copy_file):
    """
    Copy the source file corresponding to the provided file path.
    """
    with open(source_file, 'r', encoding='utf-8') as src_file:
        file_contents = src_file.read()
    with open(copy_file, 'w', encoding='utf-8') as dest_file:
        dest_file.write(file_contents)


def descriptor(df, fea_type, col_id):
    """
    Generate feature descriptors.
    """
    from pymatgen.core import Structure
    from mlrap.main.featurize import (xenonpy_elem_feat, 
                                      cbfv_feature, 
                                      matminer_elem_feat, 
                                      matminer_struct_feat)
    fea_dict = {
        "xenonpy": xenonpy_elem_feat, 
        "oliynyk": cbfv_feature, 
        "jarvis": cbfv_feature, 
        "magpie": cbfv_feature, 
        "mat2vec": cbfv_feature, 
        "onehot": cbfv_feature, 
        "random_200": cbfv_feature, 
        "deml": matminer_elem_feat,
        "matminer": matminer_elem_feat,
        "matscholar_el": matminer_elem_feat,
        "megnet_el": matminer_elem_feat,
        "structure": matminer_struct_feat, 
    }

    if fea_type == "structure":
        df["structure"] = df[col_id].apply(lambda x: Structure.from_str(x, fmt="cif"))
        col_id = "structure"

    df = fea_dict[fea_type](df, col_id)
    
    if fea_type == "structure":
        df["structure"] = df["structure"].apply(lambda x: x.to(fmt="cif"))
    return df


def build_name(df)->list:
    """
    Construct a new name without special characters for df, and return a list as a result.
    """
    cols = []
    for i in range(len(df.columns)):
        col = df.columns[i]
        for j in range(len(col)):
            if col[j] == '_' and (col[j + 1] == '(' or col[j + 1] == '['):
                break
        col = col[:j+1]
        col = col.replace("_", " ")
        col = col.replace(":", " ")
        if col.endswith(' '):
            col = col.rstrip()
        cols.append(col)

    return cols