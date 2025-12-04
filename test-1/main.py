import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf


def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = {}
    # средняя молекулярная масса молекулы
    descriptors["MolWt"] = Chem.Descriptors.MolWt(mol)
    # Вычислить взвешенную сумму сопоставленных свойств ADS
    descriptors["qed"] = Chem.Descriptors.qed(mol)
    # средняя молекулярная масса молекулы без учета атомов водорода
    descriptors["HeavyAtomMolWt"] = Chem.Descriptors.HeavyAtomMolWt(mol)
    #descriptors["FpDensityMorgan1"] = Chem.Descriptors.FpDensityMorgan1(mol)
    #descriptors["FpDensityMorgan2"] = Chem.Descriptors.FpDensityMorgan2(mol)
    #descriptors["FpDensityMorgan3"] = Chem.Descriptors.FpDensityMorgan3(mol)
    # Количество радикальных электронов, которые имеет молекула (ничего не говорит о состоянии спина)
    descriptors["NumRadicalElectrons"] = Chem.Descriptors.NumRadicalElectrons(mol)
    # Число валентных электронов, которое имеет молекула
    descriptors["NumValenceElectrons"] = Chem.Descriptors.NumValenceElectrons(mol)
    # Количество тяжелых атомов в молекуле
    descriptors["HeavyAtomCount"] = Chem.Lipinski.HeavyAtomCount(mol)
    # Количество гетероатомов
    descriptors["NumHeteroatoms"] = Chem.Lipinski.NumHeteroatoms(mol)
    # Количество вращающихся облигаций
    descriptors["NumRotatableBonds"] = Chem.Lipinski.NumRotatableBonds(mol)
    #descriptors["NumRingSystems"] = MolSurf._LabuteHelper(mol)
    #descriptors["RingCount"] = MolSurf.LabuteASA(mol)
    descriptors["GetNumAtoms"] = mol.GetNumAtoms()

    def toxicity(mol):
        if mol is None:
            return False
        # Вычисляем некоторые дескрипторы молекулы
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        # Определяем пороговые значения для каждого дескриптора
        mw_threshold = 500
        logp_threshold = 5
        tpsa_threshold = 100
        hbd_threshold = 5
        hba_threshold = 10

        # Сравниваем значения дескрипторов с пороговыми значениями
        if mw > mw_threshold:
            return True
        if logp > logp_threshold:
            return True
        if tpsa > tpsa_threshold:
            return True
        if hbd > hbd_threshold:
            return True
        if hba > hba_threshold:
            return True

        # Если все дескрипторы находятся в безопасных пределах, молекула не является токсичной
        return False

    descriptors["Toxicity"] = toxicity(mol)
    return descriptors


with open('example_input.csv', 'r') as input_file:
    data = csv.reader(input_file)
    next(data)
    output_data = []
    for row in data:
        descriptors = smiles_to_descriptors(row[0])
        output_data.append([descriptors[key] for key in descriptors])

with open('test_data.csv', 'w') as output_file:
    writer = csv.writer(output_file)
    headers = ["MolWt", "qed", "HeavyAtomMolWt",
               "NumRadicalElectrons", "NumValenceElectrons", "HeavyAtomCount", "NumHeteroatoms", "NumRotatableBonds",
               "NumRingSystems", "Toxicity"]
    writer.writerow(headers)
    for row in output_data:
        writer.writerow(row)