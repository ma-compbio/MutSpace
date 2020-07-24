# MutSpace

## Table of Contents 
1. [Introduction] (#introduction)
2. [Prerequisites] (#prerequisites)
3. [Usage] (#usage)
4. [Inputs and pre-processing] (#inputs-and-pre-processing)
5. [Outputs] (#outputs)

## Introduction 
MutSpace is a method aiming to address the computational challenge to consider large-scale sequence context for mutational signatures. It can effectively extract patient-specific features by jointly modeling mutations and patients in a latent high-dimensional space. As shown in the figure below, the input of MutSpace consists of somatic mutations and cancer patients, which are naturally connected by the fact that a somatic mutation is observed in one patient. The output of MutSpace is a set of vectors in latent high-dimensional space for mutations and cancer patients. Importantly, the similarity of those embedded vectors of mutations and cancer patients reflect the closeness of these entities. For example, patients' vectors with similar mutational landscapes tend to be close in the high-dimensional space, and mutations observed in one patient tend to be close to that patient in the latent space. The embeddings of patients reported from MutSpace can be used to various tasks, including cancer subtype identification, cancer patients clustering, or used as alternative mutational features extracted from patients as compared with traditional mutational spectrum/frequency-based method. See the [slides](https://drive.google.com/file/d/1pzsuH-5VayxSusziN9OKumZNy2VVwKB2/view?usp=sharing) presented at ISMB 2020.

## Prerequisites
MutSpace requires:
* Python (tested 3.7.3)
* Torch (tested 1.3.1)
* Numpy (tested 1.16.4)
* Tqdm (tested 4.32.1)
* Json (tested 2.0.9)
* H5py (tested 2.9.0)
* Argparse
* Random

## Usage
The main function to run MutSpace is 'train.py'. It usage is shown below:
    ```
    Run "python train.py"
    usage: train.py [-h] [--name NAME] [--data_path DATA_PATH]
                    [--ring_num RING_NUM] [--ring_width RING_WIDTH]
                    [--margin MARGIN] [--max_norm MAX_NORM] [--emb_dim EMB_DIM]
                    [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                    [--n_negative N_NEGATIVE] [--debug] [--temp TEMP] [--seed SEED]

    The following arguments are mandatory:
      --name        Name of the run. A forlder with this name will be created in the directory where this script is being executed
      ---data_path  Path to the folder containing mutation data
    The following arguments are optional:
      --ring_num    The number of sub-components of sequence context [default is 6]
      --ring_width  The length of each sub-component of seuqnce context [1]
      --margin      The constant margin in the hinge loss function [1.0]
      --max_nrom    The max norm of embeddings [10.0]
      --emb_dim     The dimension of embeddings [200]
      --epochs      Number of epochs for training [50]
      --batch_size  Batch size [4096]
      --lr          Learning rate [1e-3]
      --n_negative  Number of negative samples generated per positve sample [15]
      --temp        Normalization parameter for calculation of similarity [1.0]
      --seed        Random seed [888]
    ```

## Inputs and pre-processing
MutSpace currently only supports single nucleotide variants (SNVs), including C to A(`C->A`), C to G (`C->G`), C to T (`C->T`), T to A (`T->A`), T to C (`T->C`), and T to G (`T->G`) mutation. Mutations with reference bases as G or A need to be converted into C or T by reverse-complement mutations (e.g., `G->T` will become `C->A`). Notably, the sequence context may also need to convert to its reverse complement as we always need the upstream and downstream sequence of the mutation. We also highly recommend removing mutations located in the protein-coding regions as the sequence context of those mutations may be much different from non-coding somatic mutations and thus bias the training process.

Create a folder and put at least one TSV file (with suffix .tsv) formatted as below (line start with # will be skipped). 
    ```
    #chrom  start   end     patient_id      var_id  ref     alt     strand  var_type        upstream        downstream  cancer_id       bin_num
    chr1    755373  755374  DO225368        ICGC-BRCA-EU_DO225368_246       C       A       -       C->A    GTAGATAGGGTGGAT TGCTGGCCACGCAGG ICGC-BRCA-EU    bin_0
    chr1    755680  755681  DO224963        ICGC-BRCA-EU_DO224963_247       C       T       -       C->T    GGTGGATCTGCTGGA AGGCAGGTAGTATAG ICGC-BRCA-EU    bin_0
    ```

In the same folder, create a JSON file named __meta.json__. This file is the configuration file telling script how to extract information from the mutation file. You can modify the format of the mutation TSV file, just make sure to modify this JSON file and set the right column index (start from 0) for each entry.
    ```
    {
        "uid": 3,
        "upstream": 9,
        "downstream": 10,
        "var_type": 8,
        "a": [12],
        "b": [3, 11]
    }
* __uid__:
The column index (start from 0) for patient id. Note that each patient can only have one patient id. In the above example, uid refers to the 4th column `patient_id`.
* __upstream__:
The column index (start from 0) for the upstream sequence context (from 5' to 3') of the mutation. There is no limit for the length of the sequence except that it must be longer than the product of the number of subcomponents (`--ring_num`) and the length of subcomponent (`--ring_width`). If the reference allele is G or A, do not forget to use the reverse complement of mutation and its sequence context. In that case, the upstream still refers to the upstream (5`) of mutation not the sequence with smaller coordinates on the reference genome. In the above example, upstream refers to the 10th column `upstream'.
* __downstream__:
The column index (start from 0) for the downstream sequence context (from 5' to 3') of the mutation. In the above example, downstream refers to the 11th column `downstream`.
* __var_type__:
The column index (start from 0) for the variation type from the six substitutions: `C->A`, `C->G`, `C->T`, `T->A`, `T->C`, and `T->G`. In the above example, var_type refers to the 9th column `var_type`.
* __a__:
The column index list for additional mutations' features except for sequence context. In the above example, this refers to the 13th `bin_num`. Here we assign mutations into a 1Mb bin and give each bin a unique ID. If there is no additional features for mutations, just leave it as blank (e.g., `"a": []`)
* __b__:
The column index list for patients' features. In the above example, this refers to the 4th column patient id and the 12th column cancer type label (`cancer_id`). 

## Outputs
After the training is finished, a folder named `ckpt` should be created in the directory where train.py is being executed. A folder with name specified by the parameter `--name`` contains the output of MutSpace. You should see multiple check point files with suffix .pth. Each check point file is a model file that can be loaded using torch (e.g., `torch.load("49.pth", map_location = 'cpu')`). The file name indicates that file is captured during which epoch. For example, `0.pth` means the first epoch, and `49.pth` means the 50th epoch. Each check point file is actually a big 2D matrix with each row as an embedding of features including patient's id, cancer type, sub-components, etc.

There are also four outputs, feature_dict.json, patient_mapping.json, setting.json, and record.txt. 
* __feature_dict.json__:
This JSON file contains the relationship that maps feature to the index of row in the check point file. For example, `"DO225368": 96, "bin_0": 97, ...` means that the patient's id embeddings for patient DO225368 is the 97th row of the 2D matrix in the check point file, and the embedding of the mutation's bin with bin id of bin_0 is the 98th row of the 2D matrix in the check point file.
* __patient_mapping.json__:
This JSON file contains important information about patients' features and how to get the final patients' embedding. For example, `"DO225368": [96, [15, 97], [96, 98]]` means that this patient has patient id DO225368. He or she has two features one is patient id (denoted by 96) and another is cancer type denoted by 98. To get the final embedding of this patient you just need to sum the embedding of patient's id and cancer type, and further normalized by sqrt(2). We use 2 because each patient has two features.
* __setting.json__:
This JSON file contains the parameters used for this run.

## Cite

If you want to cite our work, please use the following information

```
@article{zhang2020cancer,
  title={Cancer mutational signatures representation by large-scale context embedding},
  author={Zhang, Yang and Xiao, Yunxuan and Yang, Muyu and Ma, Jian},
  journal={Bioinformatics},
  volume={36},
  number={Supplement\_1},
  pages={i309--i316},
  year={2020},
  publisher={Oxford University Press}
}
```

## Contact
yangz6 at andrew.cmu.edu
jianma at andrew.cmu.edu
