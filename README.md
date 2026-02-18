ProtARG: Alignment-Free Prediction of Antimicrobial Resistance from Protein Sequences Using Integrated Classical and Deep Representations
1. Summary:
                ProtARG is an alignment-free, protein-level antimicrobial resistance (AMR) prediction framework that integrates classical sequence-derived descriptors with deep contextual embeddings from pretrained protein language models. 
                Traditional AMR annotation approaches rely heavily on homology searches against curated resistance databases. While effective for close homologs, such alignment-based strategies often fail to generalize to divergent or previously uncharacterized resistance proteins.
                ProtARG addresses this limitation by combining biochemical feature engineering with transformer-based protein embeddings to enable scalable, database-independent AMR prediction. 
                The framework is modular, reproducible, and designed for large-scale AMR surveillance and discovery applications.

Three feature spaces were evaluated: 

                Classical sequence-derived descriptors.
                
                Deep embeddings from pretrained ESM models.
                
                Integrated feature representation (classical + deep).
                
2. Model Training:

   Multiple machine learning classifiers were evaluated: Logistic Regression, Support Vector Machines, Random Forest, Neural Network (Multi-Layer Perceptron).

   Three training configurations were tested: Classical-only, Deep embeddings-only, Integrated feature space.

   Threshold optimization was performed using MCC maximization on the validation set.

4. Evaluation

      Performance metrics: AUROC, Matthews Correlation Coefficient (MCC), F1-score. \n

      Final performance was reported on an independent held-out test dataset.

6. Repository Structure

  ProtARG/
  │
  ├── All_Sequences.fasta
  ├── Features.tar.xz
  ├── Trained_Models.tar.xz
  │
  ├── ProtARG_Train_Classical_Only.py
  ├── ProtARG_Train_DeepEmbeddings_Only.py
  ├── ProtARG_Training.py
  ├── feature_space_analysis.py
  │
  └── .gitattributes

7. File Descriptions

     All_Sequences.fasta: Complete curated dataset of 6,086 non-redundant protein sequences used in this study.

     Features.tar.xz: Compressed archive containing Classical feature matrices, Deep embedding feature matrices, Integrated feature representations.
   
     Trained_Models.tar.xz: Pretrained model corresponding to Logistic Regression, Support Vector Machines, Random Forest, and Neural Network

     ProtARG_Train_Classical_Only.py: Training pipeline using only classical sequence-derived descriptors.

     ProtARG_Train_DeepEmbeddings_Only.py: Training pipeline using only ESM-based embeddings.

     ProtARG_Training.py : Full integrated training pipeline combining classical and deep features.

8. Installation

     6.1. Clone Repository

           git clone https://github.com/DrRahulKaushik/ProtARG.git

           cd ProtARG

   6.2. Create Environment

        Recommended Python version: 3.9+

        conda create -n protarg python=3.9

        conda activate protarg

  6.3. Install Dependencies

          pip install numpy pandas scikit-learn xgboost lightgbm joblib

    For deep embeddings (if regenerating):
   
          pip install fair-esm torch
    
7. Running Training Pipelines

        7.1. Classical-only Training

             python ProtARG_Train_Classical_Only.py --train train.tsv --val val.tsv --test test.tsv --outdir classical_models

        7.2. Deep Embeddings-only Training

                  python ProtARG_Train_DeepEmbeddings_Only.py --train train.tsv --val val.tsv --test test.tsv --outdir deep_models

        7.3. Integrated Training

                  python ProtARG_Training.py --train train.tsv --val val.tsv --test test.tsv --outdir integrated_models

9. Reproducibility Checklist:

         Dataset curated and redundancy-reduced at 95% identity

         Stratified partitioning

         No identifier overlap across splits

         Feature integrity checks

         Fixed random seeds (42)

         Explicit threshold optimization

         Independent test set evaluation

11. Benchmarking: ProtARG was benchmarked against diverse AMR tools such as AMRFinderPlus, DeepARG, PLM-ARG, and ProtAlign-ARG

12. Citation: If you use ProtARG in your work, please cite:

Kaushik R, Re S (2026). ProtARG: Alignment-Free Prediction of Antimicrobial Resistance from Protein Sequences Using Integrated Classical and Deep Representations.
(Full citation to be added upon publication.)

License: Apache 2.0

Contact
Dr Rahul Kaushik,

AI Center for Health and Biomedical Research,

National Institutes of Biomedical Innovation, Health and Nutrition,

566-0002, Osaka, Japan
