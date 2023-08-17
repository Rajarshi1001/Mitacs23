# Mitacs Globalink Research Internship 2023

This repository contains all the work that is being done during the Mitacs GRI'23. The project was mostly based on implementation of __State of the Art__ __Graph Neural Networks__ for prediction of various material properties as well as generation of nouvel materials which seems to be possible via intensive __Density Functional Theory (DFT) Calculations__. The graph neural networks used here includes a custom GNN, __MEGNet-16__ and __CGCNN__. 

The repository includes various models from feedforward nets to sequence models for trying out bandgap prediction of __Zincblend__ and __rocksalt__ strcutures__ using features like _molecule properties (local representations)_ as well as _distributed representations (like embeddings)_. 

This also includes the __XTB__ dataset for property predicion from __Morgan fingerprints__ of organic molecules.

Every folder has a bash script called __run.sh__:

```bash
chmod +x run.sh
./run.sh
```



