# On Extended Long Short-term Memory and Dependent Bidirectional Recurrent Neural Network

This is the official implementation of extended LSTM (ELSTM) and dependent BRNN (BRNN) in paper: "On Extended Long Short-term Memory and Dependent Bidirectional Recurrent Neural Network"

## Dependencies
* Python 3.6
* Tensorflow==1.15.0

## Dataset
* [PTB Dataset](https://drive.google.com/file/d/1kS9Rola_lYy-r8MHqnuZOT775R1yGLkK/view?usp=sharing)

It is the PTB dataset for language modeling

Download the dataset, create a folder named data and unzip the files to it

## HOW TO

#Training

    ```python
    generate_sequence.py --mode=train
    ```

#Testing

    ```python
    generate_sequence.py --mode=inference
    ```

## ELSTM

<img src="ELSTM.png" width="800">

## DBRNN
<img src="DBRNN.png" height="800">

## Citations
If you find this work useful, please consider citing it.
```
@article{ELSTM,
  title={On Extended Long Short-term Memory and Dependent Bidirectional Recurrent Neural Network},
  author={Su, Yuanhang and Kuo, C.-C. Jay},
  journal={Neurocomputing},
  volume={356},
  month={Sep},
  year={2019},
  pages={151-161}
}
```
