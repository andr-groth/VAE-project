# VAE project

![png](img/model_overview.png)

Python module for the implementation of a variational autoencoder (VAE) for climate data. The VAE is a generative model that can be used to learn the underlying distribution of a dataset and to generate new samples from it.

The present methodology extends on the VAE by adding a second decoder to the model. The second decoder is trained to make predictions about the future evolution of the data from the latent space. The VAE is trained to learn the distribution of the data and the prediction decoder is trained to make predictions about the future distribution of the data.


## Implementation
The modeling framework is published in [Groth and Chavez (2024)](https://doi.org/10.1007/s00382-024-07162-w). For an implementation see the corresponding Jupyter notebooks at:

> <https://github.com/andr-groth/VAE-ENSO-emulator>

## Simple examples

To get started, see the following examples:

- [VAE](example_VAE.md): Build a variational autoencoder model.

- [VAEp](example_VAEp.md): Build a variational autoencoder model with a prediction decoder.

For more examples, see the [collection of examples](examples.md).

## Reference
Please add a reference to the following paper if you use parts of this code:

```
@Article{Groth.Chavez.2024,
  author           = {Groth, Andreas and Chavez, Erik},
  journal          = {Climate Dynamics},
  title            = {Efficient inference and learning of a generative model for {ENSO} predictions from large multi-model datasets},
  year             = {2024},
  doi              = {10.1007/s00382-024-07162-w},
  publisher        = {Springer Science and Business Media LLC},
}
```
