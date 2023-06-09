# Examples for v1.0 of `Cleanlab`

**DISCLAIMER:** Examples here are specifically for the old v1.0 version of `cleanlab`.

A brief description of the files and folders:

- `imagenet`, 'cifar10', 'mnist' - code to find label errors in these datasets and reproduce the results in the [confident learning paper](https://arxiv.org/abs/1911.00068). You will also need to `git clone` [confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce).
- [imagenet_train_crossval.py](imagenet/imagenet_train_crossval.py) - a powerful script to train cross-validated predictions on ImageNet, combine cv folds, train with on masked input (train without label errors), etc.
- [cifar10_train_crossval.py](cifar10/cifar10_train_crossval.py) - same as above, but for CIFAR.

## License

Copyright (c) 2017-2022 Cleanlab Inc.

All files listed above and contained in this folder (<https://github.com/cleanlab/examples>) are part of cleanlab.

cleanlab is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cleanlab is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License in [LICENSE](/LICENSE).
