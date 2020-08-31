# chchanges
Detect statistically meaningful changes in streams of data via online changepoint detection.
---
[![Build Status](https://semaphoreci.com/api/v1/jonathanward/chchanges/branches/master/badge.svg)](https://semaphoreci.com/jonathanward/chchanges)

For example, we can detect changes in a stream of normally distributed data, where the mean of the distribution is piecewise constant:

### The stream of data
![mean_data_stream](chchanges/demos/mean_data_stream.gif)

### The evolving posterior distribution
![mean_posterior_distribution](chchanges/demos/mean_posterior_distribution.gif)

### The probability that a changepoint was detected
![mean_changepoint_probability](chchanges/demos/mean_changepoint_probability.gif)

To generate these figures, experiment with `chchanges/demos/mean.py`

### Contributing
- By adding other Posterior distributions and Hazard functions, you can fine-tune chchanges for your specific application.
- E.g. a Multivariate Student's T posterior would enable detecting changes in the correlation of multivariate data.
- Please contribute and expand the range of chchanges uses.

### References:
- [Ryan P. Adams, David J.C. MacKay, "Bayesian Online Changepoint Detection" (2007)](https://arxiv.org/abs/0710.3742)
- [Byrd, M Gentry et al. “Lagged Exact Bayesian Online Changepoint Detection with Parameter Estimation” (2017)](https://arxiv.org/abs/1710.03276)


### Other software implementations:
- [https://github.com/dtolpin/bocd](https://github.com/dtolpin/bocd) Particularly indebted to this implementation.
- [https://github.com/hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection)
- [https://github.com/lnghiemum/LEXO](https://github.com/lnghiemum/LEXO)

---
"turn and face the strange" - David Bowie
