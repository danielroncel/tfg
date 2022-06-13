# Final Degree Project

Repository with the network formation models implemented during my Final Degree Project. It also includes the notebooks used to run the experiments and to create the plots shown in the document.

Due to storage limiation, this repository does not contain the data of all the experiments shown in the TFG document (more than 20 GB), but the results of an experiment used as example. However, with this repository all those experiments can be reproduced.

## Content

### Homophilic Barabási-Albert model (PAH model)

Implementation of the network formation model described in [Karimi et al., 2018]. This model was implemented for a better understanding of the model. The material related to this model is stored in folder *tfg/homophilic_ba*. It contains a single jupyter notebook with the model implementation, graph examples and some plots to study the effect of the parameters of the model in the disparate impact of the minority class.

### PAH without preferential attachment

Variation of the network formation model described in [Karimi et al., 2018] in which the preferential attachment mechanism has been removed. It corresponds to the first model explored in the TFG document. The material related to this model is stored in folder *tfg/homophilic_no_preferential_attachment*. It contains:

  *  model.py: Script with the implementation of the network formation model and other functions to ease the execution of the experiments.
  *  graphics.py: Script with functions to load and prepare the data to extract insights.
  *  run_experiments.ipynb: Notebook used to execute the experiments.
  *  create_plots.ipynb: Notebook used to create the plots.
  *  outputs/: folder with the data of an experiment used as example.

### PAP model

Network formation model inspired in [Karimi et al., 2018] and [Germano et al., 2019]. It corresponds to the second model explored in the TFG document. The material related to this model is stored in folder *tfg/homophilic_no_preferential_attachment*. It has the same structure as *tfg/homophilic_no_preferential_attachment*.

### Bibliography

[Karimi et al., 2018] Karimi, F., Génois, M., Wagner, C., Singer, P., and Strohmaier, M. (2018). Homophily influences ranking of minorities in social networks. Scientific reports, 8(1):1–12.

[Germano et al., 2019] Germano, F., Gómez, V., and Le Mens, G. (2019). The few-get-richer: A surprising consequence of popularity-based rankings. In The World Wide Web Conference, pages 2764–2770.
