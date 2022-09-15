# PEARL: Probabilistic Emission-Abating Reinforcement Learning
Original implementation of PEARL (Probabilsitic Emission Abating Reinforcement Learning), a building control algorithm
proposed in the paper [Low Emissions Building Control with Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2206.14191)
by [Scott R. Jeen](https://enjeeneer.io), [Alessandro Abate](https://www.cs.ox.ac.uk/people/alessandro.abate/)
and [Jonathan M. Cullen](http://www.eng.cam.ac.uk/profiles/jmc99) (2022).

<p align="center">
  <br><img src='/media/pearl.png' width="900"/><br>
   <a href="https://arxiv.org/abs/2206.14191">[Paper]</a>&emsp;
</p>

## Method
PEARL combines ideas from system identification and model-based RL to find emission-reducing control policies zero-shot, that is,
without access to a simulator or historical data _a priori_. In experiments across three varied building energy simulations, we
show PEARL outperforms an existing RBC once, and popular RL baselines in all cases, reducing building emissions by as
much as 31% whilst maintaining thermal comfort

<figure>
<p align="center">
  <img src='/media/emissions.jpg' width="600"/>
    <figcaption> Top: Cumulative emissions produced by all agents across the (a) Mixed Use, (b) Offices, and (c) Seminar Centre
environments. Curves represent the mean of 3 runs of the experiment, shaded areas are one standard deviation (too small to see
in all cases except PPO). Bottom: Mean daily building temperature produced by all agents, the green shaded area illustrates the
target temperature range [19, 24]. </figcaption>
</figure>


## Installation
First you'll need to install [Energym](https://github.com/bsl546/energym) for running the building energy simulations atop
 _EnergyPlus_. The full installation instructions can be found on their website [here](https://bsl546.github.io/energym-pages/sources/install_min.html).

Then you can install the required dependencies vai `conda`:
```commandline
conda env create --file env.yaml
conda activate PEARL
```
Then, to recreate the main results from the paper run
```commandline
python main.py
```
from the root directory. 
## Citation
If you found this work useful, or you use this project to inform your own research, please consider citing it with:
```commandline
@article{jeen2022,
  url = {https://arxiv.org/abs/2206.14191},
  author = {Jeen, Scott R. and Abate, Alessandro and Cullen, Jonathan M.},  
  title = {Low Emission Building Control with Zero Shot Reinforcement Learning},
  publisher = {arXiv},
  year = {2022},
}
```

## License 
This work licensed under CC-BY-NC, see `LICENSE.md` for further details.









